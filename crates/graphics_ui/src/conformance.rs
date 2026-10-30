use gpui::{AppContext as _, Focusable, TestAppContext};
use project::{FakeFs, Project};
use std::path::Path;

use crate::{graph::GraphicsWorkspaceItem, session::GraphicsSession};

fn init(cx: &mut TestAppContext) {
    cx.update(|cx| {
        assets::Assets.load_test_fonts(cx);
        let store = settings::SettingsStore::test(cx);
        cx.set_global(store);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        editor::init(cx);
    });
}

#[gpui::test]
async fn at_graphite_add_workspace_action_opens_editable_result(cx: &mut TestAppContext) {
    init(cx);
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        let app_state = workspace::AppState::test(cx);
        workspace::AppState::set_global(app_state, cx);
        crate::graph::init(cx);
    });
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(Path::new("/project"), serde_json::json!({}))
        .await;
    let project = Project::test(fs, [Path::new("/project")], cx).await;
    let (host, cx) =
        cx.add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
    let workspace = host.read_with(cx, |host, _| host.workspace().clone());
    assert!(workspace.read_with(cx, |workspace, cx| workspace.active_item(cx).is_none()));
    cx.dispatch_action(crate::graph::OpenGraphite);
    cx.run_until_parked();
    let item = workspace
        .read_with(cx, |workspace, cx| {
            workspace.active_item_as::<GraphicsWorkspaceItem>(cx)
        })
        .expect("the registered action opens the Graphite workspace item");
    let editor = item.read_with(cx, |item, _| item.addend_editor.clone());
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("8");
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        assert_eq!(item.session.read(cx).result(), 10.0)
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-10").is_some());
}

#[gpui::test]
fn at_graphite_add_text_edit_updates_visible_result(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let editor = item.read_with(cx, |item, _| item.augend_editor.clone());
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("11");
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.inputs(), (11.0, 3.0));
        assert_eq!(session.result(), 14.0);
        assert!(session.diagnostic().is_none());
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-14").is_some());
}

#[gpui::test]
fn at_graphite_add_invalid_input_preserves_committed_revision(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let editor = item.read_with(cx, |item, _| item.addend_editor.clone());
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("invalid", window, cx)
    });
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.inputs(), (2.0, 3.0));
        assert_eq!(session.result(), 5.0);
        assert_eq!(session.revision(), 0);
        assert_eq!(session.diagnostic(), Some("Enter a number for addend."));
    });
    assert!(cx.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
    editor.update_in(cx, |editor, window, cx| editor.set_text("-2", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result(), 0.0);
        assert_eq!(session.revision(), 1);
        assert!(session.diagnostic().is_none());
    });
}

#[gpui::test]
fn at_graphite_add_items_keep_separate_sessions(cx: &mut TestAppContext) {
    init(cx);
    let (first, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let second = cx.update(|window, cx| cx.new(|cx| GraphicsWorkspaceItem::new(window, cx)));
    let editor = first.read_with(cx, |item, _| item.addend_editor.clone());
    editor.update_in(cx, |editor, window, cx| editor.set_text("10", window, cx));
    cx.run_until_parked();
    first.read_with(cx, |item, cx| {
        assert_eq!(item.session.read(cx).result(), 12.0)
    });
    second.read_with(cx, |item, cx| {
        assert_eq!(item.session.read(cx).result(), 5.0);
        assert_eq!(item.session.read(cx).revision(), 0);
    });
}

#[test]
fn at_graphite_add_invalid_augend_preserves_result() {
    let mut session = GraphicsSession::new(2.0, 3.0);
    for text in ["", "wrong", "3x"] {
        session.update_inputs(text, "3");
        assert_eq!(session.inputs(), (2.0, 3.0));
        assert_eq!(session.result(), 5.0);
        assert_eq!(session.revision(), 0);
    }
    session.update_inputs("2", "3");
    assert_eq!(session.revision(), 0);
    assert!(session.diagnostic().is_none());
}

#[gpui::test]
fn at_graphite_add_list_text_edit_updates_broadcast_result(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-LISTS")
        .expect("list mode control is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[5.0]").is_some());
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    augend.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("1, -2, 0");
    cx.run_until_parked();
    let addend = item.read_with(cx, |item, _| item.addend_editor.clone());
    addend.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("10, 20");
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.mode(), crate::session::AddMode::Lists);
        assert_eq!(
            session.list_values(),
            Some((
                &[1.0, -2.0, 0.0][..],
                &[10.0, 20.0][..],
                &[11.0, 18.0, 20.0][..]
            ))
        );
        assert!(session.diagnostic().is_none());
    });
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-[11.0, 18.0, 20.0]")
            .is_some()
    );
}

#[gpui::test]
fn at_graphite_add_list_empty_input_and_invalid_recovery(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-LISTS")
        .expect("list mode control is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    let addend = item.read_with(cx, |item, _| item.addend_editor.clone());
    augend.update_in(cx, |editor, window, cx| editor.set_text("", window, cx));
    cx.run_until_parked();
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[]").is_some());
    addend.update_in(cx, |editor, window, cx| editor.set_text("", window, cx));
    cx.run_until_parked();
    let revision = item.read_with(cx, |item, cx| item.session.read(cx).revision());
    augend.update_in(cx, |editor, window, cx| {
        editor.set_text("1, wrong", window, cx)
    });
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.list_values(), Some((&[][..], &[][..], &[][..])));
        assert_eq!(session.revision(), revision);
        assert_eq!(
            session.diagnostic(),
            Some("Enter comma-separated numbers for augend, or leave it blank.")
        );
    });
    assert!(cx.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[]").is_some());
    augend.update_in(cx, |editor, window, cx| editor.set_text("1, 2", window, cx));
    cx.run_until_parked();
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[]").is_some());
    addend.update_in(cx, |editor, window, cx| editor.set_text("5", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        assert_eq!(
            item.session.read(cx).list_values(),
            Some((&[1.0, 2.0][..], &[5.0][..], &[6.0, 7.0][..]))
        );
        assert!(item.session.read(cx).diagnostic().is_none());
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[6.0, 7.0]").is_some());
}

#[gpui::test]
fn at_graphite_add_list_mode_keeps_items_isolated(cx: &mut TestAppContext) {
    init(cx);
    let (first, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let second = cx.update(|window, cx| cx.new(|cx| GraphicsWorkspaceItem::new(window, cx)));
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-LISTS")
        .expect("list mode control is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let editor = first.read_with(cx, |item, _| item.addend_editor.clone());
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("10, 20", window, cx)
    });
    cx.run_until_parked();
    first.read_with(cx, |item, cx| {
        assert_eq!(
            item.session.read(cx).list_values(),
            Some((&[2.0][..], &[10.0, 20.0][..], &[12.0, 22.0][..]))
        );
    });
    second.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.mode(), crate::session::AddMode::Scalar);
        assert_eq!(session.inputs(), (2.0, 3.0));
        assert_eq!(session.result(), 5.0);
        assert_eq!(session.revision(), 0);
    });
}

#[test]
fn at_graphite_add_list_invalid_tokens_preserve_committed_values() {
    let mut session = GraphicsSession::new(2.0, 3.0);
    session.set_mode(crate::session::AddMode::Lists, "1, 2", "10");
    let revision = session.revision();
    for text in [",1", "1,", "1,,2", "1,  ", "1,wrong"] {
        session.update_inputs(text, "10");
        assert_eq!(
            session.list_values(),
            Some((&[1.0, 2.0][..], &[10.0][..], &[11.0, 12.0][..]))
        );
        assert_eq!(session.revision(), revision);
        assert!(session.diagnostic().is_some());
    }
    session.update_inputs("1, 2", "10");
    assert_eq!(session.revision(), revision);
    assert!(session.diagnostic().is_none());
    session.update_inputs("1, 2", "bad");
    assert_eq!(session.revision(), revision);
    assert_eq!(
        session.diagnostic(),
        Some("Enter comma-separated numbers for addend, or leave it blank.")
    );
}

#[gpui::test]
fn at_graphite_add_mode_switch_preserves_result_until_valid(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let lists = cx
        .debug_bounds("GRAPHITE-MODE-LISTS")
        .expect("list mode control is drawn");
    cx.simulate_click(lists.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let editor = item.read_with(cx, |item, _| item.augend_editor.clone());
    editor.update_in(cx, |editor, window, cx| editor.set_text("1, 2", window, cx));
    cx.run_until_parked();
    let revision = item.read_with(cx, |item, cx| item.session.read(cx).revision());
    let scalar = cx
        .debug_bounds("GRAPHITE-MODE-SCALAR")
        .expect("scalar mode control is drawn");
    cx.simulate_click(scalar.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.mode(), crate::session::AddMode::Scalar);
        assert_eq!(
            session.list_values(),
            Some((&[1.0, 2.0][..], &[3.0][..], &[4.0, 5.0][..]))
        );
        assert_eq!(session.revision(), revision);
        assert!(session.diagnostic().is_some());
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[4.0, 5.0]").is_some());
    editor.update_in(cx, |editor, window, cx| editor.set_text("7", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.inputs(), (7.0, 3.0));
        assert_eq!(session.result(), 10.0);
        assert_eq!(session.revision(), revision + 1);
        assert!(session.diagnostic().is_none());
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-10").is_some());
}

#[gpui::test]
fn at_graphite_add_integer_text_preserves_precision(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-INTEGER")
        .expect("integer mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let addend = item.read_with(cx, |item, _| item.addend_editor.clone());
    addend.update_in(cx, |editor, window, cx| editor.set_text("2", window, cx));
    cx.run_until_parked();
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    augend.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("9007199254740993");
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.mode(), crate::session::AddMode::Integer);
        assert_eq!(
            session.integer_values(),
            Some((9007199254740993, 2, 9007199254740995))
        );
        assert!(session.diagnostic().is_none());
    });
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-9007199254740995")
            .is_some()
    );
}

#[gpui::test]
fn at_graphite_add_vector_component_text_updates_visible_result(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-VECTOR")
        .expect("vector mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[5.0, 0.0]").is_some());
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    augend.update_in(cx, |editor, window, cx| editor.set_text("-5", window, cx));
    cx.run_until_parked();
    let augend_y = item.read_with(cx, |item, _| item.augend_y_editor.clone());
    augend_y.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("2");
    cx.run_until_parked();
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[-2.0, 2.0]").is_some());
    let addend_y = item.read_with(cx, |item, _| item.addend_y_editor.clone());
    addend_y.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("10");
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[-2.0, 12.0]");
        assert_eq!(
            session.expression_text(),
            "[-5.0, 2.0] + [3.0, 10.0] = [-2.0, 12.0]"
        );
        assert!(session.diagnostic().is_none());
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[-2.0, 12.0]").is_some());
}

#[gpui::test]
fn at_graphite_add_invalid_vector_component_preserves_whole_commit(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-VECTOR")
        .expect("vector mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let before = item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        (session.expression_text(), session.revision())
    });
    let addend_y = item.read_with(cx, |item, _| item.addend_y_editor.clone());
    addend_y.update_in(cx, |editor, window, cx| editor.set_text("bad", window, cx));
    cx.run_until_parked();
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    augend.update_in(cx, |editor, window, cx| editor.set_text("8", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!((session.expression_text(), session.revision()), before);
        assert_eq!(session.diagnostic(), Some("Enter a number for addend.y."));
    });
    assert!(cx.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[5.0, 0.0]").is_some());
    addend_y.update_in(cx, |editor, window, cx| editor.set_text("-4", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(
            session.expression_text(),
            "[8.0, 0.0] + [3.0, -4.0] = [11.0, -4.0]"
        );
        assert_eq!(session.revision(), before.1 + 1);
        assert!(session.diagnostic().is_none());
    });
}

#[gpui::test]
fn at_graphite_add_mixed_vector_modes_use_ordered_operands(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-SCALAR-VECTOR")
        .expect("scalar/vector mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let addend_y = item.read_with(cx, |item, _| item.addend_y_editor.clone());
    addend_y.update_in(cx, |editor, window, cx| editor.set_text("-4", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        assert_eq!(
            item.session.read(cx).expression_text(),
            "2 + [3.0, -4.0] = [5.0, -2.0]"
        );
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[5.0, -2.0]").is_some());
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-VECTOR-SCALAR")
        .expect("vector/scalar mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let augend_y = item.read_with(cx, |item, _| item.augend_y_editor.clone());
    augend_y.update_in(cx, |editor, window, cx| editor.set_text("-4", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.mode(), crate::session::AddMode::VectorScalar);
        assert_eq!(session.expression_text(), "[2.0, -4.0] + 3 = [5.0, -1.0]");
        assert!(session.diagnostic().is_none());
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[5.0, -1.0]").is_some());
}

#[cfg(debug_assertions)]
#[gpui::test]
fn at_graphite_add_integer_source_error_keeps_commit_and_recovers(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-INTEGER")
        .expect("integer mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let addend = item.read_with(cx, |item, _| item.addend_editor.clone());
    addend.update_in(cx, |editor, window, cx| editor.set_text("0", window, cx));
    cx.run_until_parked();
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    augend.update_in(cx, |editor, window, cx| {
        editor.set_text(i64::MAX.to_string(), window, cx)
    });
    cx.run_until_parked();
    let revision = item.read_with(cx, |item, cx| item.session.read(cx).revision());
    addend.update_in(cx, |editor, window, cx| editor.set_text("1", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.integer_values(), Some((i64::MAX, 0, i64::MAX)));
        assert_eq!(session.revision(), revision);
        assert_eq!(session.diagnostic(), Some("Node graph execution panicked"));
    });
    assert!(cx.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-9223372036854775807")
            .is_some()
    );
    addend.update_in(cx, |editor, window, cx| editor.set_text("-1", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.integer_values(), Some((i64::MAX, -1, i64::MAX - 1)));
        assert_eq!(session.revision(), revision + 1);
        assert!(session.diagnostic().is_none());
    });
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-9223372036854775806")
            .is_some()
    );
}

#[gpui::test]
fn at_graphite_add_typed_items_keep_sessions_isolated(cx: &mut TestAppContext) {
    init(cx);
    let (first, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let second = cx.update(|window, cx| cx.new(|cx| GraphicsWorkspaceItem::new(window, cx)));
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-INTEGER")
        .expect("integer mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let editor = first.read_with(cx, |item, _| item.addend_editor.clone());
    editor.update_in(cx, |editor, window, cx| editor.set_text("7", window, cx));
    cx.run_until_parked();
    second.update_in(cx, |item, _, cx| {
        item.set_mode(crate::session::AddMode::Vector, cx)
    });
    cx.run_until_parked();
    first.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.integer_values(), Some((2, 7, 9)));
        assert_eq!(session.mode(), crate::session::AddMode::Integer);
    });
    second.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(
            session.expression_text(),
            "[2.0, 0.0] + [3.0, 0.0] = [5.0, 0.0]"
        );
        assert_eq!(session.mode(), crate::session::AddMode::Vector);
    });
}

#[test]
fn at_graphite_add_integer_parse_and_typed_transition_keep_commit() {
    let mut session = GraphicsSession::new(2.0, 3.0);
    session.set_mode(crate::session::AddMode::Integer, "9007199254740993", "2");
    let revision = session.revision();
    for text in [
        "",
        "1.5",
        "1e3",
        "9223372036854775808",
        "-9223372036854775809",
        "bad",
    ] {
        session.update_inputs(text, "2");
        assert_eq!(
            session.integer_values(),
            Some((9007199254740993, 2, 9007199254740995))
        );
        assert_eq!(session.revision(), revision);
        assert!(session.diagnostic().is_some());
    }
    session.update_inputs("9007199254740993", "2");
    assert_eq!(session.revision(), revision);
    assert!(session.diagnostic().is_none());
    session.set_mode_components(crate::session::AddMode::Vector, "1", "bad", "2", "3");
    assert_eq!(
        session.integer_values(),
        Some((9007199254740993, 2, 9007199254740995))
    );
    assert_eq!(session.revision(), revision);
    assert_eq!(session.diagnostic(), Some("Enter a number for augend.y."));
    session.update_components("1", "-2", "2", "3");
    assert_eq!(
        session.expression_text(),
        "[1.0, -2.0] + [2.0, 3.0] = [3.0, 1.0]"
    );
    assert_eq!(session.revision(), revision + 1);
    assert!(session.diagnostic().is_none());
}

#[gpui::test]
fn at_graphite_add_integer_lists_edit_exact_ordered_values(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-INTEGER-LISTS")
        .expect("integer list mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    augend.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("9007199254740993, -2, 0");
    cx.run_until_parked();
    let addend = item.read_with(cx, |item, _| item.addend_editor.clone());
    addend.update_in(cx, |editor, window, cx| {
        editor.set_text("2, 10", window, cx)
    });
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(
            session.expression_text(),
            "[9007199254740993, -2, 0] + [2, 10] = [9007199254740995, 8, 10]"
        );
        assert!(session.diagnostic().is_none());
    });
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-[9007199254740995, 8, 10]")
            .is_some()
    );
}

#[gpui::test]
fn at_graphite_add_vector_lists_edit_paired_columns_and_broadcast(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-VECTOR-LISTS")
        .expect("vector list mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    let augend_y = item.read_with(cx, |item, _| item.augend_y_editor.clone());
    let addend = item.read_with(cx, |item, _| item.addend_editor.clone());
    let addend_y = item.read_with(cx, |item, _| item.addend_y_editor.clone());
    augend.update_in(cx, |editor, window, cx| {
        editor.set_text("1, -2, 0", window, cx)
    });
    cx.run_until_parked();
    augend_y.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("2, 3, 4");
    cx.run_until_parked();
    addend.update_in(cx, |editor, window, cx| {
        editor.set_text("10, 20", window, cx)
    });
    cx.run_until_parked();
    addend_y.update_in(cx, |editor, window, cx| editor.set_text("5, 6", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.expression_text(), "[[1.0, 2.0], [-2.0, 3.0], [0.0, 4.0]] + [[10.0, 5.0], [20.0, 6.0]] = [[11.0, 7.0], [18.0, 9.0], [20.0, 10.0]]");
        assert!(session.diagnostic().is_none());
    });
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-[[11.0, 7.0], [18.0, 9.0], [20.0, 10.0]]")
            .is_some()
    );
}

#[gpui::test]
fn at_graphite_add_mixed_lists_keep_operand_direction(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-SCALAR-VECTOR-LISTS")
        .expect("scalar/vector list mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    let augend_y = item.read_with(cx, |item, _| item.augend_y_editor.clone());
    let addend = item.read_with(cx, |item, _| item.addend_editor.clone());
    let addend_y = item.read_with(cx, |item, _| item.addend_y_editor.clone());
    augend.update_in(cx, |editor, window, cx| {
        editor.set_text("1, 2, 3", window, cx)
    });
    cx.run_until_parked();
    addend.update_in(cx, |editor, window, cx| {
        editor.set_text("10, 20", window, cx)
    });
    cx.run_until_parked();
    addend_y.update_in(cx, |editor, window, cx| editor.set_text("5, 6", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        assert_eq!(
            item.session.read(cx).expression_text(),
            "[1.0, 2.0, 3.0] + [[10.0, 5.0], [20.0, 6.0]] = [[11.0, 6.0], [22.0, 8.0], [23.0, 9.0]]"
        );
    });
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-VECTOR-SCALAR-LISTS")
        .expect("vector/scalar list mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    augend_y.update_in(cx, |editor, window, cx| {
        editor.set_text("4, 5, 6", window, cx)
    });
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.mode(), crate::session::AddMode::VectorScalarLists);
        assert_eq!(session.expression_text(), "[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]] + [10.0, 20.0] = [[11.0, 14.0], [22.0, 25.0], [23.0, 26.0]]");
        assert!(session.diagnostic().is_none());
    });
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-[[11.0, 14.0], [22.0, 25.0], [23.0, 26.0]]")
            .is_some()
    );
}

#[gpui::test]
fn at_graphite_add_vector_list_column_errors_preserve_whole_commit(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-VECTOR-LISTS")
        .expect("vector list mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let before = item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        (session.expression_text(), session.revision())
    });
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    let augend_y = item.read_with(cx, |item, _| item.augend_y_editor.clone());
    augend.update_in(cx, |editor, window, cx| editor.set_text("1, 2", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!((session.expression_text(), session.revision()), before);
        assert_eq!(
            session.diagnostic(),
            Some("Enter the same number of x and y components for augend.")
        );
    });
    augend_y.update_in(cx, |editor, window, cx| {
        editor.set_text("4, bad", window, cx)
    });
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!((session.expression_text(), session.revision()), before);
        assert_eq!(
            session.diagnostic(),
            Some("Enter comma-separated numbers for augend.y, or leave it blank.")
        );
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[[5.0, 0.0]]").is_some());
    assert!(cx.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
    augend_y.update_in(cx, |editor, window, cx| editor.set_text("4, 5", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[[4.0, 4.0], [5.0, 5.0]]");
        assert_eq!(session.revision(), before.1 + 1);
        assert!(session.diagnostic().is_none());
    });
    augend.update_in(cx, |editor, window, cx| editor.set_text("", window, cx));
    cx.run_until_parked();
    assert!(cx.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
    augend_y.update_in(cx, |editor, window, cx| editor.set_text("", window, cx));
    cx.run_until_parked();
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[]").is_some());
    item.read_with(cx, |item, cx| {
        assert!(item.session.read(cx).diagnostic().is_none())
    });
}

#[cfg(debug_assertions)]
#[gpui::test]
fn at_graphite_add_late_integer_list_error_never_publishes_partial_result(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-INTEGER-LISTS")
        .expect("integer list mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let augend = item.read_with(cx, |item, _| item.augend_editor.clone());
    let addend = item.read_with(cx, |item, _| item.addend_editor.clone());
    augend.update_in(cx, |editor, window, cx| editor.set_text("2, 4", window, cx));
    cx.run_until_parked();
    addend.update_in(cx, |editor, window, cx| editor.set_text("1, 1", window, cx));
    cx.run_until_parked();
    let before = item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        (session.expression_text(), session.revision())
    });
    augend.update_in(cx, |editor, window, cx| {
        editor.set_text(format!("3, {}", i64::MAX), window, cx)
    });
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!((session.expression_text(), session.revision()), before);
        assert_eq!(session.diagnostic(), Some("Node graph execution panicked"));
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[3, 5]").is_some());
    augend.update_in(cx, |editor, window, cx| editor.set_text("3, 4", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[4, 5]");
        assert_eq!(session.revision(), before.1 + 1);
        assert!(session.diagnostic().is_none());
    });
}

#[gpui::test]
fn at_graphite_add_typed_list_items_keep_sessions_isolated(cx: &mut TestAppContext) {
    init(cx);
    let (first, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let second = cx.update(|window, cx| cx.new(|cx| GraphicsWorkspaceItem::new(window, cx)));
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-INTEGER-LISTS")
        .expect("integer list mode is drawn");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let editor = first.read_with(cx, |item, _| item.addend_editor.clone());
    editor.update_in(cx, |editor, window, cx| editor.set_text("7, 8", window, cx));
    cx.run_until_parked();
    second.update_in(cx, |item, _, cx| {
        item.set_mode(crate::session::AddMode::VectorLists, cx)
    });
    cx.run_until_parked();
    first.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[9, 10]");
        assert_eq!(session.mode(), crate::session::AddMode::IntegerLists);
    });
    second.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(
            session.expression_text(),
            "[[2.0, 0.0]] + [[3.0, 0.0]] = [[5.0, 0.0]]"
        );
        assert_eq!(session.mode(), crate::session::AddMode::VectorLists);
    });
}

#[test]
fn at_graphite_add_all_typed_list_modes_accept_empty_operands() {
    use crate::session::AddMode;
    let cases = [
        (AddMode::IntegerLists, "", "", "1", ""),
        (AddMode::IntegerLists, "1", "", "", ""),
        (AddMode::VectorLists, "", "", "1", "2"),
        (AddMode::VectorLists, "1", "2", "", ""),
        (AddMode::ScalarVectorLists, "", "", "1", "2"),
        (AddMode::ScalarVectorLists, "1", "", "", ""),
        (AddMode::VectorScalarLists, "", "", "1", ""),
        (AddMode::VectorScalarLists, "1", "2", "", ""),
        (AddMode::IntegerLists, "", "", "", ""),
        (AddMode::VectorLists, "", "", "", ""),
        (AddMode::ScalarVectorLists, "", "", "", ""),
        (AddMode::VectorScalarLists, "", "", "", ""),
    ];
    for (mode, augend, augend_y, addend, addend_y) in cases {
        let mut session = GraphicsSession::new(2.0, 3.0);
        session.set_mode_components(mode, augend, augend_y, addend, addend_y);
        assert_eq!(session.mode(), mode);
        assert_eq!(session.result_text(), "[]");
        assert_eq!(session.revision(), 1);
        assert!(session.diagnostic().is_none());
    }
}

#[test]
fn at_graphite_add_typed_list_invalid_tokens_and_pairing_keep_commit() {
    use crate::session::AddMode;
    let mut session = GraphicsSession::new(2.0, 3.0);
    session.set_mode(AddMode::IntegerLists, "9007199254740993, -2", "2");
    let before = (session.expression_text(), session.revision());
    for token in [
        "1,",
        "1,,2",
        "1,1.5",
        "1,9223372036854775808",
        "1,-9223372036854775809",
    ] {
        session.update_inputs(token, "2");
        assert_eq!((session.expression_text(), session.revision()), before);
        assert!(session.diagnostic().is_some());
    }
    session.update_inputs("9007199254740993, -2", "2");
    assert_eq!(session.revision(), before.1);
    assert!(session.diagnostic().is_none());
    session.set_mode_components(AddMode::VectorLists, "1, 2", "3, 4", "5", "6");
    let before = (session.expression_text(), session.revision());
    for (augend, augend_y, addend, addend_y) in [
        ("1, 2", "3", "5", "6"),
        ("1", "3, 4", "5", "6"),
        ("1, 2", "3, 4", "5, 6", "7"),
        ("1, 2", "3, 4", "5", "7, 8"),
        ("1, 2", "3, 4", "", "6"),
        ("1, bad", "3, 4", "5", "6"),
    ] {
        session.update_components(augend, augend_y, addend, addend_y);
        assert_eq!((session.expression_text(), session.revision()), before);
        assert!(session.diagnostic().is_some());
    }
    session.update_components("1, 2", "3, 4", "5", "6");
    assert_eq!(session.revision(), before.1);
    assert!(session.diagnostic().is_none());
}

#[gpui::test]
fn at_graphite_sort_native_edits_keys_and_reverse_update_visible_result(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-SORT")
        .expect("Sort is selectable");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let values = item.read_with(cx, |item, _| item.augend_editor.clone());
    let keys = item.read_with(cx, |item, _| item.addend_editor.clone());
    keys.update_in(cx, |editor, window, cx| editor.set_text("", window, cx));
    cx.run_until_parked();
    values.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("30, 10, 20, 20");
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.mode(), crate::session::AddMode::Sort);
        assert_eq!(
            session.expression_text(),
            "Sort [30.0, 10.0, 20.0, 20.0] by [] (reverse: false) = [10.0, 20.0, 20.0, 30.0]"
        );
        assert!(session.diagnostic().is_none());
        assert_eq!(
            workspace::item::Item::tab_content_text(item, 0, cx).as_ref(),
            "Graphite Sort"
        );
    });
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-[10.0, 20.0, 20.0, 30.0]")
            .is_some()
    );
    keys.update_in(cx, |editor, window, cx| {
        editor.set_text("2, 0, 1, 0", window, cx)
    });
    cx.run_until_parked();
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-[10.0, 20.0, 20.0, 30.0]")
            .is_some()
    );
    let reverse = cx
        .debug_bounds("GRAPHITE-SORT-REVERSE")
        .expect("reverse toggle is drawn");
    cx.simulate_click(reverse.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert!(session.reverse());
        assert!(session.sort_values().expect("Sort is committed").2);
        assert_eq!(session.result_text(), "[30.0, 20.0, 10.0, 20.0]");
    });
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-[30.0, 20.0, 10.0, 20.0]")
            .is_some()
    );
}

#[gpui::test]
fn at_graphite_sort_invalid_reverse_preserves_tuple_and_recovers(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-SORT")
        .expect("Sort is selectable");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let values = item.read_with(cx, |item, _| item.augend_editor.clone());
    let keys = item.read_with(cx, |item, _| item.addend_editor.clone());
    values.update_in(cx, |editor, window, cx| {
        editor.set_text("10, 20, 30", window, cx)
    });
    cx.run_until_parked();
    keys.update_in(cx, |editor, window, cx| {
        editor.set_text("2, 0, 1", window, cx)
    });
    cx.run_until_parked();
    let before = item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[20.0, 30.0, 10.0]");
        (session.expression_text(), session.revision())
    });
    keys.update_in(cx, |editor, window, cx| {
        editor.set_text("2, bad", window, cx)
    });
    cx.run_until_parked();
    let reverse = cx
        .debug_bounds("GRAPHITE-SORT-REVERSE")
        .expect("reverse toggle is drawn");
    cx.simulate_click(reverse.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert!(session.reverse());
        assert_eq!((session.expression_text(), session.revision()), before);
        let (values, keys, reverse, result) =
            session.sort_values().expect("prior tuple is retained");
        assert_eq!(values, &[10.0, 20.0, 30.0]);
        assert_eq!(keys, &[2.0, 0.0, 1.0]);
        assert!(!reverse);
        assert_eq!(result, &[20.0, 30.0, 10.0]);
        assert_eq!(
            session.diagnostic(),
            Some("Enter comma-separated numbers for sort keys, or leave it blank.")
        );
    });
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-[20.0, 30.0, 10.0]")
            .is_some()
    );
    assert!(cx.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
    keys.update_in(cx, |editor, window, cx| editor.set_text("2, 0", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[10.0, 20.0, 30.0]");
        assert_eq!(session.revision(), before.1 + 1);
        assert!(session.sort_values().expect("recovered tuple").2);
        assert!(session.diagnostic().is_none());
    });
    keys.update_in(cx, |editor, window, cx| editor.set_text("", window, cx));
    cx.run_until_parked();
    assert!(
        cx.debug_bounds("GRAPHITE-RESULT-[30.0, 20.0, 10.0]")
            .is_some()
    );
}

#[gpui::test]
fn at_graphite_sort_items_are_isolated_from_add_and_other_sort_drafts(cx: &mut TestAppContext) {
    init(cx);
    let (first, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let second = cx.update(|window, cx| cx.new(|cx| GraphicsWorkspaceItem::new(window, cx)));
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-SORT")
        .expect("Sort is selectable");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let first_keys = first.read_with(cx, |item, _| item.addend_editor.clone());
    first_keys.update_in(cx, |editor, window, cx| editor.set_text("", window, cx));
    cx.run_until_parked();
    let first_values = first.read_with(cx, |item, _| item.augend_editor.clone());
    first_values.update_in(cx, |editor, window, cx| editor.set_text("3, 1", window, cx));
    cx.run_until_parked();
    let reverse = cx
        .debug_bounds("GRAPHITE-SORT-REVERSE")
        .expect("reverse toggle is drawn");
    cx.simulate_click(reverse.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    second.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.mode(), crate::session::AddMode::Scalar);
        assert_eq!(session.result(), 5.0);
        assert_eq!(session.revision(), 0);
        assert!(!session.reverse());
    });
    second.update_in(cx, |item, _, cx| {
        item.set_mode(crate::session::AddMode::Sort, cx)
    });
    cx.run_until_parked();
    first.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[3.0, 1.0]");
        assert!(session.reverse());
    });
    second.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[2.0]");
        assert!(!session.reverse());
    });
}

#[test]
fn at_graphite_sort_short_long_and_stable_keys_follow_source() {
    use crate::session::AddMode;
    let mut session = GraphicsSession::new(2.0, 3.0);
    for (keys, result) in [
        ("", "[10.0, 20.0, 30.0]"),
        ("2, 0, 1", "[20.0, 30.0, 10.0]"),
        ("2, 0", "[20.0, 30.0, 10.0]"),
        ("2, 0, 1, -100", "[20.0, 30.0, 10.0]"),
        ("0", "[10.0, 20.0, 30.0]"),
    ] {
        session.set_mode(AddMode::Sort, "10, 20, 30", keys);
        assert_eq!(session.result_text(), result);
        assert!(session.diagnostic().is_none());
    }
    session.set_reverse(true, "10, 20, 30", "0");
    assert_eq!(session.result_text(), "[10.0, 20.0, 30.0]");
    session.set_reverse(false, "", "");
    assert_eq!(session.result_text(), "[]");
    session.update_inputs("7", "3, 2, 1");
    assert_eq!(session.result_text(), "[7.0]");
    assert!(session.diagnostic().is_none());
}

#[test]
fn at_graphite_sort_float_total_order_and_same_bits_preserve_revision() {
    use crate::session::AddMode;
    let mut session = GraphicsSession::new(2.0, 3.0);
    let values = "NaN, 0, -0, -inf, inf, -3";
    session.set_mode(AddMode::Sort, values, "");
    let result = session.sort_values().expect("Sort tuple is committed").3;
    assert_eq!(
        result
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [f64::NEG_INFINITY, -3.0, -0.0, 0.0, f64::INFINITY, f64::NAN].map(f64::to_bits)
    );
    let revision = session.revision();
    session.update_inputs(values, "");
    assert_eq!(session.revision(), revision);
    assert!(session.diagnostic().is_none());
    session.set_reverse(true, values, "");
    let result = session
        .sort_values()
        .expect("reverse Sort tuple is committed")
        .3;
    assert_eq!(
        result
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [f64::NAN, f64::INFINITY, 0.0, -0.0, -3.0, f64::NEG_INFINITY].map(f64::to_bits)
    );
}

#[test]
fn at_graphite_sort_malformed_inputs_and_operation_changes_preserve_commit() {
    use crate::session::AddMode;
    let mut session = GraphicsSession::new(2.0, 3.0);
    session.set_mode(AddMode::Sort, "bad", "");
    assert_eq!(session.inputs(), (2.0, 3.0));
    assert_eq!(session.revision(), 0);
    assert_eq!(session.expression_text(), "Last valid result: 5");
    session.update_inputs("3, 1, 2", "");
    let before = (session.expression_text(), session.revision());
    for (values, keys) in [
        ("3,", ""),
        ("3,,1", ""),
        ("bad", ""),
        ("3, 1, 2", "1,"),
        ("3, 1, 2", "1,,2"),
        ("3, 1, 2", "bad"),
    ] {
        session.update_inputs(values, keys);
        assert_eq!((session.expression_text(), session.revision()), before);
        assert!(session.diagnostic().is_some());
    }
    session.update_inputs("3, 1, 2", "");
    assert_eq!(session.revision(), before.1);
    assert!(session.diagnostic().is_none());
    session.set_mode(AddMode::Integer, "bad", "2");
    assert_eq!(
        session.expression_text(),
        "Last valid result: [1.0, 2.0, 3.0]"
    );
    assert_eq!(
        session.sort_values(),
        Some((&[3.0, 1.0, 2.0][..], &[][..], false, &[1.0, 2.0, 3.0][..]))
    );
    assert_eq!(session.result_text(), "[1.0, 2.0, 3.0]");
    assert_eq!(session.revision(), before.1);
    assert!(session.diagnostic().is_some());
    session.update_inputs("8", "2");
    assert_eq!(session.integer_values(), Some((8, 2, 10)));
    assert_eq!(session.revision(), before.1 + 1);
}

#[gpui::test]
fn at_graphite_multiply_native_edit_renders_all_twelve_source_pairs(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let cases = [
        (
            "SCALAR",
            "-2",
            "",
            "3",
            "",
            "-6",
            "GRAPHITE-MODE-MULTIPLY-SCALAR",
            "GRAPHITE-RESULT--6",
        ),
        (
            "LISTS",
            "-2, 0, 3",
            "",
            "4, -2",
            "",
            "[-8.0, -0.0, -6.0]",
            "GRAPHITE-MODE-MULTIPLY-LISTS",
            "GRAPHITE-RESULT-[-8.0, -0.0, -6.0]",
        ),
        (
            "INTEGER",
            "9007199254740993",
            "",
            "2",
            "",
            "18014398509481986",
            "GRAPHITE-MODE-MULTIPLY-INTEGER",
            "GRAPHITE-RESULT-18014398509481986",
        ),
        (
            "VECTOR",
            "2",
            "-3",
            "4",
            "5",
            "[8.0, -15.0]",
            "GRAPHITE-MODE-MULTIPLY-VECTOR",
            "GRAPHITE-RESULT-[8.0, -15.0]",
        ),
        (
            "SCALAR-VECTOR",
            "-2",
            "",
            "3",
            "4",
            "[-6.0, -8.0]",
            "GRAPHITE-MODE-MULTIPLY-SCALAR-VECTOR",
            "GRAPHITE-RESULT-[-6.0, -8.0]",
        ),
        (
            "VECTOR-SCALAR",
            "3",
            "4",
            "-2",
            "",
            "[-6.0, -8.0]",
            "GRAPHITE-MODE-MULTIPLY-VECTOR-SCALAR",
            "GRAPHITE-RESULT-[-6.0, -8.0]",
        ),
        (
            "INTEGER-LISTS",
            "-2, 0, 3",
            "",
            "4, -2",
            "",
            "[-8, 0, -6]",
            "GRAPHITE-MODE-MULTIPLY-INTEGER-LISTS",
            "GRAPHITE-RESULT-[-8, 0, -6]",
        ),
        (
            "VECTOR-LISTS",
            "1, 3",
            "2, 4",
            "10",
            "20",
            "[[10.0, 40.0], [30.0, 80.0]]",
            "GRAPHITE-MODE-MULTIPLY-VECTOR-LISTS",
            "GRAPHITE-RESULT-[[10.0, 40.0], [30.0, 80.0]]",
        ),
        (
            "SCALAR-VECTOR-LISTS",
            "2, 3",
            "",
            "4",
            "5",
            "[[8.0, 10.0], [12.0, 15.0]]",
            "GRAPHITE-MODE-MULTIPLY-SCALAR-VECTOR-LISTS",
            "GRAPHITE-RESULT-[[8.0, 10.0], [12.0, 15.0]]",
        ),
        (
            "VECTOR-SCALAR-LISTS",
            "4",
            "5",
            "2, 3",
            "",
            "[[8.0, 10.0], [12.0, 15.0]]",
            "GRAPHITE-MODE-MULTIPLY-VECTOR-SCALAR-LISTS",
            "GRAPHITE-RESULT-[[8.0, 10.0], [12.0, 15.0]]",
        ),
        (
            "AFFINE",
            "1,0,0,1,10,20",
            "",
            "2,0,0,3,0,0",
            "",
            "[2.0, 0.0, 0.0, 3.0, 10.0, 20.0]",
            "GRAPHITE-MODE-MULTIPLY-AFFINE",
            "GRAPHITE-RESULT-[2.0, 0.0, 0.0, 3.0, 10.0, 20.0]",
        ),
        (
            "AFFINE-LISTS",
            "1,0,0,1,10,20;1,0,0,1,1,2",
            "",
            "2,0,0,3,0,0",
            "",
            "[[2.0, 0.0, 0.0, 3.0, 10.0, 20.0], [2.0, 0.0, 0.0, 3.0, 1.0, 2.0]]",
            "GRAPHITE-MODE-MULTIPLY-AFFINE-LISTS",
            "GRAPHITE-RESULT-[[2.0, 0.0, 0.0, 3.0, 10.0, 20.0], [2.0, 0.0, 0.0, 3.0, 1.0, 2.0]]",
        ),
    ];
    for (
        mode,
        multiplier,
        multiplier_y,
        multiplicand,
        multiplicand_y,
        result,
        selector,
        result_selector,
    ) in cases
    {
        let bounds = cx
            .debug_bounds(selector)
            .expect("declared pair is selectable");
        cx.simulate_click(bounds.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        let first = item.read_with(cx, |item, _| item.augend_editor.clone());
        first.update_in(cx, |editor, window, cx| {
            editor.set_text("", window, cx);
            window.focus(&editor.focus_handle(cx), cx);
        });
        cx.simulate_input(multiplier);
        cx.run_until_parked();
        for (editor, text) in [
            (
                item.read_with(cx, |item, _| item.augend_y_editor.clone()),
                multiplier_y,
            ),
            (
                item.read_with(cx, |item, _| item.addend_editor.clone()),
                multiplicand,
            ),
            (
                item.read_with(cx, |item, _| item.addend_y_editor.clone()),
                multiplicand_y,
            ),
        ] {
            editor.update_in(cx, |editor, window, cx| editor.set_text(text, window, cx));
            cx.run_until_parked();
        }
        item.read_with(cx, |item, cx| {
            let session = item.session.read(cx);
            assert!(session.mode().is_multiply(), "{mode}");
            assert_eq!(session.result_text(), result, "{mode}");
            assert!(session.expression_text().starts_with("Multiply "), "{mode}");
            assert!(
                session.diagnostic().is_none(),
                "{mode}: {:?}",
                session.diagnostic()
            );
            assert_eq!(
                workspace::item::Item::tab_content_text(item, 0, cx).as_ref(),
                "Graphite Multiply"
            );
        });
        assert!(cx.debug_bounds(result_selector).is_some(), "{mode}");
    }
}

#[gpui::test]
fn at_graphite_multiply_affine_invalid_edits_preserve_whole_commit_and_recover(
    cx: &mut TestAppContext,
) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-MULTIPLY-AFFINE")
        .expect("affine pair is selectable");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let multiplier = item.read_with(cx, |item, _| item.augend_editor.clone());
    let multiplicand = item.read_with(cx, |item, _| item.addend_editor.clone());
    multiplier.update_in(cx, |editor, window, cx| {
        editor.set_text("1,0,0,1,10,20", window, cx)
    });
    cx.run_until_parked();
    multiplicand.update_in(cx, |editor, window, cx| {
        editor.set_text("2,0,0,3,0,0", window, cx)
    });
    cx.run_until_parked();
    let before = item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        (session.expression_text(), session.revision())
    });
    for text in ["1,0,0,1,10", "1,0,0,1,10,20,30", "1,0,bad,1,10,20", ""] {
        multiplier.update_in(cx, |editor, window, cx| editor.set_text(text, window, cx));
        cx.run_until_parked();
        item.read_with(cx, |item, cx| {
            let session = item.session.read(cx);
            assert_eq!((session.expression_text(), session.revision()), before);
            assert!(session.diagnostic().is_some());
        });
        assert!(
            cx.debug_bounds("GRAPHITE-RESULT-[2.0, 0.0, 0.0, 3.0, 10.0, 20.0]")
                .is_some()
        );
        assert!(cx.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
    }
    multiplier.update_in(cx, |editor, window, cx| {
        editor.set_text("1,0,0,1,1,2", window, cx)
    });
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[2.0, 0.0, 0.0, 3.0, 1.0, 2.0]");
        assert_eq!(session.revision(), before.1 + 1);
        assert!(session.diagnostic().is_none());
    });
}

#[cfg(debug_assertions)]
#[gpui::test]
fn at_graphite_multiply_late_integer_slot_failure_never_publishes_partial_result(
    cx: &mut TestAppContext,
) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-MULTIPLY-INTEGER-LISTS")
        .expect("integer list pair is selectable");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let multiplier = item.read_with(cx, |item, _| item.augend_editor.clone());
    let multiplicand = item.read_with(cx, |item, _| item.addend_editor.clone());
    multiplier.update_in(cx, |editor, window, cx| editor.set_text("2, 4", window, cx));
    cx.run_until_parked();
    multiplicand.update_in(cx, |editor, window, cx| editor.set_text("3, 2", window, cx));
    cx.run_until_parked();
    let before = item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[6, 8]");
        (session.expression_text(), session.revision())
    });
    multiplier.update_in(cx, |editor, window, cx| {
        editor.set_text(format!("3, {}", i64::MAX), window, cx)
    });
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!((session.expression_text(), session.revision()), before);
        assert_eq!(session.diagnostic(), Some("Node graph execution panicked"));
    });
    assert!(cx.debug_bounds("GRAPHITE-RESULT-[6, 8]").is_some());
    multiplier.update_in(cx, |editor, window, cx| editor.set_text("3, 4", window, cx));
    cx.run_until_parked();
    item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result_text(), "[9, 8]");
        assert_eq!(session.revision(), before.1 + 1);
        assert!(session.diagnostic().is_none());
    });
}

#[gpui::test]
fn at_graphite_multiply_items_remain_isolated_from_add_and_sort(cx: &mut TestAppContext) {
    init(cx);
    let (first, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    let second = cx.update(|window, cx| cx.new(|cx| GraphicsWorkspaceItem::new(window, cx)));
    let mode = cx
        .debug_bounds("GRAPHITE-MODE-MULTIPLY-SCALAR")
        .expect("Multiply is selectable");
    cx.simulate_click(mode.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let multiplicand = first.read_with(cx, |item, _| item.addend_editor.clone());
    multiplicand.update_in(cx, |editor, window, cx| editor.set_text("8", window, cx));
    cx.run_until_parked();
    first.read_with(cx, |item, cx| {
        assert_eq!(item.session.read(cx).result(), 16.0)
    });
    second.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.mode(), crate::session::AddMode::Scalar);
        assert_eq!(session.result(), 5.0);
        assert_eq!(session.revision(), 0);
    });
    second.update_in(cx, |item, _, cx| {
        item.set_mode(crate::session::AddMode::Sort, cx)
    });
    cx.run_until_parked();
    second.read_with(cx, |item, cx| {
        assert_eq!(item.session.read(cx).result_text(), "[2.0]")
    });
    first.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.result(), 16.0);
        assert_eq!(session.mode(), crate::session::AddMode::MultiplyScalar);
    });
}

#[test]
fn at_graphite_multiply_all_six_list_pairs_accept_empty_operands() {
    use crate::session::AddMode;
    for (mode, first, first_y, second, second_y) in [
        (AddMode::MultiplyLists, "2", "", "3", ""),
        (AddMode::MultiplyIntegerLists, "2", "", "3", ""),
        (AddMode::MultiplyVectorLists, "2", "4", "3", "5"),
        (AddMode::MultiplyScalarVectorLists, "2", "", "3", "5"),
        (AddMode::MultiplyVectorScalarLists, "2", "4", "3", ""),
        (
            AddMode::MultiplyAffineLists,
            "1,0,0,1,2,3",
            "",
            "2,0,0,3,0,0",
            "",
        ),
    ] {
        for (first, first_y, second, second_y) in [
            ("", "", second, second_y),
            (first, first_y, "", ""),
            ("", "", "", ""),
        ] {
            let mut session = GraphicsSession::new(2.0, 3.0);
            session.set_mode_components(mode, first, first_y, second, second_y);
            assert_eq!(session.mode(), mode);
            assert_eq!(session.result_text(), "[]");
            assert_eq!(session.revision(), 1);
            assert!(session.diagnostic().is_none());
        }
    }
}

#[test]
fn at_graphite_multiply_float_bits_precision_and_noop_revision_follow_source() {
    use crate::session::AddMode;
    let mut session = GraphicsSession::new(2.0, 3.0);
    for (first, second, expected) in [
        ("-0", "2", -0.0),
        ("inf", "2", f64::INFINITY),
        ("-inf", "-2", f64::INFINITY),
        ("1e308", "1e308", f64::INFINITY),
    ] {
        session.set_mode(AddMode::MultiplyScalar, first, second);
        assert_eq!(session.result().to_bits(), expected.to_bits());
        let revision = session.revision();
        session.update_inputs(first, second);
        assert_eq!(session.revision(), revision);
        assert!(session.diagnostic().is_none());
    }
    session.update_inputs("NaN", "2");
    assert!(session.result().is_nan());
    let revision = session.revision();
    session.update_inputs("NaN", "2");
    assert_eq!(session.revision(), revision);
    session.set_mode(AddMode::MultiplyInteger, "9007199254740993", "2");
    assert_eq!(
        session.expression_text(),
        "Multiply 9007199254740993 by 2 = 18014398509481986"
    );
    for invalid in ["9007199254740993.0", "9223372036854775808", "1e3"] {
        let before = (session.expression_text(), session.revision());
        session.update_inputs(invalid, "2");
        assert_eq!((session.expression_text(), session.revision()), before);
        assert!(session.diagnostic().is_some());
    }
}

#[test]
fn at_graphite_multiply_affine_order_list_repeat_and_noop_are_original() {
    use crate::session::AddMode;
    let mut session = GraphicsSession::new(2.0, 3.0);
    let translation = "1,0,0,1,10,20";
    let scale = "2,0,0,3,0,0";
    session.set_mode(AddMode::MultiplyAffine, translation, scale);
    assert_eq!(session.result_text(), "[2.0, 0.0, 0.0, 3.0, 10.0, 20.0]");
    let revision = session.revision();
    session.update_inputs(translation, scale);
    assert_eq!(session.revision(), revision);
    session.update_inputs(scale, translation);
    assert_eq!(session.result_text(), "[2.0, 0.0, 0.0, 3.0, 20.0, 60.0]");
    session.set_mode(
        AddMode::MultiplyAffineLists,
        "1,0,0,1,10,20;1,0,0,1,1,2",
        scale,
    );
    assert_eq!(
        session.result_text(),
        "[[2.0, 0.0, 0.0, 3.0, 10.0, 20.0], [2.0, 0.0, 0.0, 3.0, 1.0, 2.0]]"
    );
    let revision = session.revision();
    session.update_inputs("1,0,0,1,10,20;1,0,0,1,1,2", scale);
    assert_eq!(session.revision(), revision);
    assert!(session.diagnostic().is_none());
}

#[test]
fn at_graphite_multiply_malformed_affine_vector_and_operation_drafts_do_not_commit() {
    use crate::session::AddMode;
    let mut session = GraphicsSession::new(2.0, 3.0);
    session.set_mode(AddMode::MultiplyAffine, "bad", "bad");
    assert_eq!(session.inputs(), (2.0, 3.0));
    assert_eq!(session.revision(), 0);
    assert_eq!(session.expression_text(), "Last valid result: 5");
    let second = "2,0,0,3,0,0";
    session.update_inputs("1,0,0,1,2,3", second);
    let before = (session.expression_text(), session.revision());
    for (first, second) in [
        ("", second),
        ("1,0,0,1,2", second),
        ("1,0,0,1,2,3,4", second),
        ("1,0,bad,1,2,3", second),
        ("1,0,0,1,2,3", "2,0,0,3,0,"),
    ] {
        session.update_inputs(first, second);
        assert_eq!((session.expression_text(), session.revision()), before);
        assert!(session.diagnostic().is_some());
    }
    session.update_inputs("1,0,0,1,2,3", "2,0,0,3,0,0");
    assert_eq!(session.revision(), before.1);
    assert!(session.diagnostic().is_none());
    session.set_mode(AddMode::MultiplyAffineLists, "1,0,0,1,2,3", "2,0,0,3,0,0");
    let before = (session.expression_text(), session.revision());
    for text in [
        ";1,0,0,1,2,3",
        "1,0,0,1,2,3;",
        "1,0,0,1,2,3;;1,0,0,1,0,0",
        "1,0,0,1,2,3;1,0,0,1,2",
        "1,0,0,1,2,3;1,0,0,bad,2,3",
    ] {
        session.update_inputs(text, "2,0,0,3,0,0");
        assert_eq!((session.expression_text(), session.revision()), before);
        assert!(session.diagnostic().is_some());
    }
    session.set_mode_components(AddMode::MultiplyVectorLists, "1,2", "3,4", "5", "6");
    let before = (session.expression_text(), session.revision());
    session.update_components("1,2", "3", "5", "6");
    assert_eq!((session.expression_text(), session.revision()), before);
    assert_eq!(
        session.diagnostic(),
        Some("Enter the same number of x and y components for multiplier.")
    );
    session.update_components("1,2", "3,4", "5", "6");
    assert_eq!(session.revision(), before.1);
    assert!(session.diagnostic().is_none());
    let committed = (session.result_text(), session.revision());
    session.set_mode(AddMode::Scalar, "bad", "2");
    assert_eq!(
        session.expression_text(),
        format!("Last valid result: {}", committed.0)
    );
    assert_eq!((session.result_text(), session.revision()), committed);
    assert!(session.diagnostic().is_some());
    session.update_inputs("8", "2");
    assert_eq!(session.result(), 10.0);
    assert_eq!(session.revision(), before.1 + 1);
    assert!(session.diagnostic().is_none());

    session.set_mode(AddMode::Sort, "3,1,2", "");
    let committed = (session.result_text(), session.revision());
    assert_eq!(committed.0, "[1.0, 2.0, 3.0]");
    for mode in [AddMode::MultiplyScalar, AddMode::Integer] {
        session.set_mode(mode, "bad", "2");
        assert_eq!(
            session.expression_text(),
            format!("Last valid result: {}", committed.0)
        );
        assert_eq!((session.result_text(), session.revision()), committed);
        assert!(session.diagnostic().is_some());
    }
    session.update_inputs("8", "2");
    assert_eq!(session.integer_values(), Some((8, 2, 10)));
    assert_eq!(session.revision(), committed.1 + 1);
    assert!(session.diagnostic().is_none());
}

#[gpui::test]
fn at_graphite_invert_color_native_edit_updates_visible_original_result(cx: &mut TestAppContext) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    for (editor, text) in [
        (
            item.read_with(cx, |item, _| item.augend_editor.clone()),
            "0",
        ),
        (
            item.read_with(cx, |item, _| item.augend_y_editor.clone()),
            "0.5",
        ),
        (
            item.read_with(cx, |item, _| item.addend_editor.clone()),
            "1",
        ),
        (
            item.read_with(cx, |item, _| item.addend_y_editor.clone()),
            "0.5",
        ),
    ] {
        editor.update_in(cx, |editor, window, cx| editor.set_text(text, window, cx));
    }
    cx.run_until_parked();
    let button = cx
        .debug_bounds("GRAPHITE-MODE-INVERT-COLOR")
        .expect("native Invert selection");
    cx.simulate_click(button.center(), gpui::Modifiers::default());
    cx.run_until_parked();
    let editor = item.read_with(cx, |item, _| item.augend_editor.clone());
    editor.update_in(cx, |editor, window, cx| {
        editor.set_text("", window, cx);
        window.focus(&editor.focus_handle(cx), cx);
    });
    cx.simulate_input("0.25");
    cx.run_until_parked();
    let result = item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        let (input, output) = session
            .color_values()
            .expect("original Color evaluation committed");
        assert_eq!(input, [0.25, 0.5, 1.0, 0.5]);
        for (actual, expected) in output.into_iter().zip([0.75, 0.5, 0.0, 0.5]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!(session.diagnostic().is_none());
        assert!(session.expression_text().starts_with("Invert RGBA "));
        assert_eq!(
            workspace::item::Item::tab_content_text(item, 0, cx).as_ref(),
            "Graphite Invert"
        );
        session.result_text()
    });
    assert!(!result.is_empty());
    assert!(cx.debug_bounds("GRAPHITE-RESULT").is_some());
}

#[gpui::test]
fn at_graphite_invert_color_invalid_channels_retain_whole_result_and_recover(
    cx: &mut TestAppContext,
) {
    init(cx);
    let (item, cx) = cx.add_window_view(GraphicsWorkspaceItem::new);
    for (editor, text) in [
        (
            item.read_with(cx, |item, _| item.augend_editor.clone()),
            "0.25",
        ),
        (
            item.read_with(cx, |item, _| item.augend_y_editor.clone()),
            "0.5",
        ),
        (
            item.read_with(cx, |item, _| item.addend_editor.clone()),
            "1",
        ),
        (
            item.read_with(cx, |item, _| item.addend_y_editor.clone()),
            "0.5",
        ),
    ] {
        editor.update_in(cx, |editor, window, cx| editor.set_text(text, window, cx));
    }
    item.update(cx, |item, cx| {
        item.set_mode(crate::session::AddMode::InvertColor, cx)
    });
    cx.run_until_parked();
    let before = item.read_with(cx, |item, cx| {
        let session = item.session.read(cx);
        (
            session.color_values(),
            session.revision(),
            session.result_text(),
        )
    });
    for (editor, valid, name) in [
        (
            item.read_with(cx, |item, _| item.augend_editor.clone()),
            "0.25",
            "red",
        ),
        (
            item.read_with(cx, |item, _| item.augend_y_editor.clone()),
            "0.5",
            "green",
        ),
        (
            item.read_with(cx, |item, _| item.addend_editor.clone()),
            "1",
            "blue",
        ),
        (
            item.read_with(cx, |item, _| item.addend_y_editor.clone()),
            "0.5",
            "alpha",
        ),
    ] {
        editor.update_in(cx, |editor, window, cx| editor.set_text("bad", window, cx));
        cx.run_until_parked();
        item.read_with(cx, |item, cx| {
            let session = item.session.read(cx);
            assert_eq!(
                (
                    session.color_values(),
                    session.revision(),
                    session.result_text()
                ),
                before
            );
            assert_eq!(
                session.diagnostic(),
                Some(format!("Enter a 32-bit number for {name}.").as_str())
            );
        });
        assert!(cx.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
        editor.update_in(cx, |editor, window, cx| editor.set_text(valid, window, cx));
        cx.run_until_parked();
        item.read_with(cx, |item, cx| {
            let session = item.session.read(cx);
            assert_eq!(
                (
                    session.color_values(),
                    session.revision(),
                    session.result_text()
                ),
                before
            );
            assert!(session.diagnostic().is_none());
        });
    }
}

#[test]
fn at_graphite_invert_color_f32_noop_signed_alpha_and_failed_mode_switch() {
    use crate::session::AddMode;
    let mut session = GraphicsSession::new(2.0, 3.0);
    session.set_mode_components(AddMode::InvertColor, "0.25", "bad", "1", "0.5");
    assert_eq!(session.expression_text(), "Last valid result: 5");
    assert_eq!(session.revision(), 0);
    assert_eq!(session.inputs(), (2.0, 3.0));
    session.update_components("0.25", "0.5", "1", "-0");
    let (input, output) = session.color_values().expect("four f32 inputs evaluate");
    assert_eq!(input[3].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(output[3].to_bits(), (-0.0_f32).to_bits());
    let revision = session.revision();
    session.update_components("0.25000001", "0.5", "1", "-0.0");
    assert_eq!(session.revision(), revision);
    session.update_components("0.25", "0.5", "1", "0");
    assert_eq!(session.revision(), revision + 1);
    let (input, output) = session.color_values().expect("alpha sign changes commit");
    assert_eq!(input[3].to_bits(), 0.0_f32.to_bits());
    assert_eq!(output[3].to_bits(), 0.0_f32.to_bits());
    let retained = session.color_values();
    session.set_mode(AddMode::Integer, "bad", "2");
    assert!(session.expression_text().starts_with("Last valid result: "));
    assert_eq!(session.color_values(), retained);
    session.update_inputs("8", "2");
    assert_eq!(session.integer_values(), Some((8, 2, 10)));
}

#[gpui::test]
fn at_graphite_invert_color_two_native_items_preserve_independent_work(cx: &mut TestAppContext) {
    init(cx);
    let (first, first_context) = cx.add_window_view(GraphicsWorkspaceItem::new);
    for (editor, text) in [
        (
            first.read_with(first_context, |item, _| item.augend_editor.clone()),
            "0.25",
        ),
        (
            first.read_with(first_context, |item, _| item.augend_y_editor.clone()),
            "0.5",
        ),
        (
            first.read_with(first_context, |item, _| item.addend_editor.clone()),
            "1",
        ),
        (
            first.read_with(first_context, |item, _| item.addend_y_editor.clone()),
            "0.5",
        ),
    ] {
        editor.update_in(first_context, |editor, window, cx| {
            editor.set_text(text, window, cx)
        });
    }
    first.update(first_context, |item, cx| {
        item.set_mode(crate::session::AddMode::InvertColor, cx)
    });
    first_context.run_until_parked();
    let (second, second_context) = cx.add_window_view(GraphicsWorkspaceItem::new);
    first.read_with(second_context, |item, cx| {
        assert!(item.session.read(cx).color_values().is_some())
    });
    second.read_with(second_context, |item, cx| {
        let session = item.session.read(cx);
        assert_eq!(session.inputs(), (2.0, 3.0));
        assert_eq!(session.result(), 5.0);
        assert_eq!(session.revision(), 0);
        assert!(session.diagnostic().is_none());
    });
}

mod imported_image_item_tests {
    use super::*;
    use crate::graph::{
        CancelImageLoad, ImportedImageItem, InvertImage, MakeImageOpaque, ShowOriginalImage,
    };
    use crate::resources::tests::{
        HostReaderFixture, IndexedRequestPng, host_reader_fixture, indexed_request_png,
    };
    use crate::session::{GraphiteImageSession, GraphitePngLoad};
    use gpui::{Entity, IntoElement as _, VisualTestContext, point, px, size};
    use graphics_core::resources::{CpuImageEdit, EncodedPngBudget};
    use std::sync::Arc;
    use workspace::item::Item as _;

    fn load(fixture: &HostReaderFixture, png: &IndexedRequestPng) -> GraphitePngLoad {
        GraphitePngLoad {
            assets: fixture.assets.clone(),
            asset_identity: png.asset_identity.clone(),
            authorization: fixture.input_authorization.clone(),
            identity: png.image_identity.clone(),
            descriptor: png.descriptor,
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
        }
    }

    fn paint(window: &mut VisualTestContext, item: &Entity<ImportedImageItem>) {
        window.draw(point(px(0.), px(0.)), size(px(640.), px(480.)), |_, _| {
            item.clone().into_any_element()
        });
    }

    #[gpui::test]
    async fn at_graphite_imported_image_workspace_actions_use_injected_real_session(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
        });
        let fixture = host_reader_fixture().expect("real host service");
        let png = indexed_request_png(&fixture, 4).expect("actual producer PNG");
        let session = cx.new(|_| GraphiteImageSession::new(4096));
        session
            .update(cx, |session, cx| session.load_png(load(&fixture, &png), cx))
            .expect("exact supplied request");
        cx.run_until_parked();
        let source = session.read_with(cx, |session, _| {
            session
                .resource_state()
                .committed()
                .expect("admitted PNG")
                .clone()
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (host, cx) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = host.read_with(cx, |host, _| host.workspace().clone());
        let item = workspace.update_in(cx, |workspace, window, cx| {
            let item =
                cx.new(|cx| ImportedImageItem::from_session(session.clone(), 4096, window, cx));
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
            window.focus(&item.focus_handle(cx), cx);
            item
        });
        cx.run_until_parked();
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| workspace
                .active_item_as::<ImportedImageItem>(cx)
                .expect("native Item")
                .entity_id()),
            item.entity_id()
        );
        assert!(cx.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
        cx.dispatch_action(InvertImage);
        cx.run_until_parked();
        session.read_with(cx, |session, _| {
            assert_eq!(session.operation(), Some(CpuImageEdit::Invert));
            let expected = graphics_core::resources::evaluate_cpu_image_edit(
                &source,
                CpuImageEdit::Invert,
                4096,
            )
            .expect("original node");
            assert_eq!(session.result().expect("edit").image(), expected.image());
        });
        assert!(cx.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
        cx.dispatch_action(MakeImageOpaque);
        cx.run_until_parked();
        assert_eq!(
            session.read_with(cx, |session, _| session
                .result()
                .expect("opaque")
                .operation()),
            CpuImageEdit::MakeOpaque
        );
        cx.dispatch_action(ShowOriginalImage);
        cx.run_until_parked();
        session.read_with(cx, |session, _| {
            assert_eq!(session.operation(), None);
            assert!(session.result().is_none());
        });
        assert!(cx.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
        let mut pending = load(&fixture, &png);
        pending.identity.version_id = "caller-supplied-second-version".into();
        session
            .update(cx, |session, cx| session.load_png(pending, cx))
            .expect("pending request");
        cx.dispatch_action(CancelImageLoad);
        cx.run_until_parked();
        session.read_with(cx, |session, _| {
            assert!(session.resource_state().active_request().is_none());
            assert!(!session.resource_state().is_fresh());
            assert!(Arc::ptr_eq(
                session.resource_state().committed().expect("retained"),
                &source
            ));
        });
        assert!(cx.debug_bounds("GRAPHITE-IMPORTED-LAST-VALID").is_some());
        assert!(cx.debug_bounds("GRAPHITE-IMPORTED-SESSION-ERROR").is_some());
    }

    #[gpui::test]
    async fn at_graphite_imported_image_split_shares_source_and_presentation_but_not_focus(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixture = host_reader_fixture().expect("service");
        let png = indexed_request_png(&fixture, 3).expect("real PNG");
        let session = cx.new(|_| GraphiteImageSession::new(4096));
        session
            .update(cx, |session, cx| session.load_png(load(&fixture, &png), cx))
            .expect("source request");
        cx.run_until_parked();
        let (first, window) = cx.add_window_view(|window, cx| {
            ImportedImageItem::from_session(session.clone(), 4096, window, cx)
        });
        let split = first
            .update_in(window, |item, window, cx| {
                item.clone_on_split(None, window, cx)
            })
            .await
            .expect("Item split");
        let (first_session, first_presentation, first_focus) =
            first.read_with(window, |item, cx| {
                (
                    item.session().entity_id(),
                    item.presentation().entity_id(),
                    item.focus_handle(cx),
                )
            });
        split.read_with(window, |item, cx| {
            assert_eq!(item.session().entity_id(), first_session);
            assert_eq!(item.presentation().entity_id(), first_presentation);
            assert_ne!(item.focus_handle(cx), first_focus);
        });
        let image = first.read_with(window, |item, cx| {
            item.presentation()
                .read(cx)
                .image()
                .expect("prepared")
                .clone()
        });
        paint(window, &split);
        assert_eq!(
            split.read_with(window, |item, cx| item
                .presentation()
                .read(cx)
                .image()
                .expect("same owner")
                .id),
            image.id
        );
        let mut changed = load(&fixture, &png);
        changed.identity.version_id = "caller-supplied-refreshed-version".into();
        session
            .update(window, |session, cx| session.load_png(changed, cx))
            .expect("new source request");
        window.run_until_parked();
        paint(window, &split);
        assert_eq!(
            session.read_with(window, |session, _| session.resource_state().revision()),
            2
        );
        let replacement = split.read_with(window, |item, cx| {
            item.presentation()
                .read(cx)
                .image()
                .expect("replacement")
                .clone()
        });
        assert_ne!(replacement.id, image.id);
        assert_eq!(
            first.read_with(window, |item, cx| item
                .presentation()
                .read(cx)
                .image()
                .expect("shared replacement")
                .id),
            replacement.id
        );
    }

    #[gpui::test]
    fn at_graphite_imported_image_error_and_revocation_are_visible_and_recoverable(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixture = host_reader_fixture().expect("service");
        let png = indexed_request_png(&fixture, 4).expect("PNG");
        let session = cx.new(|_| GraphiteImageSession::new(4096));
        session
            .update(cx, |session, cx| session.load_png(load(&fixture, &png), cx))
            .expect("load");
        cx.run_until_parked();
        let (item, window) = cx.add_window_view(|window, cx| {
            ImportedImageItem::from_session(session.clone(), 4096, window, cx)
        });
        let image = item.read_with(window, |item, cx| {
            item.presentation()
                .read(cx)
                .image()
                .expect("preview")
                .clone()
        });
        let mut corrupt = load(&fixture, &png);
        corrupt.identity.content_sha256[0] ^= 1;
        session
            .update(window, |session, cx| session.load_png(corrupt, cx))
            .expect("changed expected digest");
        window.run_until_parked();
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-SESSION-ERROR")
                .is_some()
        );
        assert_eq!(
            item.read_with(window, |item, cx| item
                .presentation()
                .read(cx)
                .image()
                .expect("last valid")
                .id),
            image.id
        );
        session
            .update(window, |session, cx| {
                session.revoke(&png.image_identity, cx)
            })
            .expect("committed revoke");
        window.run_until_parked();
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-REVOKED").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert!(item.read_with(window, |item, cx| {
            item.presentation().read(cx).image().is_none()
        }));
        session
            .update(window, |session, cx| {
                session.load_png(load(&fixture, &png), cx)
            })
            .expect("explicit validated recovery");
        window.run_until_parked();
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-SESSION-ERROR")
                .is_none()
        );
        assert!(!session.read_with(window, |session, _| session.resource_state().is_revoked()));
    }

    #[gpui::test]
    fn at_graphite_imported_image_preview_failure_does_not_poison_source_or_other_item(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixture = host_reader_fixture().expect("service");
        let png = indexed_request_png(&fixture, 3).expect("PNG");
        let session = cx.new(|_| GraphiteImageSession::new(4096));
        session
            .update(cx, |session, cx| session.load_png(load(&fixture, &png), cx))
            .expect("load");
        cx.run_until_parked();
        let revision = session.read_with(cx, |session, _| session.resource_state().revision());
        let (failed, window) = cx.add_window_view(|window, cx| {
            ImportedImageItem::from_session(session.clone(), 0, window, cx)
        });
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                .is_some()
        );
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert!(failed.read_with(window, |item, _| item.preview_diagnostic().is_some()));
        assert!(session.read_with(window, |session, _| session.resource_state().is_fresh()));
        assert_eq!(
            session.read_with(window, |session, _| session.resource_state().revision()),
            revision
        );
        let other = window.new(|_| GraphiteImageSession::new(4096));
        let valid = window.update(|window, cx| {
            cx.new(|cx| ImportedImageItem::from_session(session.clone(), 4096, window, cx))
        });
        paint(window, &valid);
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
        assert!(valid.read_with(window, |item, _| item.preview_diagnostic().is_none()));
        assert!(other.read_with(window, |session, _| {
            session.resource_state().committed().is_none()
        }));
    }

    #[gpui::test]
    async fn at_graphite_imported_image_final_split_release_drops_shared_owners(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixture = host_reader_fixture().expect("service");
        let png = indexed_request_png(&fixture, 4).expect("PNG");
        let session = cx.new(|_| GraphiteImageSession::new(4096));
        session
            .update(cx, |session, cx| session.load_png(load(&fixture, &png), cx))
            .expect("load");
        cx.run_until_parked();
        let weak_session = session.downgrade();
        let window = cx.add_empty_window();
        let first = window.update(|window, cx| {
            cx.new(|cx| ImportedImageItem::from_session(session.clone(), 4096, window, cx))
        });
        let split = first
            .update_in(window, |item, window, cx| {
                item.clone_on_split(None, window, cx)
            })
            .await
            .expect("split");
        paint(window, &first);
        let (weak_presentation, image) = first.read_with(window, |item, cx| {
            (
                item.presentation().downgrade(),
                item.presentation()
                    .read(cx)
                    .image()
                    .expect("pixels")
                    .clone(),
            )
        });
        assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
        drop(session);
        drop(first);
        window.cx.update(|_| ());
        assert!(weak_session.upgrade().is_some());
        assert!(weak_presentation.upgrade().is_some());
        paint(window, &split);
        window.update(|_, _| drop(split));
        assert!(weak_session.upgrade().is_none());
        assert!(weak_presentation.upgrade().is_none());
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&image)));
    }
}

mod encoded_png_item_tests {
    use super::*;
    use crate::graph::{
        CancelImageLoad, ImportedImageItem, InvertImage, MakeImageOpaque, ShowOriginalImage,
    };
    use crate::resources::tests::{
        HostReaderFixture, IndexedRequestPng, host_reader_fixture, indexed_request_png,
    };
    use crate::session::{GraphiteEncodedPngLoad, GraphiteImageSession, GraphitePngLoad};
    use gpui::{Action as _, Entity, IntoElement as _, VisualTestContext, point, px, size};
    use graphics_core::resources::{CpuImageEdit, EncodedPngBudget};
    use std::sync::Arc;

    fn encoded_load(
        fixture: &HostReaderFixture,
        png: &IndexedRequestPng,
    ) -> GraphiteEncodedPngLoad {
        GraphiteEncodedPngLoad {
            assets: fixture.assets.clone(),
            asset_identity: png.asset_identity.clone(),
            authorization: fixture.input_authorization.clone(),
            identity: png.image_identity.clone(),
            transform: [1., 0., 0., 1., 0., 0.],
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
        }
    }

    fn known_load(fixture: &HostReaderFixture, png: &IndexedRequestPng) -> GraphitePngLoad {
        let encoded = encoded_load(fixture, png);
        GraphitePngLoad {
            assets: encoded.assets,
            asset_identity: encoded.asset_identity,
            authorization: encoded.authorization,
            identity: encoded.identity,
            descriptor: png.descriptor,
            encoded_budget: encoded.encoded_budget,
        }
    }

    fn paint(window: &mut VisualTestContext, item: &Entity<ImportedImageItem>) {
        window.draw(point(px(0.), px(0.)), size(px(640.), px(480.)), |_, _| {
            item.clone().into_any_element()
        });
    }

    #[gpui::test]
    async fn at_graphite_encoded_png_item_discovers_dimensions_and_displays_original_edits(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
        });
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 3).expect("actual RGB PNG producer");
        let session = cx.new(|_| GraphiteImageSession::new(4096));
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (host, window) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = host.read_with(window, |host, _| host.workspace().clone());
        let item = workspace.update_in(window, |workspace, window, cx| {
            let item =
                cx.new(|cx| ImportedImageItem::from_session(session.clone(), 4096, window, cx));
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
            window.focus(&item.focus_handle(cx), cx);
            item
        });
        session
            .update(window, |session, cx| {
                session.load_encoded_png(encoded_load(&fixture, &png), cx)
            })
            .expect("request without expected dimensions");
        session.read_with(window, |session, _| {
            assert!(session.resource_state().has_active_request());
            assert!(session.resource_state().active_request().is_none());
            assert_eq!(
                session
                    .resource_state()
                    .active_encoded_request()
                    .expect("encoded request")
                    .identity(),
                &png.image_identity
            );
            assert!(session.resource_state().committed().is_none());
            assert_eq!(session.resource_state().revision(), 0);
        });
        paint(window, &item);
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-LOADING").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        window.run_until_parked();
        let source = session.read_with(window, |session, _| {
            assert!(!session.resource_state().has_active_request());
            assert_eq!(session.resource_state().revision(), 1);
            assert!(session.resource_state().is_fresh());
            assert_eq!(session.operation(), None);
            let source = session
                .resource_state()
                .committed()
                .expect("original decoded source");
            assert_eq!(source.identity(), &png.image_identity);
            assert_eq!(source.descriptor(), &png.descriptor);
            assert_eq!(
                (source.image().width, source.image().height),
                (png.descriptor.width, png.descriptor.height)
            );
            source.clone()
        });
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_item_as::<ImportedImageItem>(cx)
                .expect("injected Item")
                .entity_id()),
            item.entity_id()
        );
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        for (action, operation) in [
            (InvertImage.boxed_clone(), CpuImageEdit::Invert),
            (MakeImageOpaque.boxed_clone(), CpuImageEdit::MakeOpaque),
        ] {
            window.update(|window, cx| window.dispatch_action(action, cx));
            window.run_until_parked();
            let expected =
                graphics_core::resources::evaluate_cpu_image_edit(&source, operation, 4096)
                    .expect("actual original unary node");
            session.read_with(window, |session, _| {
                assert_eq!(session.operation(), Some(operation));
                assert!(session.result_is_fresh());
                assert_eq!(
                    session.result().expect("edited result").image(),
                    expected.image()
                );
                assert_eq!(
                    session.result().expect("edited result").input_identity(),
                    source.identity()
                );
                assert_eq!(session.resource_state().revision(), 1);
                assert!(Arc::ptr_eq(
                    session
                        .resource_state()
                        .committed()
                        .expect("immutable source"),
                    &source
                ));
            });
            assert!(window.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        }
        window.dispatch_action(ShowOriginalImage);
        window.run_until_parked();
        assert!(session.read_with(window, |session, _| session.result().is_none()));
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
    }

    #[gpui::test]
    fn at_graphite_encoded_png_item_known_and_encoded_requests_supersede_then_cancel(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("actual RGBA PNG");
        let session = cx.new(|_| GraphiteImageSession::new(4096));
        let mut known = known_load(&fixture, &png);
        known.identity.version_id = "test-supplied-obsolete-known".into();
        session
            .update(cx, |session, cx| session.load_png(known, cx))
            .expect("known request");
        let old_generation = session.read_with(cx, |session, _| {
            session
                .resource_state()
                .active_request()
                .expect("known active")
                .generation()
        });
        session
            .update(cx, |session, cx| {
                session.load_encoded_png(encoded_load(&fixture, &png), cx)
            })
            .expect("encoded supersedes known");
        session.read_with(cx, |session, _| {
            assert!(session.resource_state().active_request().is_none());
            assert!(
                session
                    .resource_state()
                    .active_encoded_request()
                    .expect("encoded active")
                    .generation()
                    > old_generation
            );
        });
        cx.run_until_parked();
        let source = session.read_with(cx, |session, _| {
            assert_eq!(session.resource_state().revision(), 1);
            let source = session
                .resource_state()
                .committed()
                .expect("latest encoded source");
            assert_eq!(source.identity(), &png.image_identity);
            assert_eq!(source.descriptor(), &png.descriptor);
            source.clone()
        });
        let (item, window) = cx.add_window_view(|window, cx| {
            ImportedImageItem::from_session(session.clone(), 4096, window, cx)
        });
        let mut obsolete = encoded_load(&fixture, &png);
        obsolete.identity.version_id = "test-supplied-obsolete-encoded".into();
        session
            .update(window, |session, cx| session.load_encoded_png(obsolete, cx))
            .expect("encoded pending");
        let encoded_generation = session.read_with(window, |session, _| {
            session
                .resource_state()
                .active_encoded_request()
                .expect("encoded active")
                .generation()
        });
        let mut latest = known_load(&fixture, &png);
        latest.identity.version_id = "test-supplied-latest-known".into();
        let latest_identity = latest.identity.clone();
        session
            .update(window, |session, cx| session.load_png(latest, cx))
            .expect("known supersedes encoded");
        session.read_with(window, |session, _| {
            assert!(session.resource_state().active_encoded_request().is_none());
            assert!(
                session
                    .resource_state()
                    .active_request()
                    .expect("known active")
                    .generation()
                    > encoded_generation
            );
        });
        paint(window, &item);
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                .is_some()
        );
        window.run_until_parked();
        let committed = session.read_with(window, |session, _| {
            assert_eq!(session.resource_state().revision(), 2);
            assert!(!session.resource_state().has_active_request());
            let committed = session
                .resource_state()
                .committed()
                .expect("latest known completion");
            assert_eq!(committed.identity(), &latest_identity);
            assert_eq!(committed.image(), source.image());
            committed.clone()
        });
        let mut pending = encoded_load(&fixture, &png);
        pending.identity.version_id = "test-supplied-cancelled-encoded".into();
        session
            .update(window, |session, cx| session.load_encoded_png(pending, cx))
            .expect("encoded pending cancellation");
        item.update_in(window, |item, window, cx| {
            window.focus(&item.focus_handle(cx), cx)
        });
        paint(window, &item);
        window.dispatch_action(CancelImageLoad);
        window.run_until_parked();
        session.read_with(window, |session, _| {
            assert!(!session.resource_state().has_active_request());
            assert!(!session.resource_state().is_fresh());
            assert_eq!(session.resource_state().revision(), 2);
            assert!(Arc::ptr_eq(
                session.resource_state().committed().expect("last valid"),
                &committed
            ));
            assert_eq!(
                session.diagnostic(),
                Some(&crate::session::ImageSessionDiagnostic::Resource(
                    crate::resources::ImageResourceError::Cancelled
                ))
            );
        });
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-SESSION-ERROR")
                .is_some()
        );
    }

    #[gpui::test]
    fn at_graphite_encoded_png_item_failure_revoke_and_valid_recovery_preserve_source(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("real RGBA PNG");
        let session = cx.new(|_| GraphiteImageSession::new(4096));
        session
            .update(cx, |session, cx| {
                session.load_encoded_png(encoded_load(&fixture, &png), cx)
            })
            .expect("unknown dimension load");
        cx.run_until_parked();
        session.update(cx, |session, cx| {
            session.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        cx.run_until_parked();
        let (source, result) = session.read_with(cx, |session, _| {
            (
                session
                    .resource_state()
                    .committed()
                    .expect("source")
                    .clone(),
                session.result().expect("original edit").clone(),
            )
        });
        let (item, window) = cx.add_window_view(|window, cx| {
            ImportedImageItem::from_session(session.clone(), 4096, window, cx)
        });
        let preview = item.read_with(window, |item, cx| {
            item.presentation()
                .read(cx)
                .image()
                .expect("edited preview")
                .clone()
        });
        let mut incorrect = encoded_load(&fixture, &png);
        incorrect.identity.content_sha256[0] ^= 1;
        session
            .update(window, |session, cx| {
                session.load_encoded_png(incorrect, cx)
            })
            .expect("request with changed expected hash");
        window.run_until_parked();
        session.read_with(window, |session, _| {
            assert!(!session.resource_state().has_active_request());
            assert_eq!(session.resource_state().revision(), 1);
            assert!(Arc::ptr_eq(
                session
                    .resource_state()
                    .committed()
                    .expect("source retained"),
                &source
            ));
            assert!(Arc::ptr_eq(
                session.result().expect("last valid edit"),
                &result
            ));
            assert!(!session.result_is_fresh());
            assert_eq!(
                session.diagnostic(),
                Some(&crate::session::ImageSessionDiagnostic::Resource(
                    crate::resources::ImageResourceError::Failed(
                        graphics_core::resources::ImageAdmissionError::DigestMismatch.to_string()
                    )
                ))
            );
        });
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-SESSION-ERROR")
                .is_some()
        );
        assert_eq!(
            item.read_with(window, |item, cx| item
                .presentation()
                .read(cx)
                .image()
                .expect("retained preview")
                .id),
            preview.id
        );
        session
            .update(window, |session, cx| {
                session.revoke(&png.image_identity, cx)
            })
            .expect("revoke actual committed source");
        window.run_until_parked();
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-REVOKED").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert!(session.read_with(window, |session, _| session.result().is_none()));
        let mut pending = encoded_load(&fixture, &png);
        pending.identity.version_id = "test-supplied-revoked-pending".into();
        let pending_identity = pending.identity.clone();
        session
            .update(window, |session, cx| session.load_encoded_png(pending, cx))
            .expect("pending new generation");
        session
            .update(window, |session, cx| session.revoke(&pending_identity, cx))
            .expect("revocation cancels encoded pending request");
        window.run_until_parked();
        session.read_with(window, |session, _| {
            assert!(!session.resource_state().has_active_request());
            assert!(session.resource_state().is_revoked());
            assert_eq!(session.resource_state().revision(), 1);
        });
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-REVOKED").is_some());
        session
            .update(window, |session, cx| {
                session.load_encoded_png(encoded_load(&fixture, &png), cx)
            })
            .expect("explicit verified recovery");
        window.run_until_parked();
        session.read_with(window, |session, _| {
            assert!(session.resource_state().is_fresh());
            assert!(!session.resource_state().is_revoked());
            assert!(!session.resource_state().has_active_request());
            assert_eq!(session.resource_state().revision(), 2);
            assert_eq!(
                session
                    .resource_state()
                    .committed()
                    .expect("recovered")
                    .descriptor(),
                &png.descriptor
            );
            assert!(session.result_is_fresh());
            assert_eq!(
                session.result().expect("recovered edit").image(),
                result.image()
            );
            assert_eq!(session.diagnostic(), None);
        });
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-SESSION-ERROR")
                .is_none()
        );
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
    }
}

mod mix_session_consumer_tests {
    use super::*;
    use crate::resources::tests::{
        HostReaderFixture, IndexedRequestPng, host_reader_fixture, indexed_request_png,
    };
    use crate::session::{GraphiteEncodedPngLoad, GraphiteImageSession, GraphiteMixSession};
    use graphics_core::resources::{
        CpuBlendMode, CpuMixBudget, CpuMixError, CpuMixParameters, EncodedPngBudget,
    };
    use std::sync::Arc;

    fn load(
        fixture: &HostReaderFixture,
        png: &IndexedRequestPng,
        version: &str,
    ) -> GraphiteEncodedPngLoad {
        let mut identity = png.image_identity.clone();
        identity.version_id = version.into();
        GraphiteEncodedPngLoad {
            assets: fixture.assets.clone(),
            asset_identity: png.asset_identity.clone(),
            authorization: fixture.input_authorization.clone(),
            identity,
            transform: [1., 0., 0., 1., 0., 0.],
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
        }
    }

    fn budget() -> CpuMixBudget {
        CpuMixBudget {
            max_parent_bytes: 4096,
            max_result_bytes: 4096,
        }
    }

    fn parameters(opacity: f64) -> CpuMixParameters {
        CpuMixParameters {
            blend_mode: CpuBlendMode::Normal,
            opacity,
        }
    }

    #[gpui::test]
    fn at_graphite_mix_session_waits_for_real_parents_and_keeps_ordered_original_result(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let over_fixture = host_reader_fixture().expect("real Over service");
        let under_fixture = host_reader_fixture().expect("real Under service");
        let over_png = indexed_request_png(&over_fixture, 4).expect("Over original PNG");
        let under_png = indexed_request_png(&under_fixture, 4).expect("Under original PNG");
        let over = cx.new(|_| GraphiteImageSession::new(4096));
        let under = cx.new(|_| GraphiteImageSession::new(4096));
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                [over.clone(), under.clone()],
                parameters(37.5),
                budget(),
                cx,
            )
        });
        mix.read_with(cx, |mix, cx| {
            assert!(mix.result(cx).is_none());
            assert!(!mix.result_is_fresh(cx));
            assert!(!mix.is_pending());
        });
        over.update(cx, |parent, cx| {
            parent.load_encoded_png(load(&over_fixture, &over_png, "test-supplied-over-v1"), cx)
        })
        .expect("Over request");
        cx.run_until_parked();
        mix.read_with(cx, |mix, cx| {
            assert!(mix.result(cx).is_none());
            assert!(!mix.result_is_fresh(cx));
            assert!(!mix.is_pending());
            assert!(mix.diagnostic().expect("missing Under").contains("Under"));
        });
        under
            .update(cx, |parent, cx| {
                parent.load_encoded_png(
                    load(&under_fixture, &under_png, "test-supplied-under-v1"),
                    cx,
                )
            })
            .expect("Under request");
        cx.run_until_parked();
        mix.read_with(cx, |mix, cx| {
            assert_eq!(mix.parents()[0].entity_id(), over.entity_id());
            assert_eq!(mix.parents()[1].entity_id(), under.entity_id());
            assert!(mix.result_is_fresh(cx));
            assert!(!mix.is_pending());
            assert_eq!(mix.result_generation(), Some(mix.generation()));
            assert_eq!(mix.diagnostic(), None);
            let over_source = over
                .read(cx)
                .resource_state()
                .committed()
                .expect("Over decoded");
            let under_source = under
                .read(cx)
                .resource_state()
                .committed()
                .expect("Under decoded");
            let before = (over_source.image().clone(), under_source.image().clone());
            let original = graphics_core::resources::evaluate_cpu_image_mix(
                over_source,
                under_source,
                parameters(37.5),
                budget(),
            )
            .expect("original checked Mix");
            let result = mix.result(cx).expect("fresh original result");
            assert_eq!(result.image(), original.image());
            assert_eq!(result.attributes(), original.attributes());
            assert_eq!(result.pixels_sha256(), original.pixels_sha256());
            assert_eq!(result.parents()[0].identity(), over_source.identity());
            assert_eq!(result.parents()[1].identity(), under_source.identity());
            assert_eq!(result.parents()[0].origin(), over_source.origin());
            assert_eq!(result.parents()[1].origin(), under_source.origin());
            assert_eq!(result.parents()[0].descriptor(), &over_png.descriptor);
            assert_eq!(result.parents()[1].descriptor(), &under_png.descriptor);
            assert_eq!(over_source.image(), &before.0);
            assert_eq!(under_source.image(), &before.1);
        });
    }

    #[gpui::test]
    fn at_graphite_mix_session_each_parent_request_is_stale_before_revision_changes_and_revocation_hides(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixtures = [
            host_reader_fixture().expect("Over service"),
            host_reader_fixture().expect("Under service"),
        ];
        let pngs = [
            indexed_request_png(&fixtures[0], 4).expect("Over PNG"),
            indexed_request_png(&fixtures[1], 4).expect("Under PNG"),
        ];
        let parents = [
            cx.new(|_| GraphiteImageSession::new(4096)),
            cx.new(|_| GraphiteImageSession::new(4096)),
        ];
        for (index, parent) in parents.iter().enumerate() {
            parent
                .update(cx, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &fixtures[index],
                            &pngs[index],
                            &format!("test-supplied-role-{index}-v1"),
                        ),
                        cx,
                    )
                })
                .expect("real parent load");
        }
        cx.run_until_parked();
        let mix =
            cx.new(|cx| GraphiteMixSession::new(parents.clone(), parameters(50.), budget(), cx));
        cx.run_until_parked();
        for (index, parent) in parents.iter().enumerate() {
            let previous = mix.read_with(cx, |mix, cx| {
                mix.result(cx).expect("previous result").clone()
            });
            let revision = parent.read_with(cx, |parent, _| parent.resource_state().revision());
            parent
                .update(cx, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &fixtures[index],
                            &pngs[index],
                            &format!("test-supplied-role-{index}-v2"),
                        ),
                        cx,
                    )
                })
                .expect("changed parent version");
            parent.read_with(cx, |parent, _| {
                assert_eq!(parent.resource_state().revision(), revision);
                assert!(!parent.resource_state().is_fresh());
                assert!(parent.resource_state().has_active_request());
            });
            mix.read_with(cx, |mix, cx| {
                assert!(!mix.result_is_fresh(cx));
                assert!(Arc::ptr_eq(
                    mix.result(cx).expect("last valid result"),
                    &previous
                ));
            });
            cx.run_until_parked();
            let identity = parent.read_with(cx, |parent, _| {
                assert_eq!(parent.resource_state().revision(), revision + 1);
                parent
                    .resource_state()
                    .committed()
                    .expect("updated source")
                    .identity()
                    .clone()
            });
            mix.read_with(cx, |mix, cx| {
                assert!(mix.result_is_fresh(cx));
                assert!(!Arc::ptr_eq(mix.result(cx).expect("new result"), &previous));
                assert_eq!(
                    mix.result(cx).expect("ordered result").parents()[index].identity(),
                    &identity
                );
            });
            parent
                .update(cx, |parent, cx| parent.revoke(&identity, cx))
                .expect("revoke one actual parent");
            cx.run_until_parked();
            mix.read_with(cx, |mix, cx| {
                assert!(mix.is_revoked(cx));
                assert!(mix.result(cx).is_none());
                assert!(!mix.result_is_fresh(cx));
                assert!(!mix.is_pending());
            });
            parent
                .update(cx, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &fixtures[index],
                            &pngs[index],
                            &format!("test-supplied-role-{index}-v2"),
                        ),
                        cx,
                    )
                })
                .expect("explicit validated recovery");
            cx.run_until_parked();
            mix.read_with(cx, |mix, cx| {
                assert!(!mix.is_revoked(cx));
                assert!(mix.result_is_fresh(cx));
                assert_eq!(mix.diagnostic(), None);
            });
        }
    }

    #[gpui::test]
    fn at_graphite_mix_session_parameter_bits_supersede_cancel_retry_and_release(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("real PNG");
        let source = cx.new(|_| GraphiteImageSession::new(4096));
        source
            .update(cx, |source, cx| {
                source.load_encoded_png(load(&fixture, &png, "test-supplied-source-v1"), cx)
            })
            .expect("source load");
        cx.run_until_parked();
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                [source.clone(), source.clone()],
                parameters(25.),
                budget(),
                cx,
            )
        });
        let first_generation = mix.read_with(cx, |mix, _| {
            assert!(mix.is_pending());
            mix.generation()
        });
        mix.update(cx, |mix, cx| mix.set_parameters(parameters(50.), cx));
        let second_generation = mix.read_with(cx, |mix, _| {
            assert!(mix.is_pending());
            assert!(mix.generation() > first_generation);
            mix.generation()
        });
        mix.update(cx, |mix, cx| mix.set_parameters(parameters(0.), cx));
        assert!(mix.read_with(cx, |mix, _| mix.generation() > second_generation));
        cx.run_until_parked();
        mix.read_with(cx, |mix, cx| {
            assert!(mix.result_is_fresh(cx));
            assert_eq!(
                mix.result(cx)
                    .expect("latest parameters")
                    .parameters()
                    .opacity
                    .to_bits(),
                0_f64.to_bits()
            );
        });
        let generation = mix.read_with(cx, |mix, _| mix.generation());
        mix.update(cx, |mix, cx| mix.set_parameters(parameters(-0.), cx));
        assert!(mix.read_with(cx, |mix, _| mix.generation() > generation));
        cx.run_until_parked();
        let retained = mix.read_with(cx, |mix, cx| {
            assert_eq!(
                mix.result(cx)
                    .expect("bit-exact parameters")
                    .parameters()
                    .opacity
                    .to_bits(),
                (-0_f64).to_bits()
            );
            mix.result(cx).expect("last valid").clone()
        });
        mix.update(cx, |mix, cx| mix.set_parameters(parameters(75.), cx));
        mix.update(cx, |mix, cx| mix.cancel(cx));
        cx.run_until_parked();
        mix.read_with(cx, |mix, cx| {
            assert!(!mix.is_pending());
            assert!(!mix.result_is_fresh(cx));
            assert!(Arc::ptr_eq(
                mix.result(cx).expect("retained after cancellation"),
                &retained
            ));
            assert_eq!(mix.diagnostic(), Some("Mix evaluation cancelled"));
        });
        mix.update(cx, |mix, cx| mix.refresh(cx));
        cx.run_until_parked();
        mix.read_with(cx, |mix, cx| {
            assert!(mix.result_is_fresh(cx));
            assert_eq!(mix.result(cx).expect("retry").parameters(), parameters(75.));
            assert_eq!(mix.diagnostic(), None);
        });
        mix.update(cx, |mix, cx| mix.set_parameters(parameters(99.), cx));
        assert!(mix.read_with(cx, |mix, _| mix.is_pending()));
        let weak = mix.downgrade();
        drop(mix);
        cx.update(|_| ());
        cx.run_until_parked();
        assert!(weak.upgrade().is_none());
        source.read_with(cx, |source, _| {
            assert!(source.resource_state().is_fresh());
            assert_eq!(source.resource_state().revision(), 1);
            assert_eq!(source.diagnostic(), None);
        });
    }

    #[gpui::test]
    fn at_graphite_mix_session_duplicate_parent_owner_has_one_current_evaluation_without_restart(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("PNG");
        let source = cx.new(|_| GraphiteImageSession::new(4096));
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                [source.clone(), source.clone()],
                parameters(100.),
                budget(),
                cx,
            )
        });
        source
            .update(cx, |source, cx| {
                source.load_encoded_png(load(&fixture, &png, "test-supplied-duplicate-v1"), cx)
            })
            .expect("one source owner");
        cx.run_until_parked();
        let (generation, result) = mix.read_with(cx, |mix, cx| {
            assert!(mix.result_is_fresh(cx));
            assert_eq!(mix.parents()[0].entity_id(), source.entity_id());
            assert_eq!(mix.parents()[1].entity_id(), source.entity_id());
            let result = mix.result(cx).expect("duplicate roles");
            assert_eq!(
                result.parents()[0].identity(),
                result.parents()[1].identity()
            );
            assert_eq!(
                result.parents()[0].pixels_sha256(),
                result.parents()[1].pixels_sha256()
            );
            (mix.generation(), result.clone())
        });
        for _ in 0..3 {
            mix.update(cx, |mix, cx| mix.refresh(cx));
        }
        cx.run_until_parked();
        mix.read_with(cx, |mix, cx| {
            assert_eq!(mix.generation(), generation);
            assert!(!mix.is_pending());
            assert!(Arc::ptr_eq(
                mix.result(cx).expect("exact cached evaluation"),
                &result
            ));
        });
        mix.update(cx, |mix, cx| mix.set_parameters(parameters(37.5), cx));
        let pending_generation = mix.read_with(cx, |mix, _| mix.generation());
        for _ in 0..3 {
            mix.update(cx, |mix, cx| mix.refresh(cx));
        }
        assert_eq!(
            mix.read_with(cx, |mix, _| mix.generation()),
            pending_generation
        );
        cx.run_until_parked();
        mix.read_with(cx, |mix, cx| {
            assert!(mix.result_is_fresh(cx));
            assert_eq!(mix.result_generation(), Some(pending_generation));
            assert!(!Arc::ptr_eq(
                mix.result(cx).expect("parameter replacement"),
                &result
            ));
        });
    }

    #[gpui::test]
    fn at_graphite_mix_session_shape_failure_retains_last_valid_then_recovers(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let over_fixture = host_reader_fixture().expect("Over service");
        let under_fixture = host_reader_fixture().expect("Under service");
        let over_png = indexed_request_png(&over_fixture, 4).expect("three-pixel RGBA Over");
        let under_png = indexed_request_png(&under_fixture, 4).expect("three-pixel RGBA Under");
        let over = cx.new(|_| GraphiteImageSession::new(4096));
        let under = cx.new(|_| GraphiteImageSession::new(4096));
        over.update(cx, |parent, cx| {
            parent.load_encoded_png(load(&over_fixture, &over_png, "test-supplied-over-v1"), cx)
        })
        .expect("Over load");
        under
            .update(cx, |parent, cx| {
                parent.load_encoded_png(
                    load(&under_fixture, &under_png, "test-supplied-under-v1"),
                    cx,
                )
            })
            .expect("Under load");
        cx.run_until_parked();
        let mix = cx.new(|cx| {
            GraphiteMixSession::new([over.clone(), under.clone()], parameters(50.), budget(), cx)
        });
        cx.run_until_parked();
        let retained = mix.read_with(cx, |mix, cx| {
            mix.result(cx).expect("last valid equal shape").clone()
        });
        let mismatched =
            indexed_request_png(&under_fixture, 3).expect("actual two-pixel RGB replacement");
        assert_ne!(mismatched.descriptor.width, over_png.descriptor.width);
        under
            .update(cx, |parent, cx| {
                parent.load_encoded_png(
                    load(&under_fixture, &mismatched, "test-supplied-under-small"),
                    cx,
                )
            })
            .expect("actual changed shape");
        cx.run_until_parked();
        mix.read_with(cx, |mix, cx| {
            assert!(!mix.result_is_fresh(cx));
            assert!(!mix.is_pending());
            assert!(Arc::ptr_eq(
                mix.result(cx).expect("preserved last-valid result"),
                &retained
            ));
            assert_eq!(
                mix.diagnostic(),
                Some(CpuMixError::ParentShapeMismatch.to_string().as_str())
            );
        });
        let restored =
            indexed_request_png(&under_fixture, 4).expect("actual restored RGBA producer");
        under
            .update(cx, |parent, cx| {
                parent.load_encoded_png(
                    load(&under_fixture, &restored, "test-supplied-under-restored"),
                    cx,
                )
            })
            .expect("shape recovery");
        cx.run_until_parked();
        mix.read_with(cx, |mix, cx| {
            assert!(mix.result_is_fresh(cx));
            assert_eq!(mix.diagnostic(), None);
            assert_eq!(
                mix.result(cx).expect("fresh recovery").image(),
                retained.image()
            );
            assert_eq!(
                mix.result(cx).expect("new provenance").parents()[1]
                    .identity()
                    .version_id
                    .as_ref(),
                "test-supplied-under-restored"
            );
        });
    }

    fn refresh_preview(
        window: &mut gpui::VisualTestContext,
        presentation: &gpui::Entity<crate::image_preview::CpuPreviewPresentation>,
        mix: &gpui::Entity<GraphiteMixSession>,
    ) -> Result<(), crate::image_preview::CpuPreviewError> {
        let (parents, result, parameters, generation, result_generation, fresh) =
            mix.read_with(window, |mix, cx| {
                (
                    mix.parents().each_ref().map(|parent| {
                        crate::image_preview::CpuPreviewParent::from_state(
                            parent.read(cx).resource_state(),
                        )
                    }),
                    mix.result(cx).cloned(),
                    mix.parameters(),
                    mix.generation(),
                    mix.result_generation(),
                    mix.result_is_fresh(cx),
                )
            });
        window.update(|window, cx| {
            presentation.update(cx, |presentation, cx| {
                presentation.refresh_mix(
                    crate::image_preview::MixPreviewRequest {
                        parents,
                        result: result.as_ref(),
                        parameters,
                        current_generation: generation,
                        result_generation,
                        result_is_fresh: fresh,
                    },
                    4096,
                    window,
                    cx,
                )
            })
        })
    }

    fn paint_preview(
        window: &mut gpui::VisualTestContext,
        presentation: &gpui::Entity<crate::image_preview::CpuPreviewPresentation>,
    ) {
        use gpui::IntoElement as _;
        window.draw(
            gpui::point(gpui::px(0.), gpui::px(0.)),
            gpui::size(gpui::px(640.), gpui::px(480.)),
            |_, _| presentation.clone().into_any_element(),
        );
    }

    #[gpui::test]
    fn at_graphite_mix_preview_reuses_exact_result_and_replaces_for_either_parent_then_hides_revocation(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixtures = [
            host_reader_fixture().expect("Over service"),
            host_reader_fixture().expect("Under service"),
        ];
        let pngs = [
            indexed_request_png(&fixtures[0], 4).expect("Over PNG"),
            indexed_request_png(&fixtures[1], 4).expect("Under PNG"),
        ];
        let parents = [
            cx.new(|_| GraphiteImageSession::new(4096)),
            cx.new(|_| GraphiteImageSession::new(4096)),
        ];
        for (index, parent) in parents.iter().enumerate() {
            parent
                .update(cx, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &fixtures[index],
                            &pngs[index],
                            &format!("test-supplied-preview-{index}-v1"),
                        ),
                        cx,
                    )
                })
                .expect("parent load");
        }
        cx.run_until_parked();
        let mix =
            cx.new(|cx| GraphiteMixSession::new(parents.clone(), parameters(37.5), budget(), cx));
        cx.run_until_parked();
        let presentation = cx.new(crate::image_preview::CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        refresh_preview(window, &presentation, &mix).expect("current Mix preview");
        paint_preview(window, &presentation);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        let first = presentation.read_with(window, |presentation, _| {
            presentation.image().expect("pixels").clone()
        });
        refresh_preview(window, &presentation, &mix).expect("same result redraw");
        assert_eq!(
            presentation.read_with(window, |presentation, _| presentation
                .image()
                .expect("reuse")
                .id),
            first.id
        );
        let original = mix.read_with(window, |mix, cx| {
            mix.result(cx)
                .expect("original result")
                .image()
                .to_flat_u8()
                .0
        });
        let mut expected = original;
        for pixel in expected.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }
        assert_eq!(first.as_bytes(0), Some(expected.as_slice()));
        for (index, parent) in parents.iter().enumerate() {
            let prior = presentation.read_with(window, |presentation, _| {
                presentation.image().expect("prior image").clone()
            });
            let old_revision =
                parent.read_with(window, |parent, _| parent.resource_state().revision());
            parent
                .update(window, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &fixtures[index],
                            &pngs[index],
                            &format!("test-supplied-preview-{index}-v2"),
                        ),
                        cx,
                    )
                })
                .expect("parent change");
            assert_eq!(
                parent.read_with(window, |parent, _| parent.resource_state().revision()),
                old_revision
            );
            assert_eq!(
                refresh_preview(window, &presentation, &mix),
                Err(crate::image_preview::CpuPreviewError::ParentNotFresh)
            );
            paint_preview(window, &presentation);
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_some()
            );
            assert_eq!(
                presentation.read_with(window, |presentation, _| presentation
                    .image()
                    .expect("last valid")
                    .id),
                prior.id
            );
            window.run_until_parked();
            refresh_preview(window, &presentation, &mix).expect("both current parents and result");
            paint_preview(window, &presentation);
            let replacement = presentation.read_with(window, |presentation, _| {
                assert!(presentation.is_fresh());
                presentation.image().expect("replacement").clone()
            });
            assert_ne!(replacement.id, prior.id);
            assert!(!window.update(|window, _| window.has_image_atlas_entry(&prior)));
            assert!(window.update(|window, _| window.has_image_atlas_entry(&replacement)));
            let identity = parent.read_with(window, |parent, _| {
                parent
                    .resource_state()
                    .committed()
                    .expect("current source")
                    .identity()
                    .clone()
            });
            parent
                .update(window, |parent, cx| parent.revoke(&identity, cx))
                .expect("actual parent revoke");
            window.run_until_parked();
            refresh_preview(window, &presentation, &mix)
                .expect("revocation clears preview without conversion");
            paint_preview(window, &presentation);
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                    .is_some()
            );
            assert!(
                presentation.read_with(window, |presentation, _| presentation.image().is_none())
            );
            assert!(!window.update(|window, _| window.has_image_atlas_entry(&replacement)));
            parent
                .update(window, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &fixtures[index],
                            &pngs[index],
                            &format!("test-supplied-preview-{index}-v2"),
                        ),
                        cx,
                    )
                })
                .expect("explicit validated parent recovery");
            window.run_until_parked();
            refresh_preview(window, &presentation, &mix).expect("recovered current result");
            paint_preview(window, &presentation);
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_none()
            );
        }
    }

    #[gpui::test]
    fn at_graphite_mix_preview_source_result_and_generation_keys_stay_distinct_through_retry(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("actual PNG");
        let source = cx.new(|_| GraphiteImageSession::new(4096));
        source
            .update(cx, |source, cx| {
                source.load_encoded_png(load(&fixture, &png, "test-supplied-shared-preview"), cx)
            })
            .expect("source load");
        cx.run_until_parked();
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                [source.clone(), source.clone()],
                parameters(50.),
                budget(),
                cx,
            )
        });
        cx.run_until_parked();
        let presentation = cx.new(crate::image_preview::CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    presentation.update(cx, |presentation, cx| {
                        presentation.refresh(source.resource_state(), 4096, window, cx)
                    })
                })
            })
            .expect("original source path");
        let source_image = presentation.read_with(window, |presentation, _| {
            presentation.image().expect("source pixels").clone()
        });
        refresh_preview(window, &presentation, &mix).expect("same parents now Mix result path");
        let mixed = presentation.read_with(window, |presentation, _| {
            presentation.image().expect("Mix pixels").clone()
        });
        assert_ne!(mixed.id, source_image.id);
        refresh_preview(window, &presentation, &mix).expect("exact Mix redraw");
        assert_eq!(
            presentation.read_with(window, |presentation, _| presentation
                .image()
                .expect("cached")
                .id),
            mixed.id
        );
        let original_source = source.read_with(window, |source, _| {
            source.resource_state().committed().expect("source").clone()
        });
        let unary = Arc::new(
            graphics_core::resources::evaluate_cpu_image_edit(
                &original_source,
                graphics_core::resources::CpuImageEdit::Invert,
                4096,
            )
            .expect("original unary result"),
        );
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    presentation.update(cx, |presentation, cx| {
                        presentation.refresh_result(
                            source.resource_state(),
                            &unary,
                            4096,
                            window,
                            cx,
                        )
                    })
                })
            })
            .expect("existing unary owner path");
        let unary_image = presentation.read_with(window, |presentation, _| {
            presentation.image().expect("unary pixels").clone()
        });
        assert_ne!(unary_image.id, mixed.id);
        refresh_preview(window, &presentation, &mix).expect("Mix cannot reuse unary cache entry");
        let before_pending = presentation.read_with(window, |presentation, _| {
            presentation.image().expect("restored Mix pixels").clone()
        });
        assert_ne!(before_pending.id, unary_image.id);
        mix.update(window, |mix, cx| mix.set_parameters(parameters(-0.), cx));
        assert_eq!(
            refresh_preview(window, &presentation, &mix),
            Err(crate::image_preview::CpuPreviewError::ResultNotFresh)
        );
        paint_preview(window, &presentation);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert_eq!(
            presentation.read_with(window, |presentation, _| presentation
                .image()
                .expect("last valid")
                .id),
            before_pending.id
        );
        mix.update(window, |mix, cx| mix.cancel(cx));
        window.run_until_parked();
        assert_eq!(
            refresh_preview(window, &presentation, &mix),
            Err(crate::image_preview::CpuPreviewError::ResultNotFresh)
        );
        mix.update(window, |mix, cx| mix.refresh(cx));
        window.run_until_parked();
        refresh_preview(window, &presentation, &mix).expect("fresh generation after retry");
        paint_preview(window, &presentation);
        let replacement = presentation.read_with(window, |presentation, _| {
            assert!(presentation.is_fresh());
            presentation.image().expect("new generation").clone()
        });
        assert_ne!(replacement.id, before_pending.id);
        mix.read_with(window, |mix, cx| {
            assert!(mix.result_is_fresh(cx));
            assert_eq!(mix.result_generation(), Some(mix.generation()));
            assert_eq!(
                mix.result(cx)
                    .expect("bitwise parameters")
                    .parameters()
                    .opacity
                    .to_bits(),
                (-0_f64).to_bits()
            );
        });
        refresh_preview(window, &presentation, &mix).expect("stable replacement redraw");
        assert_eq!(
            presentation.read_with(window, |presentation, _| presentation
                .image()
                .expect("exact reuse")
                .id),
            replacement.id
        );
        assert!(Arc::ptr_eq(
            &source.read_with(window, |source, _| source
                .resource_state()
                .committed()
                .expect("unchanged source")
                .clone()),
            &original_source
        ));
    }
}

mod mix_item_consumer_tests {
    use super::*;
    use crate::graph::{CancelMixEvaluation, GraphiteMixItem, RetryMixEvaluation};
    use crate::resources::tests::{
        HostReaderFixture, IndexedRequestPng, host_reader_fixture, indexed_request_png,
    };
    use crate::session::{GraphiteEncodedPngLoad, GraphiteImageSession, GraphiteMixSession};
    use gpui::VisualTestContext;
    use graphics_core::resources::{
        CpuBlendMode, CpuMixBudget, CpuMixParameters, EncodedPngBudget,
    };
    use std::sync::Arc;
    use workspace::item::Item as _;

    fn load(
        fixture: &HostReaderFixture,
        png: &IndexedRequestPng,
        version: &str,
    ) -> GraphiteEncodedPngLoad {
        let mut identity = png.image_identity.clone();
        identity.version_id = version.into();
        GraphiteEncodedPngLoad {
            assets: fixture.assets.clone(),
            asset_identity: png.asset_identity.clone(),
            authorization: fixture.input_authorization.clone(),
            identity,
            transform: [1., 0., 0., 1., 0., 0.],
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
        }
    }

    fn paint(window: &mut VisualTestContext) {
        window.update(|window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        });
    }

    #[gpui::test]
    async fn at_graphite_mix_item_workspace_opacity_edit_invalid_draft_recovery_and_cancel_actions(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
        });
        let fixture = host_reader_fixture().expect("real PNG service");
        let png = indexed_request_png(&fixture, 4).expect("original producer PNG");
        let source = cx.new(|_| GraphiteImageSession::new(4096));
        source
            .update(cx, |source, cx| {
                source.load_encoded_png(load(&fixture, &png, "test-supplied-native-Mix-input"), cx)
            })
            .expect("actual input load");
        cx.run_until_parked();
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                [source.clone(), source.clone()],
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: 100.,
                },
                CpuMixBudget {
                    max_parent_bytes: 4096,
                    max_result_bytes: 4096,
                },
                cx,
            )
        });
        cx.run_until_parked();
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (host, window) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = host.read_with(window, |host, _| host.workspace().clone());
        let item = workspace.update_in(window, |workspace, window, cx| {
            let item = cx.new(|cx| GraphiteMixItem::from_session(mix.clone(), 4096, window, cx));
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
            window.focus(&item.focus_handle(cx), cx);
            item
        });
        window.update(|_, cx| {
            let confirm = cx
                .build_action("menu::Confirm", None)
                .expect("existing native confirm action");
            cx.bind_keys([gpui::KeyBinding::load(
                "enter",
                confirm,
                None,
                false,
                None,
                &gpui::DummyKeyboardMapper,
            )
            .expect("native opacity Enter binding")]);
        });
        window.run_until_parked();
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_item_as::<GraphiteMixItem>(cx)
                .expect("actual native Workspace Item")
                .entity_id()),
            item.entity_id()
        );
        assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        let editor = item.read_with(window, |item, _| item.opacity_editor.clone());
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.simulate_input("25");
        window.run_until_parked();
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let (generation, valid) = mix.read_with(window, |mix, cx| {
            assert_eq!(mix.parameters().opacity.to_bits(), 25_f64.to_bits());
            assert!(mix.result_is_fresh(cx));
            let input = source
                .read(cx)
                .resource_state()
                .committed()
                .expect("original source");
            let expected = graphics_core::resources::evaluate_cpu_image_mix(
                input,
                input,
                mix.parameters(),
                CpuMixBudget {
                    max_parent_bytes: 4096,
                    max_result_bytes: 4096,
                },
            )
            .expect("original checked Mix");
            assert_eq!(
                mix.result(cx).expect("native edited result").image(),
                expected.image()
            );
            (
                mix.generation(),
                mix.result(cx).expect("last-valid").clone(),
            )
        });
        let pixels = item.read_with(window, |item, cx| {
            item.presentation()
                .read(cx)
                .image()
                .expect("prepared result")
                .clone()
        });
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("not a number", window, cx);
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.run_until_parked();
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        mix.read_with(window, |mix, cx| {
            assert_eq!(mix.generation(), generation);
            assert_eq!(mix.parameters().opacity.to_bits(), 25_f64.to_bits());
            assert!(Arc::ptr_eq(
                mix.result(cx).expect("preserved result"),
                &valid
            ));
        });
        assert!(window.debug_bounds("GRAPHITE-MIX-LAST-VALID").is_some());
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        assert_eq!(
            item.read_with(window, |item, cx| item
                .presentation()
                .read(cx)
                .image()
                .expect("retained pixels")
                .id),
            pixels.id
        );
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.simulate_input("75");
        window.run_until_parked();
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        mix.read_with(window, |mix, cx| {
            assert_eq!(mix.parameters().opacity.to_bits(), 75_f64.to_bits());
            assert!(mix.result_is_fresh(cx));
            assert!(!Arc::ptr_eq(
                mix.result(cx).expect("recovered result"),
                &valid
            ));
        });
        assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        mix.update(window, |mix, cx| {
            let mut parameters = mix.parameters();
            parameters.opacity = 99.;
            mix.set_parameters(parameters, cx);
        });
        item.update_in(window, |item, window, cx| {
            window.focus(&item.focus_handle(cx), cx)
        });
        window.dispatch_action(CancelMixEvaluation);
        window.run_until_parked();
        mix.read_with(window, |mix, cx| {
            assert!(!mix.is_pending());
            assert!(!mix.result_is_fresh(cx));
            assert_eq!(mix.diagnostic(), Some("Mix evaluation cancelled"));
        });
        assert!(window.debug_bounds("GRAPHITE-MIX-LAST-VALID").is_some());
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        window.dispatch_action(RetryMixEvaluation);
        window.run_until_parked();
        mix.read_with(window, |mix, cx| {
            assert!(mix.result_is_fresh(cx));
            assert_eq!(
                mix.result(cx)
                    .expect("actual retry")
                    .parameters()
                    .opacity
                    .to_bits(),
                99_f64.to_bits()
            );
            assert_eq!(mix.diagnostic(), None);
        });
        assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
    }

    #[gpui::test]
    fn at_graphite_mix_item_either_real_parent_revoke_hides_pixels_until_verified_recovery(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        let fixtures = [
            host_reader_fixture().expect("Over service"),
            host_reader_fixture().expect("Under service"),
        ];
        let pngs = [
            indexed_request_png(&fixtures[0], 4).expect("Over PNG"),
            indexed_request_png(&fixtures[1], 4).expect("Under PNG"),
        ];
        let parents = [
            cx.new(|_| GraphiteImageSession::new(4096)),
            cx.new(|_| GraphiteImageSession::new(4096)),
        ];
        for (index, parent) in parents.iter().enumerate() {
            parent
                .update(cx, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &fixtures[index],
                            &pngs[index],
                            &format!("test-supplied-native-parent-{index}"),
                        ),
                        cx,
                    )
                })
                .expect("actual parent load");
        }
        cx.run_until_parked();
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                parents.clone(),
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: 50.,
                },
                CpuMixBudget {
                    max_parent_bytes: 4096,
                    max_result_bytes: 4096,
                },
                cx,
            )
        });
        cx.run_until_parked();
        let (item, window) = cx.add_window_view(|window, cx| {
            GraphiteMixItem::from_session(mix.clone(), 4096, window, cx)
        });
        for (index, parent) in parents.iter().enumerate() {
            let image = item.read_with(window, |item, cx| {
                item.presentation()
                    .read(cx)
                    .image()
                    .expect("current composition")
                    .clone()
            });
            assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
            let identity = parent.read_with(window, |parent, _| {
                parent
                    .resource_state()
                    .committed()
                    .expect("original parent")
                    .identity()
                    .clone()
            });
            parent
                .update(window, |parent, cx| parent.revoke(&identity, cx))
                .expect("real source revocation");
            window.run_until_parked();
            assert!(window.debug_bounds("GRAPHITE-MIX-REVOKED").is_some());
            assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            assert!(item.read_with(window, |item, cx| {
                item.presentation().read(cx).image().is_none()
            }));
            assert!(!window.update(|window, _| window.has_image_atlas_entry(&image)));
            parent
                .update(window, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &fixtures[index],
                            &pngs[index],
                            &format!("test-supplied-native-parent-{index}"),
                        ),
                        cx,
                    )
                })
                .expect("explicit actual recovery");
            window.run_until_parked();
            assert!(mix.read_with(window, |mix, cx| mix.result_is_fresh(cx)));
            assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
            assert!(window.debug_bounds("GRAPHITE-MIX-REVOKED").is_none());
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        }
    }

    #[gpui::test]
    async fn at_graphite_mix_item_split_shares_owners_with_distinct_editors_and_final_atlas_release(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
        });
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("actual PNG");
        let source = cx.new(|_| GraphiteImageSession::new(4096));
        source
            .update(cx, |source, cx| {
                source.load_encoded_png(load(&fixture, &png, "test-supplied-split-source"), cx)
            })
            .expect("original load");
        cx.run_until_parked();
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                [source.clone(), source.clone()],
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: 50.,
                },
                CpuMixBudget {
                    max_parent_bytes: 4096,
                    max_result_bytes: 4096,
                },
                cx,
            )
        });
        cx.run_until_parked();
        let weak_mix = mix.downgrade();
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let mut second_context = cx.clone();
        let (first_host, first_window) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let first_workspace =
            first_host.read_with(first_window, |host, _| host.workspace().clone());
        let app_state =
            first_workspace.read_with(first_window, |workspace, _| workspace.app_state().clone());
        let (second_host, second_window) = second_context.add_window_view(|window, cx| {
            let workspace =
                cx.new(|cx| workspace::Workspace::new(None, project, app_state, window, cx));
            workspace::MultiWorkspace::new(workspace, window, cx)
        });
        let second_workspace =
            second_host.read_with(second_window, |host, _| host.workspace().clone());
        let first = first_workspace.update_in(first_window, |workspace, window, cx| {
            let item = cx.new(|cx| GraphiteMixItem::from_session(mix.clone(), 4096, window, cx));
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
            window.focus(&item.focus_handle(cx), cx);
            item
        });
        let split = first
            .update_in(second_window, |item, window, cx| {
                item.clone_on_split(None, window, cx)
            })
            .await
            .expect("native Item split");
        second_workspace.update_in(second_window, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx);
            window.focus(&split.focus_handle(cx), cx);
        });
        assert_eq!(
            first_workspace.read_with(first_window, |workspace, cx| workspace
                .active_item_as::<GraphiteMixItem>(cx)
                .expect("first pane owns original Item")
                .entity_id()),
            first.entity_id()
        );
        assert_eq!(
            second_workspace.read_with(second_window, |workspace, cx| workspace
                .active_item_as::<GraphiteMixItem>(cx)
                .expect("second pane owns split Item")
                .entity_id()),
            split.entity_id()
        );
        let (session_id, presentation_id, editor_id, focus, weak_presentation) =
            first.read_with(first_window, |item, cx| {
                (
                    item.session().entity_id(),
                    item.presentation().entity_id(),
                    item.opacity_editor.entity_id(),
                    item.focus_handle(cx),
                    item.presentation().downgrade(),
                )
            });
        split.read_with(second_window, |item, cx| {
            assert_eq!(item.session().entity_id(), session_id);
            assert_eq!(item.presentation().entity_id(), presentation_id);
            assert_ne!(item.opacity_editor.entity_id(), editor_id);
            assert_ne!(item.focus_handle(cx), focus);
        });
        paint(first_window);
        paint(second_window);
        let image = first.read_with(first_window, |item, cx| {
            item.presentation()
                .read(cx)
                .image()
                .expect("shared pixels")
                .clone()
        });
        assert_eq!(
            split.read_with(second_window, |item, cx| item
                .presentation()
                .read(cx)
                .image()
                .expect("same presentation")
                .id),
            image.id
        );
        assert!(first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(second_window.update(|window, _| window.has_image_atlas_entry(&image)));
        let split_editor = split.read_with(second_window, |item, _| item.opacity_editor.clone());
        let result = mix.read_with(first_window, |mix, cx| {
            mix.result(cx).expect("committed composition").clone()
        });
        let generation = mix.read_with(first_window, |mix, _| mix.generation());
        second_window.update(|_, cx| {
            let confirm = cx
                .build_action("menu::Confirm", None)
                .expect("existing native confirm action");
            cx.bind_keys([gpui::KeyBinding::load(
                "enter",
                confirm,
                None,
                false,
                None,
                &gpui::DummyKeyboardMapper,
            )
            .expect("native opacity Enter binding")]);
        });
        second_window.update(|window, cx| {
            window.activate_window();
            split_editor.update(cx, |editor, cx| {
                editor.set_text("invalid opacity", window, cx);
                window.focus(&editor.focus_handle(cx), cx);
            });
        });
        second_window.run_until_parked();
        paint(second_window);
        assert!(second_window.update(|window, cx| split_editor.read(cx).is_focused(window)));
        second_window.simulate_keystrokes("enter");
        second_window.run_until_parked();
        assert!(!second_window.update(|window, cx| split_editor.read(cx).is_focused(window)));
        for invalid_pane_first in [true, false] {
            if invalid_pane_first {
                paint(second_window);
                paint(first_window);
            } else {
                paint(first_window);
                paint(second_window);
            }
            assert!(
                second_window
                    .debug_bounds("GRAPHITE-MIX-LAST-VALID")
                    .is_some()
            );
            assert!(second_window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
            assert!(first_window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
            assert!(first_window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
            first.read_with(first_window, |item, cx| {
                let presentation = item.presentation().read(cx);
                assert!(presentation.is_fresh());
                assert_eq!(
                    presentation.image().expect("stable shared pixels").id,
                    image.id
                );
            });
            mix.read_with(first_window, |mix, cx| {
                assert_eq!(mix.generation(), generation);
                assert_eq!(mix.parameters().opacity.to_bits(), 50_f64.to_bits());
                assert!(mix.result_is_fresh(cx));
                assert!(Arc::ptr_eq(
                    mix.result(cx).expect("unchanged result"),
                    &result
                ));
            });
        }
        second_window.update(|window, cx| {
            window.activate_window();
            split_editor.update(cx, |editor, cx| {
                editor.set_text("", window, cx);
                window.focus(&editor.focus_handle(cx), cx);
            });
        });
        second_window.simulate_input("75");
        second_window.run_until_parked();
        second_window.simulate_keystrokes("enter");
        second_window.run_until_parked();
        paint(first_window);
        paint(second_window);
        assert!(second_window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
        assert!(second_window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        assert_eq!(
            mix.read_with(first_window, |mix, _| mix.parameters().opacity.to_bits()),
            75_f64.to_bits()
        );
        assert_eq!(
            first.read_with(first_window, |item, cx| item
                .opacity_editor
                .read(cx)
                .text(cx)),
            "75"
        );
        let replacement = first.read_with(first_window, |item, cx| {
            item.presentation()
                .read(cx)
                .image()
                .expect("shared replacement")
                .clone()
        });
        assert_ne!(replacement.id, image.id);
        assert_eq!(
            split.read_with(second_window, |item, cx| item
                .presentation()
                .read(cx)
                .image()
                .expect("same replacement")
                .id),
            replacement.id
        );
        drop(split_editor);
        drop(mix);
        first_workspace
            .update_in(first_window, |workspace, window, cx| {
                workspace.active_pane().clone().update(cx, |pane, cx| {
                    pane.close_item_by_id(
                        first.entity_id(),
                        workspace::SaveIntent::Close,
                        window,
                        cx,
                    )
                })
            })
            .await
            .expect("native pane close succeeds");
        assert!(first_workspace.read_with(first_window, |workspace, cx| {
            workspace.active_item_as::<GraphiteMixItem>(cx).is_none()
        }));
        drop(first);
        paint(first_window);
        first_window.run_until_parked();
        assert!(weak_mix.upgrade().is_some());
        assert!(weak_presentation.upgrade().is_some());
        paint(second_window);
        second_workspace
            .update_in(second_window, |workspace, window, cx| {
                workspace.active_pane().clone().update(cx, |pane, cx| {
                    pane.close_item_by_id(
                        split.entity_id(),
                        workspace::SaveIntent::Close,
                        window,
                        cx,
                    )
                })
            })
            .await
            .expect("native pane close succeeds");
        assert!(second_workspace.read_with(second_window, |workspace, cx| {
            workspace.active_item_as::<GraphiteMixItem>(cx).is_none()
        }));
        second_window.update(|_, _| drop(split));
        paint(first_window);
        paint(second_window);
        second_window.run_until_parked();
        assert!(weak_mix.upgrade().is_none());
        assert!(weak_presentation.upgrade().is_none());
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(&replacement)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&replacement)));
        assert!(source.read_with(first_window, |source, _| source.resource_state().is_fresh()));
    }
}

mod mix_png_session_consumer_tests {
    use super::*;
    use crate::resources::tests::{
        HostReaderFixture, IndexedRequestPng, host_reader_fixture, indexed_request_png,
    };
    use crate::session::{GraphiteEncodedPngLoad, GraphiteImageSession, GraphiteMixSession};
    use gpui::Entity;
    use graphics_core::resources::{
        AlphaAssociation, CpuBlendMode, CpuMixBudget, CpuMixParameters, CpuSampleFormat,
        EncodedPngBudget, ImageColorSpace,
    };
    use std::sync::Arc;

    struct ReadyMix {
        fixtures: [HostReaderFixture; 2],
        pngs: [IndexedRequestPng; 2],
        parents: [Entity<GraphiteImageSession>; 2],
        mix: Entity<GraphiteMixSession>,
    }

    fn load(
        fixture: &HostReaderFixture,
        png: &IndexedRequestPng,
        version: &str,
    ) -> GraphiteEncodedPngLoad {
        let mut identity = png.image_identity.clone();
        identity.version_id = version.into();
        GraphiteEncodedPngLoad {
            assets: fixture.assets.clone(),
            asset_identity: png.asset_identity.clone(),
            authorization: fixture.input_authorization.clone(),
            identity,
            transform: [1., 0., 0., 1., 0., 0.],
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
        }
    }

    fn ready_mix(cx: &mut TestAppContext) -> ReadyMix {
        init(cx);
        let fixtures = [
            host_reader_fixture().expect("real Over service"),
            host_reader_fixture().expect("real Under service"),
        ];
        let pngs = [
            indexed_request_png(&fixtures[0], 4).expect("actual Over PNG"),
            indexed_request_png(&fixtures[1], 4).expect("actual Under PNG"),
        ];
        let parents = [
            cx.new(|_| GraphiteImageSession::new(4096)),
            cx.new(|_| GraphiteImageSession::new(4096)),
        ];
        for (index, parent) in parents.iter().enumerate() {
            parent
                .update(cx, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &fixtures[index],
                            &pngs[index],
                            &format!("test-supplied-PNG-parent-{index}-v1"),
                        ),
                        cx,
                    )
                })
                .expect("real indexed PNG load");
        }
        cx.run_until_parked();
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                parents.clone(),
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: 37.5,
                },
                CpuMixBudget {
                    max_parent_bytes: 4096,
                    max_result_bytes: 4096,
                },
                cx,
            )
        });
        cx.run_until_parked();
        assert!(mix.read_with(cx, |mix, cx| mix.result_is_fresh(cx)));
        ReadyMix {
            fixtures,
            pngs,
            parents,
            mix,
        }
    }

    #[gpui::test]
    async fn at_graphite_mix_session_png_preparation_exact_original_bytes_provenance_caps_and_repeat(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_mix(cx);
        let sources = ready.parents.each_ref().map(|parent| {
            parent.read_with(cx, |parent, _| {
                (
                    parent.resource_state().revision(),
                    parent.resource_state().committed().expect("source").clone(),
                )
            })
        });
        let originals = sources.each_ref().map(|(_, source)| source.image().clone());
        let (generation, result) = ready.mix.read_with(cx, |mix, cx| {
            (
                mix.generation(),
                mix.result(cx).expect("original result").clone(),
            )
        });
        let original_bytes = result.image().to_png();
        let original = graphics_core::export::prepare_cpu_mix_png(&result, original_bytes.len())
            .expect("existing original encoder oracle");
        let prepared = ready
            .mix
            .update(cx, |mix, cx| mix.prepare_png(original_bytes.len(), cx))
            .expect("fresh result starts preparation")
            .await
            .expect("current result prepared");
        assert_eq!(prepared.png().encoded_bytes(), original_bytes.as_slice());
        assert!(
            prepared
                .png()
                .encoded_bytes()
                .starts_with(b"\x89PNG\r\n\x1a\n")
        );
        assert_eq!(
            prepared.png().content_sha256(),
            original.png().content_sha256()
        );
        assert_eq!(prepared.png().width(), result.descriptor().width);
        assert_eq!(prepared.png().height(), result.descriptor().height);
        assert_eq!(prepared.png().sample_format(), CpuSampleFormat::Rgba8);
        assert_eq!(prepared.png().color_space(), ImageColorSpace::Srgb);
        assert_eq!(prepared.png().alpha(), AlphaAssociation::Straight);
        assert_eq!(prepared.png().content_type(), "image/png");
        assert_eq!(prepared.parameters(), result.parameters());
        assert_eq!(prepared.pixels_sha256(), result.pixels_sha256());
        assert_eq!(
            prepared.source_operation(),
            "raster_nodes::blending_nodes::MixNode"
        );
        assert_eq!(
            prepared.source_revision(),
            "28ed22824fae8828756f6f4163db62f3585f21e0"
        );
        for (index, (revision, source)) in sources.iter().enumerate() {
            assert_eq!(prepared.parents()[index].identity(), source.identity());
            assert_eq!(prepared.parents()[index].origin(), source.origin());
            assert_eq!(prepared.parents()[index].descriptor(), source.descriptor());
            ready.parents[index].read_with(cx, |parent, _| {
                assert_eq!(parent.resource_state().revision(), *revision);
                assert!(Arc::ptr_eq(
                    parent.resource_state().committed().expect("source"),
                    source
                ));
                assert_eq!(source.image(), &originals[index]);
            });
        }
        for cap in [0, original_bytes.len() - 1] {
            let error = ready
                .mix
                .update(cx, |mix, cx| mix.prepare_png(cap, cx))
                .expect("fresh result starts bounded encoding")
                .await
                .expect_err("insufficient result-size cap rejects PNG");
            assert!(error.contains("exceeding"));
            ready.mix.read_with(cx, |mix, cx| {
                assert_eq!(mix.generation(), generation);
                assert!(mix.result_is_fresh(cx));
                assert_eq!(mix.diagnostic(), None);
                assert!(Arc::ptr_eq(mix.result(cx).expect("same result"), &result));
            });
        }
        let repeated = ready
            .mix
            .update(cx, |mix, cx| mix.prepare_png(original_bytes.len(), cx))
            .expect("exact cap recovery")
            .await
            .expect("same immutable result repeats");
        assert_eq!(
            repeated.png().encoded_bytes(),
            prepared.png().encoded_bytes()
        );
        assert_eq!(
            repeated.png().content_sha256(),
            prepared.png().content_sha256()
        );
        assert_eq!(repeated.parameters(), prepared.parameters());
        assert_eq!(repeated.pixels_sha256(), prepared.pixels_sha256());
        for (index, parent) in repeated.parents().iter().enumerate() {
            assert_eq!(parent.identity(), prepared.parents()[index].identity());
            assert_eq!(parent.origin(), prepared.parents()[index].origin());
            assert_eq!(parent.descriptor(), prepared.parents()[index].descriptor());
            ready.parents[index].read_with(cx, |source, _| {
                assert_eq!(source.resource_state().revision(), sources[index].0);
                assert!(source.resource_state().is_fresh());
                let committed = source
                    .resource_state()
                    .committed()
                    .expect("immutable source");
                assert!(Arc::ptr_eq(committed, &sources[index].1));
                assert_eq!(committed.image(), &originals[index]);
            });
        }
    }

    #[gpui::test]
    async fn at_graphite_mix_session_png_rejects_each_same_revision_pending_parent_and_revocation(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_mix(cx);
        for (index, parent) in ready.parents.iter().enumerate() {
            let task = ready
                .mix
                .update(cx, |mix, cx| mix.prepare_png(4096, cx))
                .expect("fresh pre-request preparation");
            let revision = parent.read_with(cx, |parent, _| parent.resource_state().revision());
            parent
                .update(cx, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &ready.fixtures[index],
                            &ready.pngs[index],
                            &format!("test-supplied-PNG-parent-{index}-v2"),
                        ),
                        cx,
                    )
                })
                .expect("new verified dependency request");
            parent.read_with(cx, |parent, _| {
                assert_eq!(parent.resource_state().revision(), revision);
                assert!(parent.resource_state().has_active_request());
                assert!(!parent.resource_state().is_fresh());
            });
            assert!(
                ready
                    .mix
                    .update(cx, |mix, cx| mix.prepare_png(4096, cx))
                    .is_err()
            );
            assert!(task.await.is_err());
            cx.run_until_parked();
            let identity = parent.read_with(cx, |parent, _| {
                parent
                    .resource_state()
                    .committed()
                    .expect("refreshed parent")
                    .identity()
                    .clone()
            });
            assert!(ready.mix.read_with(cx, |mix, cx| mix.result_is_fresh(cx)));
            let task = ready
                .mix
                .update(cx, |mix, cx| mix.prepare_png(4096, cx))
                .expect("fresh pre-revocation preparation");
            parent
                .update(cx, |parent, cx| parent.revoke(&identity, cx))
                .expect("revoke actual parent");
            assert!(task.await.is_err());
            ready.mix.read_with(cx, |mix, cx| {
                assert!(mix.is_revoked(cx));
                assert!(mix.result(cx).is_none());
                assert!(!mix.result_is_fresh(cx));
            });
            assert!(
                ready
                    .mix
                    .update(cx, |mix, cx| mix.prepare_png(4096, cx))
                    .is_err()
            );
            parent
                .update(cx, |parent, cx| {
                    parent.load_encoded_png(
                        load(
                            &ready.fixtures[index],
                            &ready.pngs[index],
                            &format!("test-supplied-PNG-parent-{index}-v2"),
                        ),
                        cx,
                    )
                })
                .expect("explicit dependency recovery");
            cx.run_until_parked();
            let recovered = ready
                .mix
                .update(cx, |mix, cx| mix.prepare_png(4096, cx))
                .expect("fresh recovery starts")
                .await
                .expect("fresh recovery prepares");
            assert_eq!(recovered.parents()[index].identity(), &identity);
        }
    }

    #[gpui::test]
    async fn at_graphite_mix_session_png_rejects_parameter_bits_supersession_and_cancellation(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_mix(cx);
        ready.mix.update(cx, |mix, cx| {
            mix.set_parameters(
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: 0.,
                },
                cx,
            );
        });
        cx.run_until_parked();
        let task = ready
            .mix
            .update(cx, |mix, cx| mix.prepare_png(4096, cx))
            .expect("positive zero source result");
        ready.mix.update(cx, |mix, cx| {
            mix.set_parameters(
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: -0.,
                },
                cx,
            );
        });
        assert!(task.await.is_err());
        cx.run_until_parked();
        let prepared = ready
            .mix
            .update(cx, |mix, cx| mix.prepare_png(4096, cx))
            .expect("new bitwise parameter result")
            .await
            .expect("new result prepares");
        assert_eq!(prepared.parameters().opacity.to_bits(), (-0_f64).to_bits());
        let task = ready
            .mix
            .update(cx, |mix, cx| mix.prepare_png(4096, cx))
            .expect("fresh pre-cancellation preparation");
        ready.mix.update(cx, |mix, cx| mix.cancel(cx));
        assert!(task.await.is_err());
        assert!(
            ready
                .mix
                .update(cx, |mix, cx| mix.prepare_png(4096, cx))
                .is_err()
        );
        ready.mix.read_with(cx, |mix, _| {
            assert_eq!(mix.diagnostic(), Some("Mix evaluation cancelled"));
            assert!(!mix.is_pending());
        });
        ready.mix.update(cx, |mix, cx| mix.refresh(cx));
        cx.run_until_parked();
        let recovered = ready
            .mix
            .update(cx, |mix, cx| mix.prepare_png(4096, cx))
            .expect("explicit retry current result")
            .await
            .expect("retry prepares");
        assert_eq!(
            recovered.png().encoded_bytes(),
            prepared.png().encoded_bytes()
        );
        assert_eq!(recovered.parameters(), prepared.parameters());
    }

    #[gpui::test]
    async fn at_graphite_mix_session_png_rejects_released_session_before_returning_asset(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_mix(cx);
        let weak = ready.mix.downgrade();
        let task = ready
            .mix
            .update(cx, |mix, cx| mix.prepare_png(4096, cx))
            .expect("fresh preparation");
        drop(ready.mix);
        cx.update(|_| ());
        assert!(weak.upgrade().is_none());
        assert!(task.await.is_err());
        for parent in &ready.parents {
            assert!(parent.read_with(cx, |parent, _| parent.resource_state().is_fresh()));
        }
    }
}

mod mix_blend_menu_consumer_tests {
    use super::*;
    use crate::graph::GraphiteMixItem;
    use crate::resources::tests::{host_reader_fixture, indexed_request_png};
    use crate::session::{GraphiteEncodedPngLoad, GraphiteImageSession, GraphiteMixSession};
    use gpui::{Entity, KeyBinding, VisualTestContext};
    use graphics_core::resources::{
        CpuBlendMode, CpuMixBudget, CpuMixParameters, EncodedPngBudget,
    };
    use std::sync::Arc;

    async fn workspace_item(
        cx: &mut TestAppContext,
        blend_mode: CpuBlendMode,
    ) -> (
        Entity<GraphiteMixSession>,
        Entity<GraphiteMixItem>,
        &mut VisualTestContext,
    ) {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
            for (key, action) in [
                ("home", "menu::SelectFirst"),
                ("down", "menu::SelectNext"),
                ("enter", "menu::Confirm"),
                ("escape", "menu::Cancel"),
            ] {
                let binding = KeyBinding::load(
                    key,
                    cx.build_action(action, None)
                        .expect("actual registered menu action"),
                    Some(
                        gpui::KeyBindingContextPredicate::parse("menu")
                            .expect("menu context")
                            .into(),
                    ),
                    false,
                    None,
                    &gpui::DummyKeyboardMapper,
                )
                .expect("native test key binding");
                cx.bind_keys([binding]);
            }
        });
        let fixtures = [
            host_reader_fixture().expect("Over service"),
            host_reader_fixture().expect("Under service"),
        ];
        let parents = [
            cx.new(|_| GraphiteImageSession::new(4096)),
            cx.new(|_| GraphiteImageSession::new(4096)),
        ];
        for (index, parent) in parents.iter().enumerate() {
            let png = indexed_request_png(&fixtures[index], 4).expect("actual indexed PNG");
            let mut identity = png.image_identity;
            identity.version_id = format!("test-supplied-menu-parent-{index}").into();
            parent
                .update(cx, |parent, cx| {
                    parent.load_encoded_png(
                        GraphiteEncodedPngLoad {
                            assets: fixtures[index].assets.clone(),
                            asset_identity: png.asset_identity,
                            authorization: fixtures[index].input_authorization.clone(),
                            identity,
                            transform: [1., 0., 0., 1., 0., 0.],
                            encoded_budget: EncodedPngBudget {
                                max_encoded_bytes: 16 * 1024,
                                max_pixel_bytes: 4096,
                                max_metadata_bytes: 4096,
                            },
                        },
                        cx,
                    )
                })
                .expect("actual parent request");
        }
        cx.run_until_parked();
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                parents,
                CpuMixParameters {
                    blend_mode,
                    opacity: 37.5,
                },
                CpuMixBudget {
                    max_parent_bytes: 4096,
                    max_result_bytes: 4096,
                },
                cx,
            )
        });
        cx.run_until_parked();
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (host, window) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = host.read_with(window, |host, _| host.workspace().clone());
        let item = workspace.update_in(window, |workspace, window, cx| {
            let item = cx.new(|cx| GraphiteMixItem::from_session(mix.clone(), 4096, window, cx));
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
            window.focus(&item.focus_handle(cx), cx);
            item
        });
        window.simulate_resize(gpui::size(gpui::px(900.), gpui::px(1000.)));
        window.run_until_parked();
        (mix, item, window)
    }

    fn open_menu(window: &mut VisualTestContext) {
        let trigger = window
            .debug_bounds("GRAPHITE-MIX-BLEND-MODE")
            .expect("native dropdown trigger");
        window.simulate_click(trigger.center(), gpui::Modifiers::default());
        window.run_until_parked();
        for _ in 0..2 {
            window.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
                window.simulate_next_frame(cx);
            });
            window.run_until_parked();
        }
        assert!(window.debug_bounds("MENU_ITEM-Normal").is_some());
    }

    fn assert_original_selection(
        window: &mut VisualTestContext,
        mix: &Entity<GraphiteMixSession>,
        item: &Entity<GraphiteMixItem>,
        mode: CpuBlendMode,
    ) {
        mix.read_with(window, |mix, cx| {
            assert_eq!(mix.parameters().blend_mode, mode);
            assert_eq!(mix.parameters().opacity.to_bits(), 37.5_f64.to_bits());
            assert!(mix.result_is_fresh(cx));
            let sources = mix.parents().each_ref().map(|parent| {
                parent
                    .read(cx)
                    .resource_state()
                    .committed()
                    .expect("source")
                    .clone()
            });
            let before = sources.each_ref().map(|source| source.image().clone());
            let original = graphics_core::resources::evaluate_cpu_image_mix(
                &sources[0],
                &sources[1],
                mix.parameters(),
                CpuMixBudget {
                    max_parent_bytes: 4096,
                    max_result_bytes: 4096,
                },
            )
            .expect("original checked node");
            let result = mix.result(cx).expect("selected original result");
            assert_eq!(result.image(), original.image());
            assert_eq!(result.attributes(), original.attributes());
            assert_eq!(result.pixels_sha256(), original.pixels_sha256());
            assert_eq!(result.parameters().blend_mode, mode);
            let (mut bgra, _, _) = original.image().to_flat_u8();
            for pixel in bgra.as_chunks_mut::<4>().0.iter_mut() {
                pixel.swap(0, 2);
            }
            let presentation = item.read(cx).presentation().read(cx);
            assert!(presentation.is_fresh());
            assert_eq!(
                presentation
                    .image()
                    .expect("visible selected pixels")
                    .as_bytes(0),
                Some(bgra.as_slice())
            );
            for (index, source) in sources.iter().enumerate() {
                assert_eq!(result.parents()[index].identity(), source.identity());
                assert_eq!(source.image(), &before[index]);
            }
        });
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
    }

    #[gpui::test]
    async fn at_graphite_mix_item_all_source_svg_modes_are_mouse_selectable_with_original_results(
        cx: &mut TestAppContext,
    ) {
        let (mix, item, window) = workspace_item(cx, CpuBlendMode::Normal).await;
        let options = [
            (CpuBlendMode::Normal, "MENU_ITEM-Normal"),
            (CpuBlendMode::Darken, "MENU_ITEM-Darken"),
            (CpuBlendMode::Multiply, "MENU_ITEM-Multiply"),
            (CpuBlendMode::ColorBurn, "MENU_ITEM-Color Burn"),
            (CpuBlendMode::Lighten, "MENU_ITEM-Lighten"),
            (CpuBlendMode::Screen, "MENU_ITEM-Screen"),
            (CpuBlendMode::ColorDodge, "MENU_ITEM-Color Dodge"),
            (CpuBlendMode::Overlay, "MENU_ITEM-Overlay"),
            (CpuBlendMode::SoftLight, "MENU_ITEM-Soft Light"),
            (CpuBlendMode::HardLight, "MENU_ITEM-Hard Light"),
            (CpuBlendMode::Difference, "MENU_ITEM-Difference"),
            (CpuBlendMode::Exclusion, "MENU_ITEM-Exclusion"),
            (CpuBlendMode::Hue, "MENU_ITEM-Hue"),
            (CpuBlendMode::Saturation, "MENU_ITEM-Saturation"),
            (CpuBlendMode::Color, "MENU_ITEM-Color"),
            (CpuBlendMode::Luminosity, "MENU_ITEM-Luminosity"),
        ];
        let declared = CpuBlendMode::list_svg_subset()
            .into_iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            declared,
            options.iter().map(|(mode, _)| *mode).collect::<Vec<_>>()
        );
        assert_eq!(declared.len(), 16);
        for (mode, selector) in options {
            open_menu(window);
            assert!(window.debug_bounds("MENU_ITEM-Erase").is_none());
            let entry = window
                .debug_bounds(selector)
                .expect("original mode rendered in native menu");
            window.simulate_click(entry.center(), gpui::Modifiers::default());
            window.run_until_parked();
            assert!(window.debug_bounds("MENU_ITEM-Normal").is_none());
            assert_original_selection(window, &mix, &item, mode);
            assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
            assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        }
    }

    #[gpui::test]
    async fn at_graphite_mix_item_keyboard_menu_preserves_outside_mode_and_invalid_local_opacity(
        cx: &mut TestAppContext,
    ) {
        let (mix, item, window) = workspace_item(cx, CpuBlendMode::Erase).await;
        window.update(|_, cx| {
            let confirm = cx
                .build_action("menu::Confirm", None)
                .expect("existing native confirm action");
            cx.bind_keys([gpui::KeyBinding::load(
                "enter",
                confirm,
                None,
                false,
                None,
                &gpui::DummyKeyboardMapper,
            )
            .expect("native opacity Enter binding")]);
        });
        assert!(CpuBlendMode::Erase.index_in_list_svg_subset().is_none());
        assert_original_selection(window, &mix, &item, CpuBlendMode::Erase);
        let (generation, result, image_id) = mix.read_with(window, |mix, cx| {
            (
                mix.generation(),
                mix.result(cx).expect("stored outside mode").clone(),
                item.read(cx)
                    .presentation()
                    .read(cx)
                    .image()
                    .expect("outside mode pixels")
                    .id,
            )
        });
        open_menu(window);
        assert!(window.debug_bounds("MENU_ITEM-Erase").is_none());
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert!(window.debug_bounds("MENU_ITEM-Normal").is_none());
        mix.read_with(window, |mix, cx| {
            assert_eq!(mix.generation(), generation);
            assert_eq!(mix.parameters().blend_mode, CpuBlendMode::Erase);
            assert!(Arc::ptr_eq(
                mix.result(cx).expect("outside mode retained"),
                &result
            ));
        });
        let editor = item.read_with(window, |item, _| item.opacity_editor.clone());
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("invalid opacity", window, cx);
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.run_until_parked();
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        mix.read_with(window, |mix, cx| {
            assert_eq!(mix.generation(), generation);
            assert_eq!(mix.parameters().blend_mode, CpuBlendMode::Erase);
            assert_eq!(mix.parameters().opacity.to_bits(), 37.5_f64.to_bits());
            assert!(Arc::ptr_eq(
                mix.result(cx).expect("malformed draft retains result"),
                &result
            ));
        });
        item.read_with(window, |item, cx| {
            let presentation = item.presentation().read(cx);
            assert!(presentation.is_fresh());
            assert_eq!(presentation.image().expect("stable pixels").id, image_id);
        });
        assert!(window.debug_bounds("GRAPHITE-MIX-LAST-VALID").is_some());
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        open_menu(window);
        window.simulate_keystrokes("home down down enter");
        window.run_until_parked();
        assert!(window.debug_bounds("MENU_ITEM-Normal").is_none());
        assert_original_selection(window, &mix, &item, CpuBlendMode::Multiply);
        assert_eq!(
            editor.read_with(window, |editor, cx| editor.text(cx)),
            "invalid opacity"
        );
        assert!(window.debug_bounds("GRAPHITE-MIX-LAST-VALID").is_some());
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.simulate_input("37.5");
        window.run_until_parked();
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_original_selection(window, &mix, &item, CpuBlendMode::Multiply);
        assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
    }
}
mod mix_number_commit_consumer_tests {
    use super::*;
    use crate::graph::GraphiteMixItem;
    use crate::resources::tests::{host_reader_fixture, indexed_request_png};
    use crate::session::{GraphiteEncodedPngLoad, GraphiteImageSession, GraphiteMixSession};
    use gpui::{AnyWindowHandle, Entity, VisualTestContext};
    use graphics_core::resources::{
        CpuBlendMode, CpuMixBudget, CpuMixParameters, CpuMixResult, EncodedPngBudget,
    };
    use std::sync::Arc;
    use workspace::item::Item as _;

    struct NativeMix {
        mix: Entity<GraphiteMixSession>,
        item: Entity<GraphiteMixItem>,
        workspace: Entity<workspace::Workspace>,
        project: Entity<Project>,
        window: AnyWindowHandle,
    }

    async fn native_mix(cx: &mut TestAppContext) -> NativeMix {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
        });
        let fixture = host_reader_fixture().expect("actual service");
        let png = indexed_request_png(&fixture, 4).expect("actual producer PNG");
        let mut identity = png.image_identity;
        identity.version_id = "test-supplied-number-commit-parent".into();
        let source = cx.new(|_| GraphiteImageSession::new(4096));
        source
            .update(cx, |source, cx| {
                source.load_encoded_png(
                    GraphiteEncodedPngLoad {
                        assets: fixture.assets.clone(),
                        asset_identity: png.asset_identity,
                        authorization: fixture.input_authorization.clone(),
                        identity,
                        transform: [1., 0., 0., 1., 0., 0.],
                        encoded_budget: EncodedPngBudget {
                            max_encoded_bytes: 16 * 1024,
                            max_pixel_bytes: 4096,
                            max_metadata_bytes: 4096,
                        },
                    },
                    cx,
                )
            })
            .expect("real verified request");
        cx.run_until_parked();
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                [source.clone(), source],
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: 37.5,
                },
                CpuMixBudget {
                    max_parent_bytes: 4096,
                    max_result_bytes: 4096,
                },
                cx,
            )
        });
        cx.run_until_parked();
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (host, window) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = host.read_with(window, |host, _| host.workspace().clone());
        let item = workspace.update_in(window, |workspace, window, cx| {
            let item = cx.new(|cx| GraphiteMixItem::from_session(mix.clone(), 4096, window, cx));
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
            window.focus(&item.focus_handle(cx), cx);
            item
        });
        window.update(|_, cx| {
            let bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/default-linux.json",
                cx,
            )
            .expect("actual application default Linux keymap");
            cx.bind_keys(bindings);
        });
        window.run_until_parked();
        NativeMix {
            mix,
            item,
            workspace,
            project,
            window: window.update(|window, _| window.window_handle()),
        }
    }

    fn type_draft(
        window: &mut VisualTestContext,
        item: &Entity<GraphiteMixItem>,
        expression: &str,
    ) {
        let editor = item.read_with(window, |item, _| item.opacity_editor.clone());
        editor.update_in(window, |editor, window, cx| {
            window.activate_window();
            editor.set_text("", window, cx);
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.simulate_input(expression);
        window.run_until_parked();
        window.update(|window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        });
        window.run_until_parked();
        assert_eq!(
            editor.read_with(window, |editor, cx| editor.text(cx)),
            expression
        );
    }

    fn snapshot(
        window: &VisualTestContext,
        mix: &Entity<GraphiteMixSession>,
    ) -> (u64, Arc<CpuMixResult>) {
        mix.read_with(window, |mix, cx| {
            (
                mix.generation(),
                mix.result(cx).expect("committed original result").clone(),
            )
        })
    }

    fn assert_unchanged(
        window: &VisualTestContext,
        mix: &Entity<GraphiteMixSession>,
        before: &(u64, Arc<CpuMixResult>),
    ) {
        mix.read_with(window, |mix, cx| {
            assert_eq!(mix.generation(), before.0);
            assert_eq!(mix.parameters(), before.1.parameters());
            assert!(mix.result_is_fresh(cx));
            assert!(Arc::ptr_eq(mix.result(cx).expect("same result"), &before.1));
        });
    }

    fn blur(window: &mut VisualTestContext, item: &Entity<GraphiteMixItem>) {
        window.update(|window, cx| window.focus(&item.read(cx).focus_handle(cx), cx));
        window.run_until_parked();
    }

    #[gpui::test]
    async fn at_graphite_mix_number_draft_is_local_enter_commits_expression_once_and_blur_is_noop(
        cx: &mut TestAppContext,
    ) {
        let native = native_mix(cx).await;
        let mut window = VisualTestContext::from_window(native.window, cx);
        let before = snapshot(&window, &native.mix);
        type_draft(&mut window, &native.item, "sqrt(16) * 12.5");
        assert_unchanged(&window, &native.mix, &before);
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        native.mix.read_with(&window, |mix, cx| {
            assert_eq!(mix.parameters().opacity.to_bits(), 50_f64.to_bits());
            assert_eq!(mix.generation(), before.0 + 1);
            assert!(mix.result_is_fresh(cx));
            assert!(!Arc::ptr_eq(
                mix.result(cx).expect("new expression result"),
                &before.1
            ));
        });
        assert_eq!(
            native
                .item
                .read_with(&window, |item, cx| item.opacity_editor.read(cx).text(cx)),
            "50"
        );
        assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        let committed = snapshot(&window, &native.mix);
        blur(&mut window, &native.item);
        assert_unchanged(&window, &native.mix, &committed);
        type_draft(&mut window, &native.item, "100 / 2");
        assert_unchanged(&window, &native.mix, &committed);
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_unchanged(&window, &native.mix, &committed);
        assert_eq!(
            native
                .item
                .read_with(&window, |item, cx| item.opacity_editor.read(cx).text(cx)),
            "50"
        );
    }

    #[gpui::test]
    async fn at_graphite_mix_number_invalid_commit_clamps_recovery_and_native_reset_preserve_state(
        cx: &mut TestAppContext,
    ) {
        let native = native_mix(cx).await;
        let mut window = VisualTestContext::from_window(native.window, cx);
        let before = snapshot(&window, &native.mix);
        for expression in ["50%", "sqrt(-1)", "0/0", "2@"] {
            type_draft(&mut window, &native.item, expression);
            assert_unchanged(&window, &native.mix, &before);
            window.simulate_keystrokes("enter");
            window.run_until_parked();
            assert_unchanged(&window, &native.mix, &before);
            assert!(window.debug_bounds("GRAPHITE-MIX-LAST-VALID").is_some());
            assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        }
        for (expression, expected) in [("1/0", 100_f64), ("-25", 0.), ("1/4", 0.25)] {
            let previous = snapshot(&window, &native.mix);
            type_draft(&mut window, &native.item, expression);
            assert_unchanged(&window, &native.mix, &previous);
            blur(&mut window, &native.item);
            native.mix.read_with(&window, |mix, cx| {
                assert_eq!(mix.parameters().opacity.to_bits(), expected.to_bits());
                assert_eq!(mix.generation(), previous.0 + 1);
                assert!(mix.result_is_fresh(cx));
            });
            assert_eq!(
                native
                    .item
                    .read_with(&window, |item, cx| item.opacity_editor.read(cx).text(cx)),
                expected.to_string()
            );
            assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        }
        let valid = snapshot(&window, &native.mix);
        type_draft(&mut window, &native.item, "1 / 4");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_unchanged(&window, &native.mix, &valid);
        let reset = window
            .debug_bounds("GRAPHITE-MIX-RESET-OPACITY")
            .expect("native reset control");
        window.simulate_click(reset.center(), gpui::Modifiers::default());
        window.run_until_parked();
        native.mix.read_with(&window, |mix, cx| {
            assert_eq!(mix.parameters().opacity.to_bits(), 100_f64.to_bits());
            assert_eq!(mix.generation(), valid.0 + 1);
            assert!(mix.result_is_fresh(cx));
        });
        assert_eq!(
            native
                .item
                .read_with(&window, |item, cx| item.opacity_editor.read(cx).text(cx)),
            "100"
        );
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
    }

    #[gpui::test]
    async fn at_graphite_mix_number_split_focused_drafts_and_errors_survive_external_parameters_locally(
        cx: &mut TestAppContext,
    ) {
        let native = native_mix(cx).await;
        let mut first_window = VisualTestContext::from_window(native.window, cx);
        let app_state = native
            .workspace
            .read_with(&first_window, |workspace, _| workspace.app_state().clone());
        let project = native.project.clone();
        let second_root = cx.add_window(|window, cx| {
            let workspace =
                cx.new(|cx| workspace::Workspace::new(None, project, app_state, window, cx));
            workspace::MultiWorkspace::new(workspace, window, cx)
        });
        let mut second_window = VisualTestContext::from_window(second_root.into(), cx);
        let host = second_root.root(cx).expect("second real native root");
        let second_workspace = host.read_with(&second_window, |host, _| host.workspace().clone());
        let split = native
            .item
            .update_in(&mut second_window, |item, window, cx| {
                item.clone_on_split(None, window, cx)
            })
            .await
            .expect("real Item split");
        second_workspace.update_in(&mut second_window, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx);
        });
        type_draft(&mut second_window, &split, "50%");
        assert!(
            second_window.update(|window, cx| split
                .read(cx)
                .opacity_editor
                .read(cx)
                .is_focused(window))
        );
        second_window.simulate_keystrokes("enter");
        second_window.run_until_parked();
        assert!(
            !second_window.update(|window, cx| split
                .read(cx)
                .opacity_editor
                .read(cx)
                .is_focused(window))
        );
        assert!(second_window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        assert!(first_window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        type_draft(&mut first_window, &native.item, "sqrt(81)");
        native.mix.update(&mut first_window, |mix, cx| {
            mix.set_parameters(
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Screen,
                    opacity: 80.,
                },
                cx,
            )
        });
        first_window.run_until_parked();
        assert_eq!(
            native.item.read_with(&first_window, |item, cx| item
                .opacity_editor
                .read(cx)
                .text(cx)),
            "sqrt(81)"
        );
        assert_eq!(
            split.read_with(&second_window, |item, cx| item
                .opacity_editor
                .read(cx)
                .text(cx)),
            "50%"
        );
        let external = snapshot(&first_window, &native.mix);
        assert_eq!(external.1.parameters().blend_mode, CpuBlendMode::Screen);
        assert_eq!(external.1.parameters().opacity.to_bits(), 80_f64.to_bits());
        assert!(
            second_window
                .debug_bounds("GRAPHITE-MIX-LAST-VALID")
                .is_some()
        );
        assert!(second_window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        assert!(first_window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        first_window.simulate_keystrokes("enter");
        first_window.run_until_parked();
        native.mix.read_with(&first_window, |mix, cx| {
            assert_eq!(mix.parameters().blend_mode, CpuBlendMode::Screen);
            assert_eq!(mix.parameters().opacity.to_bits(), 9_f64.to_bits());
            assert_eq!(mix.generation(), external.0 + 1);
            assert!(mix.result_is_fresh(cx));
        });
        assert_eq!(
            native.item.read_with(&first_window, |item, cx| item
                .opacity_editor
                .read(cx)
                .text(cx)),
            "9"
        );
        assert_eq!(
            split.read_with(&second_window, |item, cx| item
                .opacity_editor
                .read(cx)
                .text(cx)),
            "50%"
        );
        assert!(second_window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        let shared = native
            .item
            .read_with(&first_window, |item, _| item.presentation().clone());
        split.read_with(&second_window, |item, cx| {
            assert_eq!(item.session().entity_id(), native.mix.entity_id());
            assert_eq!(item.presentation().entity_id(), shared.entity_id());
            assert!(item.presentation().read(cx).is_fresh());
        });
        let before_recovery = snapshot(&second_window, &native.mix);
        type_draft(&mut second_window, &split, "5/2");
        assert_unchanged(&second_window, &native.mix, &before_recovery);
        second_window.simulate_keystrokes("enter");
        second_window.run_until_parked();
        native.mix.read_with(&second_window, |mix, cx| {
            assert_eq!(mix.parameters().opacity.to_bits(), 2.5_f64.to_bits());
            assert_eq!(mix.parameters().blend_mode, CpuBlendMode::Screen);
            assert!(mix.result_is_fresh(cx));
        });
        assert!(second_window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        assert!(first_window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
    }

    #[gpui::test]
    async fn at_graphite_mix_number_escape_restores_latest_committed_opacity_without_evaluation(
        cx: &mut TestAppContext,
    ) {
        let native = native_mix(cx).await;
        let mut window = VisualTestContext::from_window(native.window, cx);
        let editor = native
            .item
            .read_with(&window, |item, _| item.opacity_editor.clone());
        let before = snapshot(&window, &native.mix);
        type_draft(&mut window, &native.item, "sqrt(81)");
        assert_unchanged(&window, &native.mix, &before);
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert_unchanged(&window, &native.mix, &before);
        assert_eq!(
            editor.read_with(&window, |editor, cx| editor.text(cx)),
            "37.5"
        );
        assert!(!window.update(|window, cx| editor.read(cx).is_focused(window)));
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());

        type_draft(&mut window, &native.item, "50%");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_unchanged(&window, &native.mix, &before);
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        native.mix.update(&mut window, |mix, cx| {
            mix.set_parameters(
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Screen,
                    opacity: 80.,
                },
                cx,
            );
        });
        window.run_until_parked();
        let external = snapshot(&window, &native.mix);
        assert_eq!(external.1.parameters().blend_mode, CpuBlendMode::Screen);
        assert_eq!(external.1.parameters().opacity.to_bits(), 80_f64.to_bits());
        assert_eq!(
            editor.read_with(&window, |editor, cx| editor.text(cx)),
            "50%"
        );
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_some());
        editor.update_in(&mut window, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.run_until_parked();
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert_unchanged(&window, &native.mix, &external);
        assert_eq!(
            editor.read_with(&window, |editor, cx| editor.text(cx)),
            "80"
        );
        assert!(!window.update(|window, cx| editor.read(cx).is_focused(window)));
        assert!(window.debug_bounds("GRAPHITE-MIX-ERROR").is_none());
        assert!(window.debug_bounds("GRAPHITE-MIX-LAST-VALID").is_none());
        assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
        blur(&mut window, &native.item);
        assert_unchanged(&window, &native.mix, &external);
    }
}

mod mix_color_session_controls {
    use crate::session::{AddMode, GraphicsSession};
    use graphics_core::resources::{CpuBlendMode, CpuMixParameters};

    fn assert_original_result(session: &GraphicsSession) {
        let (over, under, parameters, output) =
            session.mix_color_values().expect("committed Color Mix");
        let expected = graphics_core::raster::evaluate_mix_color_gamma_srgb_channels(
            over,
            under,
            parameters.blend_mode,
            parameters.opacity,
        )
        .expect("original Color constructor");
        assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));
    }

    #[test]
    fn at_graphite_mix_color_session_preserves_over_under_order_alpha_and_original_result() {
        let mut session = GraphicsSession::new(2., 3.);
        assert_eq!(
            session.mix_color_parameters().blend_mode,
            CpuBlendMode::Normal
        );
        assert_eq!(
            session.mix_color_parameters().opacity.to_bits(),
            100_f64.to_bits()
        );
        for (over, under) in [
            ("1, 0, 0, 0", "0, 0, 1, 1"),
            ("1, 0, 0, 0.5", "0, 0, 1, 1"),
            ("1, 0, 0, 1", "0, 0, 1, 0.5"),
        ] {
            session.set_mode(AddMode::MixColor, over, under);
            assert_eq!(session.mode(), AddMode::MixColor);
            assert!(session.diagnostic().is_none());
            assert_original_result(&session);
            assert!(session.expression_text().starts_with("Mix Over RGBA"));
        }
        session.update_inputs("1, 0, 0, 0.5", "0, 0, 1, 1");
        let forward = session.mix_color_values().expect("forward");
        session.update_inputs("0, 0, 1, 1", "1, 0, 0, 0.5");
        let reversed = session.mix_color_values().expect("reversed");
        assert_eq!(reversed.0, forward.1);
        assert_eq!(reversed.1, forward.0);
        assert_ne!(reversed.3.map(f32::to_bits), forward.3.map(f32::to_bits));
        assert_original_result(&session);
    }

    #[test]
    fn at_graphite_mix_color_session_invalid_channels_retain_committed_tuple_and_recover() {
        let mut session = GraphicsSession::new(2., 3.);
        session.set_mode(
            AddMode::MixColor,
            "0.25, 0.5, 0.75, 0.5",
            "0.8, 0.2, 0.1, 1",
        );
        let committed = session.mix_color_values().expect("initial Color Mix");
        let revision = session.revision();
        let expression = session.expression_text();
        for (over, under) in [
            ("", "0.8, 0.2, 0.1, 1"),
            ("0.25, 0.5, 0.75", "0.8, 0.2, 0.1, 1"),
            ("0.25, 0.5, 0.75, 0.5, 1", "0.8, 0.2, 0.1, 1"),
            ("0.25, 0.5, 0.75, invalid", "0.8, 0.2, 0.1, 1"),
            ("0.25, 0.5, 0.75, 0.5", "0.8, 0.2, invalid, 1"),
        ] {
            session.update_inputs(over, under);
            assert!(session.diagnostic().is_some());
            assert_eq!(session.revision(), revision);
            assert_eq!(
                session.mix_color_values().expect("preserved tuple"),
                committed
            );
            assert_eq!(session.expression_text(), expression);
        }
        let draft_parameters = CpuMixParameters {
            blend_mode: CpuBlendMode::Screen,
            opacity: 25.,
        };
        session.set_mix_color_parameters(draft_parameters, "invalid", "0.8, 0.2, 0.1, 1");
        assert_eq!(session.mix_color_parameters(), draft_parameters);
        assert_eq!(
            session
                .mix_color_values()
                .expect("prior committed parameters")
                .2,
            committed.2
        );
        assert_eq!(session.revision(), revision);
        assert!(session.diagnostic().is_some());
        session.update_inputs("0.25, 0.5, 0.75, 0.5", "0.8, 0.2, 0.1, 1");
        assert_eq!(session.revision(), revision + 1);
        assert_eq!(
            session.mix_color_values().expect("recovered").2,
            draft_parameters
        );
        assert!(session.diagnostic().is_none());
        assert_original_result(&session);
    }

    #[test]
    fn at_graphite_mix_color_session_unchanged_updates_preserve_revision_and_input_bits() {
        let mut session = GraphicsSession::new(2., 3.);
        session.set_mode(AddMode::MixColor, "0, 0.25, 0.5, 0.5", "0.75, 0.5, 0.25, 1");
        let revision = session.revision();
        session.update_inputs("0.0, .25, .50, .5", ".75, .50, .25, 1.0");
        session.set_mix_color_parameters(
            session.mix_color_parameters(),
            "0, .25, .5, .5",
            ".75, .5, .25, 1",
        );
        assert_eq!(session.revision(), revision);
        session.update_inputs("-0, .25, .5, .5", ".75, .5, .25, 1");
        assert_eq!(session.revision(), revision + 1);
        assert_eq!(
            session.mix_color_values().expect("signed input").0[0].to_bits(),
            (-0_f32).to_bits()
        );
        let over = "-0, .25, .5, .5";
        let under = ".75, .5, .25, 1";
        session.set_mix_color_parameters(
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: 0.,
            },
            over,
            under,
        );
        let zero_revision = session.revision();
        session.set_mix_color_parameters(
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: -0.,
            },
            over,
            under,
        );
        assert_eq!(session.revision(), zero_revision + 1);
        assert_eq!(
            session
                .mix_color_values()
                .expect("signed parameter")
                .2
                .opacity
                .to_bits(),
            (-0_f64).to_bits()
        );
        let signed_revision = session.revision();
        session.set_mix_color_parameters(session.mix_color_parameters(), over, under);
        assert_eq!(session.revision(), signed_revision);
        assert_original_result(&session);
    }

    #[test]
    fn at_graphite_mix_color_session_keeps_original_opacity_clamp_and_mode_parameters() {
        let mut session = GraphicsSession::new(2., 3.);
        let over = ".25, .5, .75, .5";
        let under = ".8, .2, .1, 1";
        session.set_mode(AddMode::MixColor, over, under);
        for (opacity, effective) in [
            (-25., 0_f64),
            (125., 100.),
            (f64::NEG_INFINITY, 0.),
            (f64::INFINITY, 100.),
            (f64::NAN, 0.),
            (12.5, 12.5),
        ] {
            let parameters = CpuMixParameters {
                blend_mode: CpuBlendMode::Multiply,
                opacity,
            };
            session.set_mix_color_parameters(parameters, over, under);
            let committed = session
                .mix_color_values()
                .expect("original opacity behavior");
            assert_eq!(committed.2, parameters);
            assert_eq!(committed.2.opacity.to_bits(), opacity.to_bits());
            assert_eq!(
                parameters.effective_opacity_percent().to_bits(),
                effective.to_bits()
            );
            assert!(session.diagnostic().is_none());
            assert_original_result(&session);
            let revision = session.revision();
            session.set_mix_color_parameters(parameters, over, under);
            assert_eq!(session.revision(), revision);
        }
    }

    #[test]
    fn at_graphite_mix_color_session_isolation_and_cross_family_errors_preserve_last_valid() {
        let mut first = GraphicsSession::new(2., 3.);
        let mut second = GraphicsSession::new(7., 8.);
        first.set_mix_color_parameters(
            CpuMixParameters {
                blend_mode: CpuBlendMode::Screen,
                opacity: 50.,
            },
            "invalid",
            "invalid",
        );
        assert_eq!(first.revision(), 0);
        assert_eq!(first.result(), 5.);
        assert!(first.diagnostic().is_none());
        first.set_mode(AddMode::MixColor, "invalid", "0, 0, 1, 1");
        assert_eq!(first.result(), 5.);
        assert!(first.expression_text().starts_with("Last valid result:"));
        assert!(first.diagnostic().is_some());
        first.update_inputs("1, 0, 0, .5", "0, 0, 1, 1");
        let committed = first.mix_color_values().expect("first recovered");
        let revision = first.revision();
        assert_original_result(&first);
        assert_eq!(second.inputs(), (7., 8.));
        assert_eq!(second.result(), 15.);
        assert_eq!(second.revision(), 0);
        second.update_inputs("9", "10");
        assert_eq!(second.result(), 19.);
        assert_eq!(first.mix_color_values().expect("isolated"), committed);
        assert_eq!(first.revision(), revision);
        first.set_mode(AddMode::Scalar, "invalid", "3");
        assert_eq!(
            first.mix_color_values().expect("last Color tuple"),
            committed
        );
        assert_eq!(first.revision(), revision);
        assert!(first.expression_text().starts_with("Last valid result:"));
        assert!(first.diagnostic().is_some());
        first.update_inputs("4", "3");
        assert_eq!(first.inputs(), (4., 3.));
        assert_eq!(first.result(), 7.);
        assert!(first.diagnostic().is_none());
    }
}

mod mix_color_workspace_controls {
    use super::*;
    use crate::session::AddMode;
    use gpui::{Entity, VisualTestContext};
    use graphics_core::resources::{CpuBlendMode, CpuMixParameters};

    struct ColorSnapshot {
        revision: u64,
        over: [f32; 4],
        under: [f32; 4],
        parameters: CpuMixParameters,
        output: [f32; 4],
    }

    async fn native_item(
        cx: &mut TestAppContext,
    ) -> (Entity<GraphicsWorkspaceItem>, &mut VisualTestContext) {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
            crate::graph::init(cx);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (host, window) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        window.update(|_, cx| {
            let bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/default-linux.json",
                cx,
            )
            .expect("actual application default Linux keymap");
            cx.bind_keys(bindings);
        });
        window.dispatch_action(crate::graph::OpenGraphite);
        window.run_until_parked();
        let item = host.read_with(window, |host, cx| {
            host.workspace()
                .read(cx)
                .active_item_as::<GraphicsWorkspaceItem>(cx)
                .expect("registered native Workspace Item")
        });
        window.simulate_resize(gpui::size(gpui::px(1100.), gpui::px(1100.)));
        window.run_until_parked();
        (item, window)
    }

    fn click(window: &mut VisualTestContext, selector: &'static str) {
        let bounds = window.debug_bounds(selector).expect("native control");
        window.simulate_click(bounds.center(), gpui::Modifiers::default());
        window.run_until_parked();
    }

    fn type_text(window: &mut VisualTestContext, editor: &Entity<editor::Editor>, text: &str) {
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.simulate_input(text);
        window.run_until_parked();
        assert_eq!(editor.read_with(window, |editor, cx| editor.text(cx)), text);
    }

    fn ready_color(window: &mut VisualTestContext, item: &Entity<GraphicsWorkspaceItem>) {
        click(window, "GRAPHITE-MODE-MIX-COLOR");
        let (over, under) = item.read_with(window, |item, _| {
            (item.augend_editor.clone(), item.addend_editor.clone())
        });
        type_text(window, &over, "1, 0, 0, 0.5");
        type_text(window, &under, "0, 0, 1, 1");
        item.read_with(window, |item, cx| {
            assert_eq!(item.session.read(cx).mode(), AddMode::MixColor);
            assert!(item.session.read(cx).diagnostic().is_none());
        });
        assert_original(window, item);
    }

    fn snapshot(window: &VisualTestContext, item: &Entity<GraphicsWorkspaceItem>) -> ColorSnapshot {
        item.read_with(window, |item, cx| {
            let session = item.session.read(cx);
            let (over, under, parameters, output) =
                session.mix_color_values().expect("committed Color");
            ColorSnapshot {
                revision: session.revision(),
                over,
                under,
                parameters,
                output,
            }
        })
    }

    fn assert_preserved(
        window: &VisualTestContext,
        item: &Entity<GraphicsWorkspaceItem>,
        before: &ColorSnapshot,
    ) {
        let after = snapshot(window, item);
        assert_eq!(after.revision, before.revision);
        assert_eq!(after.over.map(f32::to_bits), before.over.map(f32::to_bits));
        assert_eq!(
            after.under.map(f32::to_bits),
            before.under.map(f32::to_bits)
        );
        assert_eq!(after.parameters, before.parameters);
        assert_eq!(
            after.output.map(f32::to_bits),
            before.output.map(f32::to_bits)
        );
    }

    fn assert_original(window: &mut VisualTestContext, item: &Entity<GraphicsWorkspaceItem>) {
        let current = snapshot(window, item);
        let expected = graphics_core::raster::evaluate_mix_color_gamma_srgb_channels(
            current.over,
            current.under,
            current.parameters.blend_mode,
            current.parameters.opacity,
        )
        .expect("original Color constructor");
        assert_eq!(current.output.map(f32::to_bits), expected.map(f32::to_bits));
        let text = item.read_with(window, |item, cx| item.session.read(cx).result_text());
        assert!(window.debug_bounds("GRAPHITE-RESULT").is_some());
        assert_eq!(text, format!("{expected:?}"));
    }

    #[gpui::test]
    async fn at_graphite_color_mix_workspace_selection_order_alpha_and_source_mode_menu(
        cx: &mut TestAppContext,
    ) {
        let (item, window) = native_item(cx).await;
        let prior_components = item.read_with(window, |item, cx| {
            (
                item.augend_y_editor.read(cx).text(cx),
                item.addend_y_editor.read(cx).text(cx),
            )
        });
        ready_color(window, &item);
        let first = snapshot(window, &item);
        assert_eq!(first.over, [1., 0., 0., 0.5]);
        assert_eq!(first.under, [0., 0., 1., 1.]);
        assert_eq!(first.parameters.blend_mode, CpuBlendMode::Normal);
        assert_eq!(first.parameters.opacity.to_bits(), 100_f64.to_bits());
        let (over, under, opacity) = item.read_with(window, |item, _| {
            (
                item.augend_editor.clone(),
                item.addend_editor.clone(),
                item.mix_opacity_editor.clone(),
            )
        });
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "100"
        );
        type_text(window, &over, "0, 0, 1, 1");
        type_text(window, &under, "1, 0, 0, 0.5");
        let reversed = snapshot(window, &item);
        assert_eq!(reversed.over, first.under);
        assert_eq!(reversed.under, first.over);
        assert_ne!(
            reversed.output.map(f32::to_bits),
            first.output.map(f32::to_bits)
        );
        assert_original(window, &item);
        type_text(window, &under, "1, 0, 0, 0");
        assert_original(window, &item);
        let before_menu = snapshot(window, &item);
        click(window, "GRAPHITE-COLOR-MIX-BLEND-MODE");
        for _ in 0..2 {
            window.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
                window.simulate_next_frame(cx);
            });
            window.run_until_parked();
        }
        click(window, "MENU_ITEM-Screen");
        let after_menu = snapshot(window, &item);
        assert_eq!(after_menu.parameters.blend_mode, CpuBlendMode::Screen);
        assert_eq!(
            after_menu.parameters.opacity.to_bits(),
            before_menu.parameters.opacity.to_bits()
        );
        assert_eq!(after_menu.over, before_menu.over);
        assert_eq!(after_menu.under, before_menu.under);
        assert_eq!(after_menu.revision, before_menu.revision + 1);
        assert_original(window, &item);
        assert_eq!(
            item.read_with(window, |item, cx| {
                (
                    item.augend_y_editor.read(cx).text(cx),
                    item.addend_y_editor.read(cx).text(cx),
                )
            }),
            prior_components
        );
    }

    #[gpui::test]
    async fn at_graphite_color_mix_workspace_opacity_expression_commit_noop_escape_and_recovery(
        cx: &mut TestAppContext,
    ) {
        let (item, window) = native_item(cx).await;
        ready_color(window, &item);
        let opacity = item.read_with(window, |item, _| item.mix_opacity_editor.clone());
        let before = snapshot(window, &item);
        type_text(window, &opacity, "sqrt(16) * 12.5");
        assert_preserved(window, &item, &before);
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let committed = snapshot(window, &item);
        assert_eq!(committed.parameters.opacity.to_bits(), 50_f64.to_bits());
        assert_eq!(committed.revision, before.revision + 1);
        assert_eq!(committed.over, before.over);
        assert_eq!(committed.under, before.under);
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "50"
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-COLOR-MIX-OPACITY-ERROR")
                .is_none()
        );
        assert_original(window, &item);
        type_text(window, &opacity, "100 / 2");
        assert_preserved(window, &item, &committed);
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_preserved(window, &item, &committed);
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "50"
        );
        type_text(window, &opacity, "50%");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_preserved(window, &item, &committed);
        assert!(
            window
                .debug_bounds("GRAPHITE-COLOR-MIX-OPACITY-ERROR")
                .is_some()
        );
        assert!(item.read_with(window, |item, cx| {
            item.session.read(cx).diagnostic().is_none()
        }));
        opacity.update_in(window, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx)
        });
        window.run_until_parked();
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert_preserved(window, &item, &committed);
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "50"
        );
        assert!(!window.update(|window, cx| opacity.read(cx).is_focused(window)));
        assert!(
            window
                .debug_bounds("GRAPHITE-COLOR-MIX-OPACITY-ERROR")
                .is_none()
        );
        type_text(window, &opacity, "1 / 0");
        assert_preserved(window, &item, &committed);
        let over = item.read_with(window, |item, _| item.augend_editor.clone());
        over.update_in(window, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx)
        });
        window.run_until_parked();
        let recovered = snapshot(window, &item);
        assert_eq!(recovered.parameters.opacity.to_bits(), 100_f64.to_bits());
        assert_eq!(recovered.revision, committed.revision + 1);
        assert_eq!(recovered.over, committed.over);
        assert_eq!(recovered.under, committed.under);
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "100"
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-COLOR-MIX-OPACITY-ERROR")
                .is_none()
        );
        assert_original(window, &item);
    }

    #[gpui::test]
    async fn at_graphite_color_mix_workspace_invalid_parent_recovery_and_auxiliary_fields_survive(
        cx: &mut TestAppContext,
    ) {
        let (item, window) = native_item(cx).await;
        let (first_component, second_component) = item.read_with(window, |item, _| {
            (item.augend_y_editor.clone(), item.addend_y_editor.clone())
        });
        first_component.update_in(window, |editor, window, cx| {
            editor.set_text("7", window, cx)
        });
        second_component.update_in(window, |editor, window, cx| {
            editor.set_text("9", window, cx)
        });
        window.run_until_parked();
        ready_color(window, &item);
        let (over, under, opacity) = item.read_with(window, |item, _| {
            (
                item.augend_editor.clone(),
                item.addend_editor.clone(),
                item.mix_opacity_editor.clone(),
            )
        });
        let committed = snapshot(window, &item);
        type_text(window, &under, "0, 0, invalid, 1");
        assert_preserved(window, &item, &committed);
        assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
        type_text(window, &opacity, "1 / 4");
        assert_preserved(window, &item, &committed);
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_preserved(window, &item, &committed);
        assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
        assert_eq!(
            item.read_with(window, |item, cx| item
                .session
                .read(cx)
                .mix_color_parameters()
                .opacity
                .to_bits()),
            0.25_f64.to_bits()
        );
        type_text(window, &under, "0, 0, 1, 1");
        let recovered = snapshot(window, &item);
        assert_eq!(recovered.parameters.opacity.to_bits(), 0.25_f64.to_bits());
        assert_eq!(recovered.revision, committed.revision + 1);
        assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_none());
        assert_original(window, &item);
        type_text(window, &opacity, "-25");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let clamped = snapshot(window, &item);
        assert_eq!(clamped.parameters.opacity.to_bits(), 0_f64.to_bits());
        assert_eq!(clamped.over, recovered.over);
        assert_eq!(clamped.under, recovered.under);
        assert_eq!(
            over.read_with(window, |editor, cx| editor.text(cx)),
            "1, 0, 0, 0.5"
        );
        assert_eq!(
            under.read_with(window, |editor, cx| editor.text(cx)),
            "0, 0, 1, 1"
        );
        assert_eq!(
            first_component.read_with(window, |editor, cx| editor.text(cx)),
            "7"
        );
        assert_eq!(
            second_component.read_with(window, |editor, cx| editor.text(cx)),
            "9"
        );
        assert_original(window, &item);
    }

    #[gpui::test]
    async fn at_graphite_color_mix_workspace_external_parameters_sync_only_clean_unfocused_drafts(
        cx: &mut TestAppContext,
    ) {
        let (item, window) = native_item(cx).await;
        ready_color(window, &item);
        let (session, opacity, under, first_component, second_component) =
            item.read_with(window, |item, _| {
                (
                    item.session.clone(),
                    item.mix_opacity_editor.clone(),
                    item.addend_editor.clone(),
                    item.augend_y_editor.clone(),
                    item.addend_y_editor.clone(),
                )
            });
        let auxiliary = (
            first_component.read_with(window, |editor, cx| editor.text(cx)),
            second_component.read_with(window, |editor, cx| editor.text(cx)),
        );
        let change_opacity = |window: &mut VisualTestContext, opacity| {
            session.update(window, |session, cx| {
                let mut parameters = session.mix_color_parameters();
                parameters.opacity = opacity;
                session.set_mix_color_parameters(parameters, "1, 0, 0, 0.5", "0, 0, 1, 1");
                cx.notify();
            });
            window.run_until_parked();
        };
        assert!(!window.update(|window, cx| opacity.read(cx).is_focused(window)));
        change_opacity(window, 25.);
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "25"
        );
        assert_original(window, &item);
        let visible = snapshot(window, &item);
        assert_eq!(visible.parameters.opacity.to_bits(), 25_f64.to_bits());

        click(window, "GRAPHITE-MODE-SCALAR");
        assert!(window.debug_bounds("GRAPHITE-COLOR-MIX-OPACITY").is_none());
        change_opacity(window, 75.);
        assert_preserved(window, &item, &visible);
        assert_eq!(
            session.read_with(window, |session, _| session
                .mix_color_parameters()
                .opacity
                .to_bits()),
            75_f64.to_bits()
        );
        click(window, "GRAPHITE-MODE-MIX-COLOR");
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "75"
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-COLOR-MIX-OPACITY-ERROR")
                .is_none()
        );
        assert_original(window, &item);
        let shown = snapshot(window, &item);
        assert_eq!(shown.parameters.opacity.to_bits(), 75_f64.to_bits());

        opacity.update_in(window, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.run_until_parked();
        change_opacity(window, 80.);
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "75"
        );
        let focused = snapshot(window, &item);
        assert_eq!(focused.parameters.opacity.to_bits(), 80_f64.to_bits());
        under.update_in(window, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.run_until_parked();
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "80"
        );
        assert_preserved(window, &item, &focused);

        type_text(window, &opacity, "sqrt(81)");
        assert_preserved(window, &item, &focused);
        change_opacity(window, 90.);
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "sqrt(81)"
        );
        let latest = snapshot(window, &item);
        assert_eq!(latest.parameters.opacity.to_bits(), 90_f64.to_bits());
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "90"
        );
        assert!(!window.update(|window, cx| opacity.read(cx).is_focused(window)));
        assert_preserved(window, &item, &latest);
        assert!(
            window
                .debug_bounds("GRAPHITE-COLOR-MIX-OPACITY-ERROR")
                .is_none()
        );

        type_text(window, &opacity, "50%");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert!(
            window
                .debug_bounds("GRAPHITE-COLOR-MIX-OPACITY-ERROR")
                .is_some()
        );
        change_opacity(window, 70.);
        let after_error = snapshot(window, &item);
        assert_eq!(after_error.parameters.opacity.to_bits(), 70_f64.to_bits());
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "50%"
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-COLOR-MIX-OPACITY-ERROR")
                .is_some()
        );
        opacity.update_in(window, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx);
        });
        window.run_until_parked();
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert_eq!(
            opacity.read_with(window, |editor, cx| editor.text(cx)),
            "70"
        );
        assert_preserved(window, &item, &after_error);
        assert!(
            window
                .debug_bounds("GRAPHITE-COLOR-MIX-OPACITY-ERROR")
                .is_none()
        );
        assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_none());
        assert_eq!(
            first_component.read_with(window, |editor, cx| editor.text(cx)),
            auxiliary.0
        );
        assert_eq!(
            second_component.read_with(window, |editor, cx| editor.text(cx)),
            auxiliary.1
        );
        assert_original(window, &item);
    }
}

mod asset_binding_registration_controls {
    use super::*;
    use crate::interop::{GraphiteAssetBindingError, GraphiteAssetImageRequest};
    use crate::resources::tests::{IndexedRequestPng, host_reader_fixture, indexed_request_png};

    fn request(png: &IndexedRequestPng) -> GraphiteAssetImageRequest {
        GraphiteAssetImageRequest {
            asset_identity: png.asset_identity.clone(),
            identity: png.image_identity.clone(),
            transform: [1., 0., 0., 1., 0., 0.],
            encoded_budget: graphics_core::resources::EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
            max_result_bytes: 4096,
        }
    }

    #[gpui::test]
    fn at_graphite_asset_binding_host_registration_reuses_exact_service_and_grant(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real indexed service");
        let png = indexed_request_png(&fixture, 4).expect("original PNG");
        let binding = cx
            .update(|cx| {
                crate::register_asset_binding(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    cx,
                )
            })
            .expect("host binding");
        let session = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("real reader");
        cx.run_until_parked();
        assert!(session.read_with(cx, |session, _| session.resource_state().is_fresh()));
        let revision = session.read_with(cx, |session, _| session.resource_state().revision());
        let repeated = cx
            .update(|cx| {
                crate::register_asset_binding(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    cx,
                )
            })
            .expect("same host binding");
        assert_eq!(binding.entity_id(), repeated.entity_id());
        assert_eq!(
            cx.update(|cx| crate::asset_binding(cx))
                .expect("registered binding")
                .entity_id(),
            binding.entity_id()
        );
        assert!(binding.read_with(cx, |binding, _| binding.is_active()));
        assert!(session.read_with(cx, |session, _| session.resource_state().is_fresh()));
        assert_eq!(
            session.read_with(cx, |session, _| session.resource_state().revision()),
            revision
        );
        cx.update(crate::clear_asset_binding)
            .expect("shutdown revokes input");
        assert!(!binding.read_with(cx, |binding, _| binding.is_active()));
        assert!(session.read_with(cx, |session, _| session.resource_state().is_revoked()));
        assert!(cx.update(|cx| crate::asset_binding(cx)).is_none());
        cx.update(crate::clear_asset_binding)
            .expect("idempotent shutdown");
    }

    #[gpui::test]
    fn at_graphite_asset_binding_host_same_profile_arc_replacement_revokes_old_owner(
        cx: &mut TestAppContext,
    ) {
        let first = host_reader_fixture().expect("first actual service");
        let second = host_reader_fixture().expect("second actual service");
        assert_eq!(
            first.input_authorization.profile_id(),
            second.input_authorization.profile_id()
        );
        assert!(!std::sync::Arc::ptr_eq(&first.assets, &second.assets));
        let first_png = indexed_request_png(&first, 4).expect("first encoded input");
        let second_png = indexed_request_png(&second, 3).expect("second encoded input");
        let old = cx
            .update(|cx| {
                crate::register_asset_binding(
                    first.assets.clone(),
                    first.input_authorization.clone(),
                    cx,
                )
            })
            .expect("first host binding");
        let old_session = old
            .update(cx, |binding, cx| {
                binding.open_image(request(&first_png), cx)
            })
            .expect("first session");
        cx.run_until_parked();
        assert!(old_session.read_with(cx, |session, _| session.resource_state().is_fresh()));
        let current = cx
            .update(|cx| {
                crate::register_asset_binding(
                    second.assets.clone(),
                    second.input_authorization.clone(),
                    cx,
                )
            })
            .expect("replacement binding");
        assert_ne!(old.entity_id(), current.entity_id());
        assert!(!old.read_with(cx, |binding, _| binding.is_active()));
        assert!(old_session.read_with(cx, |session, _| session.resource_state().is_revoked()));
        assert!(old_session.read_with(cx, |session, _| session.result().is_none()));
        let current_session = current
            .update(cx, |binding, cx| {
                binding.open_image(request(&second_png), cx)
            })
            .expect("replacement reader");
        cx.run_until_parked();
        assert!(current.read_with(cx, |binding, _| binding.is_active()));
        assert!(current_session.read_with(cx, |session, _| session.resource_state().is_fresh()));
        assert_eq!(
            current_session.read_with(cx, |session, _| session
                .resource_state()
                .committed()
                .expect("decoded replacement")
                .identity()
                .clone()),
            second_png.image_identity.clone()
        );
        assert!(old_session.read_with(cx, |session, _| session.resource_state().is_revoked()));
        cx.update(crate::clear_asset_binding)
            .expect("replace shutdown");
    }

    #[gpui::test]
    fn at_graphite_asset_binding_host_failed_replacement_closes_previous_binding(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("actual service");
        let png = indexed_request_png(&fixture, 4).expect("indexed PNG");
        let binding = cx
            .update(|cx| {
                crate::register_asset_binding(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    cx,
                )
            })
            .expect("initial binding");
        let session = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("pending real source");
        let foreign = sim_runtime::authorize_native_input_reader("other-profile")
            .expect("genuine other profile grant");
        let error = cx
            .update(|cx| crate::register_asset_binding(fixture.assets.clone(), foreign, cx))
            .expect_err("foreign profile cannot replace binding");
        assert_eq!(error, GraphiteAssetBindingError::ProfileMismatch);
        assert!(!binding.read_with(cx, |binding, _| binding.is_active()));
        assert!(cx.update(|cx| crate::asset_binding(cx)).is_none());
        cx.run_until_parked();
        assert!(session.read_with(cx, |session, _| {
            !session.resource_state().has_active_request()
                && session.resource_state().revision() == 0
                && session.resource_state().diagnostic()
                    == Some(&crate::resources::ImageResourceError::Revoked)
        }));
        assert!(session.read_with(cx, |session, _| session.result().is_none()));
        assert!(session.read_with(cx, |session, _| {
            session.resource_state().committed().is_none()
        }));
        cx.update(crate::clear_asset_binding)
            .expect("failed replacement shutdown");
    }

    #[gpui::test]
    fn at_graphite_asset_binding_host_output_grant_reuse_and_removal_revoke_previous_owner(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real asset service");
        let png = indexed_request_png(&fixture, 4).expect("actual indexed PNG");
        let policy = sim_runtime::PermissionPolicy::native_runtime_services(
            fixture.input_authorization.profile_id(),
        )
        .expect("existing native policy");
        let output = policy
            .authorize_one(
                sim_runtime::NATIVE_API_ASSET_READER_SUBJECT,
                sim_runtime::Capability::Asset {
                    namespace: "output".to_owned(),
                    action: sim_runtime::AssetOperation::Read,
                },
            )
            .expect("only Output read is projected");
        let binding = cx
            .update(|cx| {
                crate::register_asset_binding_with_output_reader(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    Some(output.clone()),
                    cx,
                )
            })
            .expect("matching host grants");
        let repeated = cx
            .update(|cx| {
                crate::register_asset_binding_with_output_reader(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    Some(output),
                    cx,
                )
            })
            .expect("exact reuse");
        assert_eq!(binding.entity_id(), repeated.entity_id());
        let session = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("existing Input request remains admitted");
        cx.run_until_parked();
        assert!(session.read_with(cx, |session, _| session.resource_state().is_fresh()));
        let replacement = cx
            .update(|cx| {
                crate::register_asset_binding(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    cx,
                )
            })
            .expect("host drops Output grant");
        assert_ne!(binding.entity_id(), replacement.entity_id());
        assert!(!binding.read_with(cx, |binding, _| binding.is_active()));
        assert!(session.read_with(cx, |session, _| session.resource_state().is_revoked()));
        assert!(replacement.read_with(cx, |binding, _| {
            binding.is_current(&fixture.assets, &fixture.input_authorization)
        }));
        cx.update(crate::clear_asset_binding).expect("shutdown");
    }

    #[gpui::test]
    fn at_graphite_asset_binding_host_foreign_output_grant_fails_after_old_binding_closes(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical service");
        let binding = cx
            .update(|cx| {
                crate::register_asset_binding(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    cx,
                )
            })
            .expect("old binding");
        let policy = sim_runtime::PermissionPolicy::native_runtime_services("other-profile")
            .expect("real foreign policy");
        let foreign = policy
            .authorize_one(
                sim_runtime::NATIVE_API_ASSET_READER_SUBJECT,
                sim_runtime::Capability::Asset {
                    namespace: "output".to_owned(),
                    action: sim_runtime::AssetOperation::Read,
                },
            )
            .expect("sealed foreign grant");
        let result = cx.update(|cx| {
            crate::register_asset_binding_with_output_reader(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                Some(foreign),
                cx,
            )
        });
        assert!(matches!(
            result,
            Err(GraphiteAssetBindingError::ProfileMismatch)
        ));
        assert!(!binding.read_with(cx, |binding, _| binding.is_active()));
        assert!(cx.update(|cx| crate::asset_binding(cx)).is_none());
    }

    #[gpui::test]
    async fn at_graphite_committed_output_workspace_opens_edits_and_revokes_visible_preview(
        cx: &mut TestAppContext,
    ) {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let state = workspace::AppState::test(cx);
            workspace::AppState::set_global(state, cx);
        });
        let fixture = host_reader_fixture().expect("real canonical asset service");
        let png = indexed_request_png(&fixture, 4).expect("real Comfy PNG producer");
        let profile = fixture.input_authorization.profile_id();
        let writer = sim_runtime::authorize_native_output_committer(profile)
            .expect("existing publication grant");
        let policy = sim_runtime::PermissionPolicy::native_runtime_services(profile)
            .expect("existing policy");
        let reader = policy
            .authorize_one(
                sim_runtime::NATIVE_API_ASSET_READER_SUBJECT,
                sim_runtime::Capability::Asset {
                    namespace: "output".to_owned(),
                    action: sim_runtime::AssetOperation::Read,
                },
            )
            .expect("read-only consumer");
        let roots = fixture
            .assets
            .lock()
            .expect("actual service")
            .roots()
            .clone();
        let mut publisher =
            sim_runtime::OutputCommitter::open(roots).expect("fixture owns one publisher");
        let proposal = sim_runtime::OutputProposal::new(
            "60000000-0000-0000-0000-000000000001"
                .parse()
                .expect("proposal UUID"),
            sim_runtime::assets::AssetNamespace::Output,
            "graphite-workspace-source",
            "png",
            0,
            png.descriptor.width,
            png.descriptor.height,
            png.encoded.clone(),
        )
        .expect("actual PNG proposal");
        let receipts = publisher
            .commit_proposal_batch_and_register(
                &[proposal],
                "2026-10-08T12:00:00+00:00".parse().expect("fixture time"),
                &mut fixture.assets.lock().expect("actual indexed owner"),
                &writer,
                &sim_tensor::CancellationToken::default(),
            )
            .expect("real publication transaction");
        let receipt = receipts
            .into_iter()
            .next()
            .expect("genuine committed receipt");
        let receipt_before = receipt.clone();
        let output_identity = receipt.operation().identity.clone();
        let before = fixture
            .assets
            .lock()
            .expect("service")
            .record(&output_identity)
            .expect("real registered output");
        cx.update(|cx| {
            crate::register_asset_binding_with_output_reader(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                Some(reader.clone()),
                cx,
            )
        })
        .expect("same host owner and read-only grant");
        let mut output = request(&png);
        output.asset_identity = output_identity.clone();
        output.identity.asset_id = output_identity
            .to_reference()
            .expect("canonical reference")
            .into();
        output.identity.version_id = "supplied-test-version-not-a-durable-catalog".into();
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (host, window) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = host.read_with(window, |host, _| host.workspace().clone());
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                crate::open_committed_asset_image(output, &receipt, 4096, workspace, window, cx)
            })
            .expect("real committed image Item");
        window.run_until_parked();
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_item_as::<crate::ImportedImageItem>(cx)
                .expect("active image")
                .entity_id()),
            item.entity_id()
        );
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        let session = item.read_with(window, |item, _| item.session().clone());
        window.dispatch_action(crate::InvertImage);
        window.run_until_parked();
        assert!(session.read_with(window, |session, _| session.result_is_fresh()));
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        let service = fixture.assets.lock().expect("same service");
        assert_eq!(service.record(&output_identity), Some(before));
        assert_eq!(
            service
                .read_verified(
                    &output_identity,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("unchanged producer bytes"),
            png.encoded
        );
        drop(service);
        assert_eq!(receipt, receipt_before);
        window
            .update(|_, cx| crate::clear_asset_binding(cx))
            .expect("real owner revocation");
        window.run_until_parked();
        assert!(session.read_with(window, |session, _| session.resource_state().is_revoked()));
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert_eq!(receipt, receipt_before);
    }
}

mod black_and_white_workspace_controls {
    use super::*;
    use crate::graph::ImportedImageItem;
    use crate::resources::tests::{
        HostReaderFixture, IndexedRequestPng, host_reader_fixture, indexed_request_png,
    };
    use crate::session::{GraphiteImageSession, GraphitePngLoad};
    use gpui::{Entity, VisualTestContext};
    use graphics_core::resources::{
        AdmittedCpuImage, CpuBlackAndWhiteParameters, CpuImageEdit, CpuImageResult,
        EncodedPngBudget,
    };
    use std::sync::Arc;
    use workspace::item::Item as _;

    struct NativeImage {
        fixture: HostReaderFixture,
        png: IndexedRequestPng,
        session: Entity<GraphiteImageSession>,
        item: Entity<ImportedImageItem>,
        workspace: Entity<workspace::Workspace>,
    }

    fn load(native: &NativeImage) -> GraphitePngLoad {
        GraphitePngLoad {
            assets: native.fixture.assets.clone(),
            asset_identity: native.png.asset_identity.clone(),
            authorization: native.fixture.input_authorization.clone(),
            identity: native.png.image_identity.clone(),
            descriptor: native.png.descriptor,
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
        }
    }

    async fn native_image(cx: &mut TestAppContext) -> (NativeImage, &mut VisualTestContext) {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
            crate::graph::init(cx);
        });
        let fixture = host_reader_fixture().expect("genuine service/grant fixture");
        let png = indexed_request_png(&fixture, 4).expect("actual indexed RGBA PNG producer");
        let session = cx.new(|_| GraphiteImageSession::new(4096));
        session
            .update(cx, |session, cx| {
                session.load_png(
                    GraphitePngLoad {
                        assets: fixture.assets.clone(),
                        asset_identity: png.asset_identity.clone(),
                        authorization: fixture.input_authorization.clone(),
                        identity: png.image_identity.clone(),
                        descriptor: png.descriptor,
                        encoded_budget: EncodedPngBudget {
                            max_encoded_bytes: 16 * 1024,
                            max_pixel_bytes: 4096,
                            max_metadata_bytes: 4096,
                        },
                    },
                    cx,
                )
            })
            .expect("authenticated original network load");
        cx.run_until_parked();
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (host, window) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = host.read_with(window, |host, _| host.workspace().clone());
        let item = workspace.update_in(window, |workspace, window, cx| {
            let item =
                cx.new(|cx| ImportedImageItem::from_session(session.clone(), 4096, window, cx));
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
            item.focus_handle(cx).focus(window, cx);
            item
        });
        window.update(|_, cx| {
            let bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/default-linux.json",
                cx,
            )
            .expect("actual Linux application keymap");
            cx.bind_keys(bindings);
        });
        window.simulate_resize(gpui::size(gpui::px(1400.), gpui::px(1400.)));
        window.run_until_parked();
        (
            NativeImage {
                fixture,
                png,
                session,
                item,
                workspace,
            },
            window,
        )
    }

    fn click(window: &mut VisualTestContext, selector: &'static str) {
        let bounds = window
            .debug_bounds(selector)
            .expect("visible native control");
        window.simulate_click(bounds.center(), gpui::Modifiers::default());
        window.run_until_parked();
    }

    fn edit(
        window: &mut VisualTestContext,
        item: &Entity<ImportedImageItem>,
        index: usize,
        text: &str,
    ) {
        let editor = item.read_with(window, |item, _| {
            item.black_and_white_editors[index].clone()
        });
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input(text);
        window.run_until_parked();
    }

    fn committed(
        window: &VisualTestContext,
        native: &NativeImage,
    ) -> (
        Arc<AdmittedCpuImage>,
        Arc<CpuImageResult>,
        CpuBlackAndWhiteParameters,
    ) {
        native.session.read_with(window, |session, _| {
            let Some(CpuImageEdit::BlackAndWhite(parameters)) = session.operation() else {
                panic!("selected original Black & White operation");
            };
            assert!(session.result_is_fresh());
            (
                session
                    .resource_state()
                    .committed()
                    .expect("original parent")
                    .clone(),
                session.result().expect("original result").clone(),
                parameters,
            )
        })
    }

    fn assert_original(window: &VisualTestContext, native: &NativeImage) {
        let (source, result, parameters) = committed(window, native);
        let expected = graphics_core::resources::evaluate_cpu_image_edit(
            &source,
            CpuImageEdit::BlackAndWhite(parameters),
            4096,
        )
        .expect("original source constructor oracle");
        assert_eq!(result.image(), expected.image());
        assert_eq!(result.pixels_sha256(), expected.pixels_sha256());
        assert_eq!(result.attributes(), expected.attributes());
        assert_eq!(result.input_identity(), source.identity());
        assert_eq!(result.input_origin(), source.origin());
        assert_eq!(result.operation(), CpuImageEdit::BlackAndWhite(parameters));
        let prepared = graphics_core::export::prepare_cpu_result_png(&result, 4096)
            .expect("original result PNG preparation receipt");
        assert_eq!(prepared.input_identity(), source.identity());
        assert_eq!(prepared.input_origin(), source.origin());
        assert_eq!(
            prepared.source_operation(),
            Some(CpuImageEdit::BlackAndWhite(parameters))
        );
        assert_eq!(
            prepared.source_pixels_sha256(),
            Some(result.pixels_sha256())
        );
        assert_eq!(
            (prepared.width(), prepared.height()),
            (result.image().width, result.image().height)
        );
        assert_eq!(
            prepared.source_revision(),
            "28ed22824fae8828756f6f4163db62f3585f21e0"
        );
        assert_eq!(
            (result.image().width, result.image().height),
            (source.image().width, source.image().height)
        );
        for (actual, original) in result.image().data.iter().zip(&source.image().data) {
            assert_eq!(
                actual.to_gamma_srgb_channels()[3].to_bits(),
                original.to_gamma_srgb_channels()[3].to_bits()
            );
        }
    }

    #[gpui::test]
    async fn at_graphite_black_and_white_workspace_template_mapping_and_original_pixels(
        cx: &mut TestAppContext,
    ) {
        let (native, window) = native_image(cx).await;
        let record = native
            .fixture
            .assets
            .lock()
            .expect("service")
            .record(&native.png.asset_identity)
            .expect("indexed producer");
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
        click(window, "GRAPHITE-BLACK-WHITE-SELECT");
        let (source, result, parameters) = committed(window, &native);
        assert!(!parameters.use_tint);
        assert_eq!(
            parameters.tint_gamma_srgb.map(f32::to_bits),
            [225_f32 / 255., 211_f32 / 255., 179_f32 / 255., 1.].map(f32::to_bits)
        );
        assert_eq!(
            [
                parameters.reds,
                parameters.yellows,
                parameters.greens,
                parameters.cyans,
                parameters.blues,
                parameters.magentas
            ],
            [40., 60., 40., 60., 20., 80.]
        );
        assert!((result.image().data[0].to_gamma_srgb_channels()[0] - 0.4).abs() < 1e-5);
        assert!(native.item.read_with(window, |item, cx| {
            item.black_and_white_editors[0].read(cx).read_only(cx)
        }));
        assert_original(window, &native);
        for (index, value) in [
            (1, "11"),
            (2, "22"),
            (3, "33"),
            (4, "44"),
            (5, "55"),
            (6, "66"),
        ] {
            edit(window, &native.item, index, value);
            window.simulate_keystrokes("enter");
            window.run_until_parked();
            assert_original(window, &native);
        }
        let (current_source, current_result, parameters) = committed(window, &native);
        assert!(Arc::ptr_eq(&source, &current_source));
        assert_eq!(
            [
                parameters.reds,
                parameters.yellows,
                parameters.greens,
                parameters.cyans,
                parameters.blues,
                parameters.magentas
            ],
            [11., 22., 33., 44., 55., 66.]
        );
        assert!((current_result.image().data[0].to_gamma_srgb_channels()[0] - 0.11).abs() < 1e-5);
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
        let service = native.fixture.assets.lock().expect("same service");
        assert_eq!(service.record(&native.png.asset_identity), Some(record));
        let bytes = service
            .read_verified(
                &native.png.asset_identity,
                &native.fixture.input_authorization,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("unchanged indexed PNG");
        assert_eq!(bytes.as_slice(), native.png.encoded.as_slice());
    }

    #[gpui::test]
    async fn at_graphite_black_and_white_workspace_expression_enter_blur_escape_reset_recovery(
        cx: &mut TestAppContext,
    ) {
        let (native, window) = native_image(cx).await;
        click(window, "GRAPHITE-BLACK-WHITE-SELECT");
        let (source, before, _) = committed(window, &native);
        edit(window, &native.item, 1, "5 * 11");
        assert!(Arc::ptr_eq(&before, &committed(window, &native).1));
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let (_, after, parameters) = committed(window, &native);
        assert_eq!(parameters.reds, 55.);
        assert!(!Arc::ptr_eq(&before, &after));
        native.item.update_in(window, |item, window, cx| {
            item.focus_handle(cx).focus(window, cx)
        });
        window.run_until_parked();
        assert!(Arc::ptr_eq(&after, &committed(window, &native).1));
        edit(window, &native.item, 1, "1 +");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert!(native.item.read_with(window, |item, _| {
            item.black_and_white_diagnostic().is_some()
        }));
        assert!(window.debug_bounds("GRAPHITE-BLACK-WHITE-ERROR").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                .is_some()
        );
        assert!(Arc::ptr_eq(&after, &committed(window, &native).1));
        native.session.update(window, |session, cx| {
            let Some(CpuImageEdit::BlackAndWhite(mut parameters)) = session.operation() else {
                panic!("operation")
            };
            parameters.reds = 63.;
            session.set_operation(Some(CpuImageEdit::BlackAndWhite(parameters)), cx);
        });
        window.run_until_parked();
        let (_, external, _) = committed(window, &native);
        edit(window, &native.item, 1, "invalid");
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        let red = native
            .item
            .read_with(window, |item, _| item.black_and_white_editors[1].clone());
        assert_eq!(red.read_with(window, |editor, cx| editor.text(cx)), "63");
        assert!(!window.update(|window, cx| red.read(cx).is_focused(window)));
        assert!(native.item.read_with(window, |item, _| {
            item.black_and_white_diagnostic().is_none()
        }));
        assert!(Arc::ptr_eq(&external, &committed(window, &native).1));
        edit(window, &native.item, 1, "1000");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_eq!(committed(window, &native).2.reds, 300.);
        edit(window, &native.item, 1, "-1000");
        native.item.update_in(window, |item, window, cx| {
            item.focus_handle(cx).focus(window, cx)
        });
        window.run_until_parked();
        assert_eq!(committed(window, &native).2.reds, -200.);
        click(window, "GRAPHITE-BLACK-WHITE-REDS-RESET");
        assert_eq!(committed(window, &native).2.reds, 40.);
        assert!(Arc::ptr_eq(&source, &committed(window, &native).0));
        assert_original(window, &native);
    }

    #[gpui::test]
    async fn at_graphite_black_and_white_workspace_tint_disabled_alpha_and_invalid_recovery(
        cx: &mut TestAppContext,
    ) {
        let (native, window) = native_image(cx).await;
        click(window, "GRAPHITE-BLACK-WHITE-SELECT");
        let (_, before, initial) = committed(window, &native);
        click(window, "GRAPHITE-BLACK-WHITE-USE-TINT");
        assert!(!native.item.read_with(window, |item, cx| {
            item.black_and_white_editors[0].read(cx).read_only(cx)
        }));
        edit(window, &native.item, 0, "1, 0, 0, 0.5");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let (_, tinted, parameters) = committed(window, &native);
        assert!(parameters.use_tint);
        assert_eq!(parameters.tint_gamma_srgb, [1., 0., 0., 0.5]);
        assert_original(window, &native);
        edit(window, &native.item, 0, "1, 0, 0, -0");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let (_, zero_alpha, parameters) = committed(window, &native);
        assert_eq!(parameters.tint_gamma_srgb[3].to_bits(), (-0_f32).to_bits());
        assert_eq!(zero_alpha.image(), tinted.image());
        edit(window, &native.item, 0, "1, 0, 0");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert!(native.item.read_with(window, |item, _| {
            item.black_and_white_diagnostic().is_some()
        }));
        assert!(Arc::ptr_eq(&zero_alpha, &committed(window, &native).1));
        edit(window, &native.item, 0, "1, 0, 0, 1");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert!(native.item.read_with(window, |item, _| {
            item.black_and_white_diagnostic().is_none()
        }));
        click(window, "GRAPHITE-BLACK-WHITE-USE-TINT");
        let (_, disabled, parameters) = committed(window, &native);
        assert!(!parameters.use_tint);
        assert_eq!(parameters.tint_gamma_srgb, [1., 0., 0., 1.]);
        assert_eq!(disabled.image(), before.image());
        assert_eq!(
            [
                parameters.reds,
                parameters.yellows,
                parameters.greens,
                parameters.cyans,
                parameters.blues,
                parameters.magentas
            ],
            [
                initial.reds,
                initial.yellows,
                initial.greens,
                initial.cyans,
                initial.blues,
                initial.magentas
            ]
        );
        assert!(native.item.read_with(window, |item, cx| {
            item.black_and_white_editors[0].read(cx).read_only(cx)
        }));
        assert_original(window, &native);
    }

    #[gpui::test]
    async fn at_graphite_black_and_white_workspace_split_local_drafts_and_source_cancel_revoke(
        cx: &mut TestAppContext,
    ) {
        let (native, window) = native_image(cx).await;
        click(window, "GRAPHITE-BLACK-WHITE-SELECT");
        edit(window, &native.item, 1, "invalid");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let split = native
            .item
            .update_in(window, |item, window, cx| {
                item.clone_on_split(None, window, cx)
            })
            .await
            .expect("existing split");
        native.workspace.update_in(window, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx)
        });
        window.run_until_parked();
        let (session, presentation, editor) = native.item.read_with(window, |item, _| {
            (
                item.session().entity_id(),
                item.presentation().entity_id(),
                item.black_and_white_editors[1].entity_id(),
            )
        });
        split.read_with(window, |item, _| {
            assert_eq!(item.session().entity_id(), session);
            assert_eq!(item.presentation().entity_id(), presentation);
            assert_ne!(item.black_and_white_editors[1].entity_id(), editor);
            assert!(item.black_and_white_diagnostic().is_none());
        });
        edit(window, &split, 1, "71");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_eq!(committed(window, &native).2.reds, 71.);
        assert!(split.read_with(window, |item, cx| item.presentation().read(cx).is_fresh()));
        native.workspace.update_in(window, |workspace, window, cx| {
            assert!(workspace.activate_item(&native.item, true, true, window, cx))
        });
        window.run_until_parked();
        assert_eq!(
            native
                .item
                .read_with(window, |item, cx| item.black_and_white_editors[1]
                    .read(cx)
                    .text(cx)),
            "invalid"
        );
        assert!(native.item.read_with(window, |item, _| {
            item.black_and_white_diagnostic().is_some()
        }));
        assert!(
            native
                .item
                .read_with(window, |item, cx| item.presentation().read(cx).is_fresh())
        );
        edit(window, &native.item, 1, "invalid");
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert_eq!(
            native
                .item
                .read_with(window, |item, cx| item.black_and_white_editors[1]
                    .read(cx)
                    .text(cx)),
            "71"
        );
        let (source, last, _) = committed(window, &native);
        let mut pending = load(&native);
        pending.identity.version_id = "explicit-test-successor-version".into();
        native.session.update(window, |session, cx| {
            session.load_png(pending, cx).expect("successor request");
            session.cancel(cx);
        });
        window.run_until_parked();
        native.session.read_with(window, |session, _| {
            assert!(!session.result_is_fresh());
            assert!(Arc::ptr_eq(session.result().expect("last valid"), &last));
            assert!(Arc::ptr_eq(
                session
                    .resource_state()
                    .committed()
                    .expect("retained parent"),
                &source
            ));
            assert!(!session.resource_state().has_active_request());
        });
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                .is_some()
        );
        native
            .session
            .update(window, |session, cx| session.load_png(load(&native), cx))
            .expect("explicit recovery");
        window.run_until_parked();
        assert_original(window, &native);
        native
            .session
            .update(window, |session, cx| session.revoke(source.identity(), cx))
            .expect("host revocation");
        window.run_until_parked();
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-REVOKED").is_some());
        native.session.read_with(window, |session, _| {
            assert!(session.resource_state().is_revoked());
            assert!(session.result().is_none());
        });
        assert!(native.item.read_with(window, |item, cx| {
            item.presentation().read(cx).image().is_none()
        }));
        assert!(split.read_with(window, |item, cx| {
            item.presentation().read(cx).image().is_none()
        }));
    }

    fn edit_brightness_contrast(
        window: &mut VisualTestContext,
        item: &Entity<ImportedImageItem>,
        index: usize,
        text: &str,
    ) {
        let editor = item.read_with(window, |item, _| {
            item.brightness_contrast_editors[index].clone()
        });
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input(text);
        window.run_until_parked();
    }

    fn committed_brightness_contrast(
        window: &VisualTestContext,
        native: &NativeImage,
    ) -> (
        Arc<AdmittedCpuImage>,
        Arc<CpuImageResult>,
        graphics_core::resources::CpuBrightnessContrastParameters,
    ) {
        native.session.read_with(window, |session, _| {
            let Some(CpuImageEdit::BrightnessContrast(parameters)) = session.operation() else {
                panic!("selected original Brightness/Contrast operation");
            };
            assert!(session.result_is_fresh());
            (
                session
                    .resource_state()
                    .committed()
                    .expect("original parent")
                    .clone(),
                session.result().expect("original result").clone(),
                parameters,
            )
        })
    }

    fn assert_original_brightness_contrast(window: &VisualTestContext, native: &NativeImage) {
        let (source, result, parameters) = committed_brightness_contrast(window, native);
        let expected = graphics_core::resources::evaluate_cpu_image_edit(
            &source,
            CpuImageEdit::BrightnessContrast(parameters),
            4096,
        )
        .expect("original Brightness/Contrast constructor oracle");
        assert_eq!(result.image(), expected.image());
        assert_eq!(result.pixels_sha256(), expected.pixels_sha256());
        assert_eq!(result.attributes(), expected.attributes());
        assert_eq!(result.input_identity(), source.identity());
        assert_eq!(result.input_origin(), source.origin());
        assert_eq!(
            result.operation(),
            CpuImageEdit::BrightnessContrast(parameters)
        );
        assert_eq!(
            (result.image().width, result.image().height),
            (source.image().width, source.image().height)
        );
        for (actual, original) in result.image().data.iter().zip(&source.image().data) {
            assert_eq!(
                actual.to_gamma_srgb_channels()[3].to_bits(),
                original.to_gamma_srgb_channels()[3].to_bits()
            );
        }
        let prepared = graphics_core::export::prepare_cpu_result_png(&result, 4096)
            .expect("original result PNG receipt");
        assert_eq!(prepared.input_identity(), source.identity());
        assert_eq!(prepared.input_origin(), source.origin());
        assert_eq!(
            prepared.source_operation(),
            Some(CpuImageEdit::BrightnessContrast(parameters))
        );
        assert_eq!(
            prepared.source_pixels_sha256(),
            Some(result.pixels_sha256())
        );
        assert_eq!(
            prepared.source_revision(),
            "28ed22824fae8828756f6f4163db62f3585f21e0"
        );
    }

    #[gpui::test]
    async fn at_graphite_brightness_contrast_workspace_neutral_mapping_original_pixels_and_provenance(
        cx: &mut TestAppContext,
    ) {
        let (native, window) = native_image(cx).await;
        let record = native
            .fixture
            .assets
            .lock()
            .expect("service")
            .record(&native.png.asset_identity)
            .expect("indexed producer");
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-SELECT");
        let (source, neutral, parameters) = committed_brightness_contrast(window, &native);
        assert_eq!(parameters.brightness.to_bits(), 0_f64.to_bits());
        assert_eq!(parameters.contrast.to_bits(), 0_f64.to_bits());
        assert!(!parameters.use_classic);
        assert_eq!(parameters.classic_pivot, 127.);
        assert!(
            window
                .debug_bounds("GRAPHITE-BRIGHTNESS-CONTRAST-BRIGHTNESS")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-BRIGHTNESS-CONTRAST-CONTRAST")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-BRIGHTNESS-CONTRAST-CLASSIC-PIVOT")
                .is_none()
        );
        for (actual, original) in neutral.image().data.iter().zip(&source.image().data) {
            for (actual, original) in actual
                .to_gamma_srgb_channels()
                .into_iter()
                .zip(original.to_gamma_srgb_channels())
            {
                assert!((actual - original).abs() < 1e-5);
            }
        }
        assert_original_brightness_contrast(window, &native);
        edit_brightness_contrast(window, &native.item, 0, "63");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        edit_brightness_contrast(window, &native.item, 1, "24");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let (current_source, result, parameters) = committed_brightness_contrast(window, &native);
        assert!(Arc::ptr_eq(&current_source, &source));
        assert_eq!(parameters.brightness, 63.);
        assert_eq!(parameters.contrast, 24.);
        assert_eq!(parameters.classic_pivot, 127.);
        assert!(!parameters.use_classic);
        assert_ne!(result.image(), neutral.image());
        assert_original_brightness_contrast(window, &native);
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
        let service = native.fixture.assets.lock().expect("same service");
        assert_eq!(service.record(&native.png.asset_identity), Some(record));
        assert_eq!(
            service
                .read_verified(
                    &native.png.asset_identity,
                    &native.fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("immutable indexed PNG"),
            native.png.encoded
        );
    }

    #[gpui::test]
    async fn at_graphite_brightness_contrast_workspace_expression_commit_escape_bounds_and_recovery(
        cx: &mut TestAppContext,
    ) {
        let (native, window) = native_image(cx).await;
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-SELECT");
        let (source, before, _) = committed_brightness_contrast(window, &native);
        edit_brightness_contrast(window, &native.item, 0, "5 * 11");
        assert!(Arc::ptr_eq(
            &before,
            &committed_brightness_contrast(window, &native).1
        ));
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let (_, after, parameters) = committed_brightness_contrast(window, &native);
        assert_eq!(parameters.brightness, 55.);
        assert!(!Arc::ptr_eq(&before, &after));
        native.item.update_in(window, |item, window, cx| {
            item.focus_handle(cx).focus(window, cx)
        });
        window.run_until_parked();
        assert!(Arc::ptr_eq(
            &after,
            &committed_brightness_contrast(window, &native).1
        ));
        edit_brightness_contrast(window, &native.item, 0, "1 +");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert!(native.item.read_with(window, |item, cx| {
            item.brightness_contrast_diagnostic(cx).is_some()
        }));
        assert!(
            window
                .debug_bounds("GRAPHITE-BRIGHTNESS-CONTRAST-ERROR")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                .is_some()
        );
        assert!(Arc::ptr_eq(
            &after,
            &committed_brightness_contrast(window, &native).1
        ));
        native.session.update(window, |session, cx| {
            let Some(CpuImageEdit::BrightnessContrast(mut parameters)) = session.operation() else {
                panic!("operation")
            };
            parameters.brightness = 63.;
            session.set_operation(Some(CpuImageEdit::BrightnessContrast(parameters)), cx);
        });
        window.run_until_parked();
        let (_, external, _) = committed_brightness_contrast(window, &native);
        let brightness = native.item.read_with(window, |item, _| {
            item.brightness_contrast_editors[0].clone()
        });
        assert_eq!(
            brightness.read_with(window, |editor, cx| editor.text(cx)),
            "1 +"
        );
        edit_brightness_contrast(window, &native.item, 0, "invalid");
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert_eq!(
            brightness.read_with(window, |editor, cx| editor.text(cx)),
            "63"
        );
        assert!(!window.update(|window, cx| brightness.read(cx).is_focused(window)));
        assert!(native.item.read_with(window, |item, cx| {
            item.brightness_contrast_diagnostic(cx).is_none()
        }));
        assert!(Arc::ptr_eq(
            &external,
            &committed_brightness_contrast(window, &native).1
        ));
        for (index, text, expected) in [
            (0, "1000", 150.),
            (0, "-1000", -150.),
            (1, "-1000", -50.),
            (1, "1000", 100.),
        ] {
            edit_brightness_contrast(window, &native.item, index, text);
            native.item.update_in(window, |item, window, cx| {
                item.focus_handle(cx).focus(window, cx)
            });
            window.run_until_parked();
            let parameters = committed_brightness_contrast(window, &native).2;
            assert_eq!(
                if index == 0 {
                    parameters.brightness
                } else {
                    parameters.contrast
                },
                expected
            );
            assert_original_brightness_contrast(window, &native);
        }
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-BRIGHTNESS-RESET");
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-CONTRAST-RESET");
        let (_, _, parameters) = committed_brightness_contrast(window, &native);
        assert_eq!((parameters.brightness, parameters.contrast), (0., 0.));
        assert!(Arc::ptr_eq(
            &source,
            &committed_brightness_contrast(window, &native).0
        ));
        assert_original_brightness_contrast(window, &native);
    }

    #[gpui::test]
    async fn at_graphite_brightness_contrast_workspace_classic_preserves_values_and_conditional_pivot(
        cx: &mut TestAppContext,
    ) {
        let (native, window) = native_image(cx).await;
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-SELECT");
        for (index, text) in [(0, "140"), (1, "-40")] {
            edit_brightness_contrast(window, &native.item, index, text);
            window.simulate_keystrokes("enter");
            window.run_until_parked();
        }
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-USE-CLASSIC");
        let (_, _, classic) = committed_brightness_contrast(window, &native);
        assert!(classic.use_classic);
        assert_eq!((classic.brightness, classic.contrast), (140., -40.));
        assert_eq!(classic.classic_pivot, 127.);
        assert!(
            window
                .debug_bounds("GRAPHITE-BRIGHTNESS-CONTRAST-CLASSIC-PIVOT")
                .is_some()
        );
        assert_eq!(
            native
                .item
                .read_with(window, |item, cx| item.brightness_contrast_editors[0]
                    .read(cx)
                    .text(cx)),
            "140"
        );
        assert_original_brightness_contrast(window, &native);
        for (text, expected) in [("500", 255.), ("-20", 0.), ("255 / 3", 85.)] {
            edit_brightness_contrast(window, &native.item, 2, text);
            window.simulate_keystrokes("enter");
            window.run_until_parked();
            assert_eq!(
                committed_brightness_contrast(window, &native)
                    .2
                    .classic_pivot,
                expected
            );
            assert_original_brightness_contrast(window, &native);
        }
        let (_, before_invalid, _) = committed_brightness_contrast(window, &native);
        edit_brightness_contrast(window, &native.item, 2, "1 +");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert!(Arc::ptr_eq(
            &before_invalid,
            &committed_brightness_contrast(window, &native).1
        ));
        assert!(
            window
                .debug_bounds("GRAPHITE-BRIGHTNESS-CONTRAST-ERROR")
                .is_some()
        );
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-USE-CLASSIC");
        assert!(
            window
                .debug_bounds("GRAPHITE-BRIGHTNESS-CONTRAST-CLASSIC-PIVOT")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-BRIGHTNESS-CONTRAST-ERROR")
                .is_none()
        );
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
        let (_, _, current) = committed_brightness_contrast(window, &native);
        assert!(!current.use_classic);
        assert_eq!(
            (current.brightness, current.contrast, current.classic_pivot),
            (140., -40., 85.)
        );
        assert_original_brightness_contrast(window, &native);
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-USE-CLASSIC");
        assert!(
            window
                .debug_bounds("GRAPHITE-BRIGHTNESS-CONTRAST-ERROR")
                .is_some()
        );
        edit_brightness_contrast(window, &native.item, 2, "invalid");
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert_eq!(
            native
                .item
                .read_with(window, |item, cx| item.brightness_contrast_editors[2]
                    .read(cx)
                    .text(cx)),
            "85"
        );
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-CLASSIC-PIVOT-RESET");
        assert_eq!(
            committed_brightness_contrast(window, &native)
                .2
                .classic_pivot,
            127.
        );
        for (index, text, expected) in [(0, "1000", 100.), (0, "-1000", -100.), (1, "-1000", -100.)]
        {
            edit_brightness_contrast(window, &native.item, index, text);
            window.simulate_keystrokes("enter");
            window.run_until_parked();
            let parameters = committed_brightness_contrast(window, &native).2;
            assert_eq!(
                if index == 0 {
                    parameters.brightness
                } else {
                    parameters.contrast
                },
                expected
            );
        }
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-USE-CLASSIC");
        let (_, _, current) = committed_brightness_contrast(window, &native);
        assert_eq!((current.brightness, current.contrast), (-100., -100.));
        assert!(!current.use_classic);
        click(window, "GRAPHITE-BLACK-WHITE-SELECT");
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-SELECT");
        assert_eq!(committed_brightness_contrast(window, &native).2, current);
        assert_original_brightness_contrast(window, &native);
    }

    #[gpui::test]
    async fn at_graphite_brightness_contrast_workspace_split_drafts_cancel_recovery_and_revoke(
        cx: &mut TestAppContext,
    ) {
        let (native, window) = native_image(cx).await;
        click(window, "GRAPHITE-BRIGHTNESS-CONTRAST-SELECT");
        edit_brightness_contrast(window, &native.item, 0, "invalid");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        let split = native
            .item
            .update_in(window, |item, window, cx| {
                item.clone_on_split(None, window, cx)
            })
            .await
            .expect("existing split");
        native.workspace.update_in(window, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx)
        });
        window.run_until_parked();
        let (session, presentation, editor) = native.item.read_with(window, |item, _| {
            (
                item.session().entity_id(),
                item.presentation().entity_id(),
                item.brightness_contrast_editors[0].entity_id(),
            )
        });
        split.read_with(window, |item, cx| {
            assert_eq!(item.session().entity_id(), session);
            assert_eq!(item.presentation().entity_id(), presentation);
            assert_ne!(item.brightness_contrast_editors[0].entity_id(), editor);
            assert!(item.brightness_contrast_diagnostic(cx).is_none());
        });
        edit_brightness_contrast(window, &split, 0, "71");
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert_eq!(
            committed_brightness_contrast(window, &native).2.brightness,
            71.
        );
        assert!(split.read_with(window, |item, cx| item.presentation().read(cx).is_fresh()));
        native.workspace.update_in(window, |workspace, window, cx| {
            assert!(workspace.activate_item(&native.item, true, true, window, cx))
        });
        window.run_until_parked();
        assert_eq!(
            native
                .item
                .read_with(window, |item, cx| item.brightness_contrast_editors[0]
                    .read(cx)
                    .text(cx)),
            "invalid"
        );
        assert!(native.item.read_with(window, |item, cx| {
            item.brightness_contrast_diagnostic(cx).is_some()
        }));
        assert!(
            native
                .item
                .read_with(window, |item, cx| item.presentation().read(cx).is_fresh())
        );
        edit_brightness_contrast(window, &native.item, 0, "invalid");
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        assert_eq!(
            native
                .item
                .read_with(window, |item, cx| item.brightness_contrast_editors[0]
                    .read(cx)
                    .text(cx)),
            "71"
        );
        let (source, last, _) = committed_brightness_contrast(window, &native);
        let mut pending = load(&native);
        pending.identity.version_id = "explicit-brightness-test-successor-version".into();
        native.session.update(window, |session, cx| {
            session.load_png(pending, cx).expect("successor request");
            session.cancel(cx);
        });
        window.run_until_parked();
        native.session.read_with(window, |session, _| {
            assert!(!session.result_is_fresh());
            assert!(Arc::ptr_eq(session.result().expect("last valid"), &last));
            assert!(Arc::ptr_eq(
                session
                    .resource_state()
                    .committed()
                    .expect("retained parent"),
                &source
            ));
            assert!(!session.resource_state().has_active_request());
        });
        assert!(
            window
                .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                .is_some()
        );
        native
            .session
            .update(window, |session, cx| session.load_png(load(&native), cx))
            .expect("explicit recovery");
        window.run_until_parked();
        assert_original_brightness_contrast(window, &native);
        native
            .session
            .update(window, |session, cx| session.revoke(source.identity(), cx))
            .expect("host revocation");
        window.run_until_parked();
        assert!(window.debug_bounds("GRAPHITE-IMPORTED-REVOKED").is_some());
        native.session.read_with(window, |session, _| {
            assert!(session.resource_state().is_revoked());
            assert!(session.result().is_none());
        });
        assert!(native.item.read_with(window, |item, cx| {
            item.presentation().read(cx).image().is_none()
        }));
        assert!(split.read_with(window, |item, cx| {
            item.presentation().read(cx).image().is_none()
        }));
    }
}

mod unary_png_preparation_controls {
    use super::*;
    use crate::resources::tests::{
        HostReaderFixture, IndexedRequestPng, host_reader_fixture, indexed_request_png,
    };
    use crate::session::{
        GraphiteEncodedPngLoad, GraphiteImageSession, GraphiteMixSession, ImageSessionDiagnostic,
    };
    use gpui::Entity;
    use graphics_core::resources::{
        AlphaAssociation, CpuBlendMode, CpuBrightnessContrastParameters, CpuImageEdit,
        CpuMixBudget, CpuMixParameters, CpuSampleFormat, EncodedPngBudget, ImageColorSpace,
    };
    use std::sync::Arc;

    struct ReadyImage {
        fixture: HostReaderFixture,
        png: IndexedRequestPng,
        owner: Entity<GraphiteImageSession>,
    }

    fn load(ready: &ReadyImage, version: &str) -> GraphiteEncodedPngLoad {
        let mut identity = ready.png.image_identity.clone();
        identity.version_id = version.into();
        GraphiteEncodedPngLoad {
            assets: ready.fixture.assets.clone(),
            asset_identity: ready.png.asset_identity.clone(),
            authorization: ready.fixture.input_authorization.clone(),
            identity,
            transform: [1., 0., 0., 1., 0., 0.],
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
        }
    }

    fn ready_image(cx: &mut TestAppContext) -> ReadyImage {
        init(cx);
        let fixture = host_reader_fixture().expect("real indexed service and Input grant");
        let png = indexed_request_png(&fixture, 4).expect("actual tensor PNG producer");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        assert!(
            owner
                .update(cx, |owner, cx| owner.prepare_png(4096, cx))
                .is_err()
        );
        let ready = ReadyImage {
            fixture,
            png,
            owner,
        };
        ready
            .owner
            .update(cx, |owner, cx| {
                owner.load_encoded_png(load(&ready, "test-supplied-unary-png-v1"), cx)
            })
            .expect("real authenticated original-network PNG load");
        cx.run_until_parked();
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.resource_state().is_fresh())
        );
        ready
    }

    #[gpui::test]
    async fn at_graphite_unary_png_original_and_edits_preserve_bytes_and_provenance(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_image(cx);
        let original_file = ready
            .fixture
            .assets
            .lock()
            .expect("actual canonical asset service")
            .read_verified(
                &ready.png.asset_identity,
                &ready.fixture.input_authorization,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("actual indexed producer file");
        let indexed_record = ready
            .fixture
            .assets
            .lock()
            .expect("real asset service")
            .record(&ready.png.asset_identity)
            .expect("indexed record");
        let source = ready.owner.read_with(cx, |owner, _| {
            owner
                .resource_state()
                .committed()
                .expect("admitted source")
                .clone()
        });
        let source_image = source.image().clone();
        let source_revision = ready
            .owner
            .read_with(cx, |owner, _| owner.resource_state().revision());
        let mix = cx.new(|cx| {
            GraphiteMixSession::new(
                [ready.owner.clone(), ready.owner.clone()],
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: 37.5,
                },
                CpuMixBudget {
                    max_parent_bytes: 4096,
                    max_result_bytes: 4096,
                },
                cx,
            )
        });
        cx.run_until_parked();
        let (mix_generation, mix_result) = mix.read_with(cx, |mix, cx| {
            (
                mix.generation(),
                mix.result(cx).expect("original Mix result").clone(),
            )
        });
        mix.update(cx, |mix, cx| mix.cancel(cx));

        for operation in [
            None,
            Some(CpuImageEdit::Invert),
            Some(CpuImageEdit::MakeOpaque),
        ] {
            ready
                .owner
                .update(cx, |owner, cx| owner.set_operation(operation, cx));
            cx.run_until_parked();
            let result = ready
                .owner
                .read_with(cx, |owner, _| owner.result().cloned());
            let expected = match &result {
                Some(result) => graphics_core::export::prepare_cpu_result_png(result, 4096),
                None => graphics_core::export::prepare_png(&source, 4096),
            }
            .expect("single existing original PNG encoder");
            let prepared = ready
                .owner
                .update(cx, |owner, cx| owner.prepare_png(4096, cx))
                .expect("fresh original or selected edit")
                .await
                .expect("coherent preparation");
            assert_eq!(prepared.encoded_bytes(), expected.encoded_bytes());
            assert_eq!(prepared.content_sha256(), expected.content_sha256());
            assert_eq!(prepared.input_identity(), source.identity());
            assert_eq!(prepared.input_origin(), source.origin());
            assert_eq!(prepared.source_operation(), operation);
            assert_eq!(
                prepared.source_pixels_sha256(),
                result.as_ref().map(|result| result.pixels_sha256())
            );
            assert_eq!(prepared.source_revision(), expected.source_revision());
            assert_eq!(
                (prepared.width(), prepared.height()),
                (source_image.width, source_image.height)
            );
            assert_eq!(prepared.sample_format(), CpuSampleFormat::Rgba8);
            assert_eq!(prepared.color_space(), ImageColorSpace::Srgb);
            assert_eq!(prepared.alpha(), AlphaAssociation::Straight);
            assert_eq!(prepared.content_type(), "image/png");
            if operation.is_none() {
                mix.read_with(cx, |mix, cx| {
                    assert_eq!(mix.generation(), mix_generation);
                    assert_eq!(mix.diagnostic(), Some("Mix evaluation cancelled"));
                    assert!(!mix.result_is_fresh(cx));
                    assert!(Arc::ptr_eq(
                        mix.result(cx).expect("same last valid Mix"),
                        &mix_result
                    ));
                });
            }

            ready.owner.read_with(cx, |owner, _| {
                assert_eq!(owner.operation(), operation);
                assert_eq!(owner.resource_state().revision(), source_revision);
                assert!(Arc::ptr_eq(
                    owner.resource_state().committed().expect("same source"),
                    &source
                ));
                assert_eq!(source.image(), &source_image);
                if let Some(result) = &result {
                    assert!(owner.result_is_fresh());
                    assert!(Arc::ptr_eq(
                        owner.result().expect("same committed edit"),
                        result
                    ));
                }
            });
        }
        assert_eq!(
            ready
                .fixture
                .assets
                .lock()
                .expect("actual canonical asset service")
                .read_verified(
                    &ready.png.asset_identity,
                    &ready.fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("unchanged producer bytes"),
            original_file
        );
        let after_record = ready
            .fixture
            .assets
            .lock()
            .expect("same service")
            .record(&ready.png.asset_identity)
            .expect("same indexed record");
        assert_eq!(after_record.identity, indexed_record.identity);
        assert_eq!(after_record.sha256, indexed_record.sha256);
        assert_eq!(after_record.byte_size, indexed_record.byte_size);
        assert_eq!(
            after_record.metadata.workflow,
            indexed_record.metadata.workflow
        );
    }

    #[gpui::test]
    async fn at_graphite_unary_png_rejects_stale_bitwise_operations_and_repeated_preparation(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_image(cx);
        let task = ready
            .owner
            .update(cx, |owner, cx| owner.prepare_png(4096, cx))
            .expect("fresh source");
        let revision = ready
            .owner
            .read_with(cx, |owner, _| owner.resource_state().revision());
        ready
            .owner
            .update(cx, |owner, cx| {
                owner.load_encoded_png(load(&ready, "test-supplied-unary-png-v2"), cx)
            })
            .expect("successor request");
        ready.owner.read_with(cx, |owner, _| {
            assert_eq!(owner.resource_state().revision(), revision);
            assert!(owner.resource_state().has_active_request());
            assert!(!owner.resource_state().is_fresh());
        });
        assert!(
            ready
                .owner
                .update(cx, |owner, cx| owner.prepare_png(4096, cx))
                .is_err()
        );
        assert!(task.await.is_err());
        cx.run_until_parked();
        ready.owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        cx.run_until_parked();
        let task = ready
            .owner
            .update(cx, |owner, cx| owner.prepare_png(4096, cx))
            .expect("fresh Invert");
        ready.owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::MakeOpaque), cx)
        });
        assert!(task.await.is_err());
        cx.run_until_parked();
        let parameters = CpuBrightnessContrastParameters {
            brightness: 0.,
            contrast: 0.,
            use_classic: false,
            classic_pivot: 127.,
        };
        ready.owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::BrightnessContrast(parameters)), cx)
        });
        cx.run_until_parked();
        let task = ready
            .owner
            .update(cx, |owner, cx| owner.prepare_png(4096, cx))
            .expect("positive-zero operation");
        let changed = CpuBrightnessContrastParameters {
            brightness: -0.,
            ..parameters
        };
        ready.owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::BrightnessContrast(changed)), cx)
        });
        assert!(task.await.is_err());
        cx.run_until_parked();
        let result = ready.owner.read_with(cx, |owner, _| {
            owner.result().expect("new bitwise operation").clone()
        });
        let first = ready
            .owner
            .update(cx, |owner, cx| owner.prepare_png(4096, cx))
            .expect("first preparation");
        let second = ready
            .owner
            .update(cx, |owner, cx| owner.prepare_png(4096, cx))
            .expect("successor preparation");
        assert!(first.await.is_err());
        let prepared = second.await.expect("latest preparation accepted");
        assert_eq!(
            prepared.source_operation(),
            Some(CpuImageEdit::BrightnessContrast(changed))
        );
        let expected =
            graphics_core::export::prepare_cpu_result_png(&result, 4096).expect("original codec");
        assert_eq!(prepared.encoded_bytes(), expected.encoded_bytes());
        ready.owner.read_with(cx, |owner, _| {
            assert!(Arc::ptr_eq(owner.result().expect("same result"), &result));
            assert!(owner.diagnostic().is_none());
            assert!(owner.result_is_fresh());
        });
    }

    #[gpui::test]
    async fn at_graphite_unary_png_idle_cancel_revocation_and_release_prevent_publication(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_image(cx);
        let source = ready.owner.read_with(cx, |owner, _| {
            owner
                .resource_state()
                .committed()
                .expect("current source")
                .clone()
        });
        let task = ready
            .owner
            .update(cx, |owner, cx| owner.prepare_png(4096, cx))
            .expect("idle session starts preparation");
        ready.owner.update(cx, |owner, cx| owner.cancel(cx));
        assert!(task.await.is_err());
        ready.owner.read_with(cx, |owner, _| {
            assert_eq!(
                owner.diagnostic(),
                Some(&ImageSessionDiagnostic::Resource(
                    crate::resources::ImageResourceError::Cancelled
                ))
            );
            assert!(owner.resource_state().is_fresh());
            assert!(Arc::ptr_eq(
                owner.resource_state().committed().expect("input retained"),
                &source
            ));
        });
        let recovered = ready
            .owner
            .update(cx, |owner, cx| owner.prepare_png(4096, cx))
            .expect("explicit preparation retry")
            .await
            .expect("retry succeeds");
        assert_eq!(recovered.encoded_bytes(), source.image().to_png());
        let task = ready
            .owner
            .update(cx, |owner, cx| owner.prepare_png(4096, cx))
            .expect("pre-revocation preparation");
        ready
            .owner
            .update(cx, |owner, cx| owner.revoke(source.identity(), cx))
            .expect("actual committed resource revocation");
        assert!(task.await.is_err());
        assert!(
            ready
                .owner
                .update(cx, |owner, cx| owner.prepare_png(4096, cx))
                .is_err()
        );
        assert_eq!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.diagnostic().cloned()),
            Some(ImageSessionDiagnostic::Resource(
                crate::resources::ImageResourceError::Revoked
            ))
        );
        ready
            .owner
            .update(cx, |owner, cx| {
                owner.load_encoded_png(load(&ready, "test-supplied-unary-png-v1"), cx)
            })
            .expect("explicit validated recovery");
        cx.run_until_parked();
        let task = ready
            .owner
            .update(cx, |owner, cx| owner.prepare_png(4096, cx))
            .expect("pre-release preparation");
        let weak = ready.owner.downgrade();
        drop(ready.owner);
        cx.run_until_parked();
        assert!(weak.upgrade().is_none());
        assert!(task.await.is_err());
        assert_eq!(source.image().to_png(), recovered.encoded_bytes());
    }

    #[gpui::test]
    async fn at_graphite_unary_png_cap_recovery_and_scene_required_preserve_committed_tuple(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_image(cx);
        for operation in [None, Some(CpuImageEdit::Invert)] {
            ready
                .owner
                .update(cx, |owner, cx| owner.set_operation(operation, cx));
            cx.run_until_parked();
            let result = ready
                .owner
                .read_with(cx, |owner, _| owner.result().cloned());
            let source = ready.owner.read_with(cx, |owner, _| {
                owner
                    .resource_state()
                    .committed()
                    .expect("current source")
                    .clone()
            });
            let revision = ready
                .owner
                .read_with(cx, |owner, _| owner.resource_state().revision());
            let prepared = ready
                .owner
                .update(cx, |owner, cx| owner.prepare_png(4096, cx))
                .expect("uncapped original output")
                .await
                .expect("actual original encoder");
            for cap in [0, prepared.encoded_bytes().len() - 1] {
                let error = ready
                    .owner
                    .update(cx, |owner, cx| owner.prepare_png(cap, cx))
                    .expect("cap checked after encoding")
                    .await
                    .expect_err("output cap rejects");
                assert_eq!(
                    error,
                    graphics_core::export::PngExportError::EncodedSizeExceeded {
                        actual: prepared.encoded_bytes().len(),
                        limit: cap
                    }
                    .to_string()
                );
                ready.owner.read_with(cx, |owner, _| {
                    assert_eq!(owner.operation(), operation);
                    assert_eq!(owner.resource_state().revision(), revision);
                    assert!(Arc::ptr_eq(
                        owner.resource_state().committed().expect("same source"),
                        &source
                    ));
                    if let Some(result) = &result {
                        assert!(Arc::ptr_eq(owner.result().expect("same result"), result));
                    }
                });
            }
            let recovered = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_png(prepared.encoded_bytes().len(), cx)
                })
                .expect("exact cap retry")
                .await
                .expect("recovered output");
            assert_eq!(recovered.encoded_bytes(), prepared.encoded_bytes());
        }
        let mut transformed = load(&ready, "test-supplied-unary-transformed");
        transformed.transform = [2., 0., 0., 1., 4., 5.];
        ready
            .owner
            .update(cx, |owner, cx| owner.load_encoded_png(transformed, cx))
            .expect("actual transformed source admission");
        cx.run_until_parked();
        for operation in [None, Some(CpuImageEdit::Invert)] {
            ready
                .owner
                .update(cx, |owner, cx| owner.set_operation(operation, cx));
            cx.run_until_parked();
            let error = ready
                .owner
                .update(cx, |owner, cx| owner.prepare_png(4096, cx))
                .expect("fresh transformed selection")
                .await
                .expect_err("scene rendering required");
            assert_eq!(
                error,
                graphics_core::export::PngExportError::SceneRenderRequired.to_string()
            );
            ready.owner.read_with(cx, |owner, _| {
                assert_eq!(owner.operation(), operation);
                assert!(owner.resource_state().is_fresh());
                assert_eq!(
                    owner
                        .resource_state()
                        .committed()
                        .expect("source retained")
                        .descriptor()
                        .transform,
                    [2., 0., 0., 1., 4., 5.]
                );
                if operation.is_some() {
                    assert!(owner.result_is_fresh());
                }
            });
        }
    }

    fn scene_configuration() -> graphics_core::render::CpuSvgSceneConfiguration {
        graphics_core::render::CpuSvgSceneConfiguration {
            viewport: graphite_core_types::transform::Footprint {
                transform: graphite_glam::DAffine2::IDENTITY,
                resolution: graphite_glam::UVec2::new(16, 8),
                quality: graphite_core_types::transform::Footprint::DEFAULT.quality,
            },
            scale: 1.,
            max_svg_bytes: 16 * 1024,
        }
    }

    fn ready_scene_result(cx: &mut TestAppContext) -> ReadyImage {
        let ready = ready_image(cx);
        let mut transformed = load(&ready, "test-supplied-scene-parent");
        transformed.transform = [16., 0., 0., 8., 0., 0.];
        ready
            .owner
            .update(cx, |owner, cx| owner.load_encoded_png(transformed, cx))
            .expect("real authenticated transformed source");
        cx.run_until_parked();
        ready.owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        cx.run_until_parked();
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.result_is_fresh())
        );
        ready
    }

    #[gpui::test]
    async fn at_r20_2_scene_session_original_conversion_retains_authenticated_parent(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        let result = ready.owner.read_with(cx, |owner, _| {
            owner.result().expect("original result").clone()
        });
        let before_pixels = result.pixels().image().clone();
        let before_attributes = result.pixels().attributes().clone();
        let prepared = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 1., 2048, cx)
            })
            .expect("fresh selected scene request")
            .await
            .expect("real SVG conversion");
        ready.owner.read_with(cx, |owner, _| {
            assert!(owner.scene_preview_is_current(&prepared));
            assert!(Arc::ptr_eq(
                owner.result().expect("same edit result"),
                &result
            ));
            assert!(Arc::ptr_eq(
                owner
                    .resource_state()
                    .committed()
                    .expect("same authenticated source"),
                &prepared.source
            ));
            assert_eq!(owner.resource_state().revision(), prepared.source_revision);
        });
        assert_eq!(
            (
                prepared.image.size(0).width.0,
                prepared.image.size(0).height.0
            ),
            (32, 16)
        );
        let renderer = cx.update(|cx| cx.svg_renderer());
        let original = renderer
            .render_single_frame(prepared.scene.svg().svg.as_bytes(), 1.)
            .expect("existing native SVG renderer oracle");
        assert_eq!(prepared.image.as_bytes(0), original.as_bytes(0));
        assert_eq!(result.pixels().image(), &before_pixels);
        assert_eq!(result.pixels().attributes(), &before_attributes);
        assert_eq!(
            prepared
                .result
                .as_ref()
                .expect("edited scene receipt")
                .input_identity(),
            prepared.source.identity()
        );
        assert_eq!(
            prepared
                .result
                .as_ref()
                .expect("edited scene receipt")
                .descriptor()
                .transform
                .map(f64::to_bits),
            [16., 0., 0., 8., 0., 0.].map(f64::to_bits)
        );
    }

    #[gpui::test]
    async fn at_r20_2_scene_session_supersession_cancel_and_stale_consumption(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        let first = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 1., 2048, cx)
            })
            .expect("first scene request");
        let second = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 0.5, 512, cx)
            })
            .expect("configuration successor");
        assert!(first.await.is_err());
        let completed = second.await.expect("current configuration conversion");
        assert_eq!(
            (
                completed.image.size(0).width.0,
                completed.image.size(0).height.0
            ),
            (16, 8)
        );
        ready.owner.update(cx, |owner, cx| owner.cancel(cx));
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&completed))
        );
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 1., 2048, cx)
            })
            .expect("new pending scene");
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&completed))
        );
        ready.owner.update(cx, |owner, cx| owner.cancel(cx));
        assert!(pending.await.is_err());
        assert_eq!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.diagnostic().cloned()),
            Some(ImageSessionDiagnostic::Resource(
                crate::resources::ImageResourceError::Cancelled
            ))
        );
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 1., 2048, cx)
            })
            .expect("explicit retry");
        ready.owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::MakeOpaque), cx)
        });
        assert!(pending.await.is_err());
        cx.run_until_parked();
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 1., 2048, cx)
            })
            .expect("new operation scene");
        ready
            .owner
            .update(cx, |owner, cx| {
                owner.load_encoded_png(load(&ready, "test-supplied-scene-successor"), cx)
            })
            .expect("actual authenticated load successor");
        assert!(pending.await.is_err());
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&completed))
        );
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn at_r20_2_scene_session_cap_recovery_revocation_and_release(cx: &mut TestAppContext) {
        let ready = ready_scene_result(cx);
        let result = ready.owner.read_with(cx, |owner, _| {
            owner.result().expect("committed edit").clone()
        });
        let error = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 1., 2047, cx)
            })
            .expect("admitted job")
            .await
            .err()
            .expect("actual preview cap rejection");
        assert!(error.contains("byte limit"));
        ready.owner.read_with(cx, |owner, _| {
            assert!(owner.result_is_fresh());
            assert!(Arc::ptr_eq(
                owner.result().expect("input/edit preserved"),
                &result
            ));
        });
        let completed = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 1., 2048, cx)
            })
            .expect("exact-cap recovery")
            .await
            .expect("current scene preview");
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 1., 2048, cx)
            })
            .expect("pre-revocation conversion");
        ready
            .owner
            .update(cx, |owner, cx| owner.revoke(result.input_identity(), cx))
            .expect("actual resource revocation");
        assert!(pending.await.is_err());
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&completed))
        );
        assert!(
            ready
                .owner
                .update(cx, |owner, cx| owner.prepare_scene_preview(
                    scene_configuration(),
                    1.,
                    2048,
                    cx
                ))
                .is_err()
        );
        let another = ready_scene_result(cx);
        let pending = another
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(scene_configuration(), 1., 2048, cx)
            })
            .expect("pre-release conversion");
        let weak = another.owner.downgrade();
        drop(another.owner);
        cx.run_until_parked();
        assert!(weak.upgrade().is_none());
        assert!(pending.await.is_err());
    }

    fn exact_scene_configuration(
        width: u32,
        height: u32,
        scale: f64,
    ) -> graphics_core::render::CpuSvgSceneConfiguration {
        let mut configuration = scene_configuration();
        configuration.viewport.resolution = graphite_glam::UVec2::new(width, height);
        configuration.scale = scale;
        configuration
    }

    fn assert_exact_scene_frame(
        preview: &crate::session::PreparedImageScenePreview,
        width: u32,
        height: u32,
    ) {
        assert_eq!(preview.image.frame_count(), 1);
        assert_eq!(preview.image.size(0).width.0, width as i32);
        assert_eq!(preview.image.size(0).height.0, height as i32);
        let bytes = preview.image.as_bytes(0).expect("single native frame");
        assert_eq!(bytes.len(), width as usize * height as usize * 4);
        assert!(bytes.as_chunks::<4>().0.iter().any(|pixel| pixel[3] > 0));
    }

    #[gpui::test]
    async fn at_r20_2_exact_target_session_original_and_edit_keep_physical_size_and_provenance(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        let source = ready.owner.read_with(cx, |owner, _| {
            owner
                .resource_state()
                .committed()
                .expect("authenticated source")
                .clone()
        });
        let source_pixels = source.image().clone();
        let source_descriptor = *source.descriptor();
        let source_identity = source.identity().clone();
        let source_origin = source.origin();
        let revision = ready
            .owner
            .read_with(cx, |owner, _| owner.resource_state().revision());
        let renderer = cx.update(|cx| cx.svg_renderer());
        for operation in [None, Some(CpuImageEdit::Invert)] {
            ready
                .owner
                .update(cx, |owner, cx| owner.set_operation(operation, cx));
            cx.run_until_parked();
            let result = ready
                .owner
                .read_with(cx, |owner, _| owner.result().cloned());
            let result_pixels = result
                .as_ref()
                .map(|result| result.pixels().image().clone());
            let result_attributes = result
                .as_ref()
                .map(|result| result.pixels().attributes().clone());
            for scale in [1., 1.5, 2.] {
                let configuration = exact_scene_configuration(35, 21, scale);
                let preview = ready
                    .owner
                    .update(cx, |owner, cx| {
                        owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
                    })
                    .expect("fresh exact target request")
                    .await
                    .expect("actual original scene conversion");
                assert_exact_scene_frame(&preview, 35, 21);
                assert_eq!(
                    preview.scene.configuration().scale.to_bits(),
                    scale.to_bits()
                );
                assert_eq!(
                    preview.scene.configuration().viewport.resolution,
                    configuration.viewport.resolution
                );
                assert_eq!(
                    preview
                        .scene
                        .configuration()
                        .viewport
                        .transform
                        .to_cols_array()
                        .map(f64::to_bits),
                    configuration
                        .viewport
                        .transform
                        .to_cols_array()
                        .map(f64::to_bits)
                );
                assert_eq!(preview.source_revision, revision);
                assert!(Arc::ptr_eq(&preview.source, &source));
                match (&preview.result, &result) {
                    (None, None) => assert_eq!(operation, None),
                    (Some(retained), Some(result)) => {
                        assert!(Arc::ptr_eq(retained, result));
                        assert_eq!(retained.operation(), CpuImageEdit::Invert);
                        assert_eq!(retained.input_identity(), &source_identity);
                        assert_eq!(retained.input_origin(), source_origin);
                        assert_eq!(
                            retained.pixels().image(),
                            result_pixels.as_ref().expect("original result pixels")
                        );
                        assert_eq!(
                            retained.pixels().attributes(),
                            result_attributes
                                .as_ref()
                                .expect("original result attributes")
                        );
                    }
                    _ => panic!("selection changed the source/result receipt role"),
                }
                ready.owner.read_with(cx, |owner, _| {
                    assert!(owner.scene_preview_is_current(&preview));
                    assert_eq!(owner.operation(), operation);
                    assert!(Arc::ptr_eq(
                        owner.resource_state().committed().expect("same source"),
                        &source
                    ));
                });
                let x = (1.5 * scale) as usize;
                let y = (2. * scale) as usize;
                let offset = (y * 35 + x) * 4;
                let pixel = preview
                    .image
                    .as_bytes(0)
                    .expect("actual frame")
                    .get(offset..offset + 4)
                    .expect("source-derived first red region");
                assert!(pixel[3] > 0);
                if operation.is_none() {
                    assert!(
                        pixel[2] > pixel[0] && pixel[2] > pixel[1],
                        "original first producer pixel remains red: {pixel:?}"
                    );
                } else {
                    assert!(
                        pixel[0] > pixel[2] && pixel[1] > pixel[2],
                        "original inversion of the first red pixel is cyan: {pixel:?}"
                    );
                }
                let oracle = crate::render::prepare_cpu_svg_preview_at_target(
                    &preview.scene,
                    &renderer,
                    35 * 21 * 4,
                )
                .expect("existing exact converter binding");
                assert_eq!(preview.image.as_bytes(0), oracle.as_bytes(0));
            }
            let configuration = exact_scene_configuration(35, 21, 1.);
            let legacy = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview(configuration, 1., 70 * 42 * 4, cx)
                })
                .expect("separate unchanged legacy request")
                .await
                .expect("original legacy supersampling");
            assert_exact_scene_frame(&legacy, 70, 42);
            assert!(
                ready
                    .owner
                    .read_with(cx, |owner, _| owner.scene_preview_is_current(&legacy))
            );
        }
        assert_eq!(source.image(), &source_pixels);
        assert_eq!(*source.descriptor(), source_descriptor);
        assert_eq!(
            source.descriptor().transform.map(f64::to_bits),
            source_descriptor.transform.map(f64::to_bits)
        );
        assert_eq!(source.identity(), &source_identity);
        assert_eq!(source.origin(), source_origin);
        assert_eq!(
            ready.png.encoded,
            ready
                .fixture
                .assets
                .lock()
                .expect("canonical asset service")
                .read_verified(
                    &ready.png.asset_identity,
                    &ready.fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024,
                )
                .expect("indexed producer remains unchanged")
        );
    }

    #[gpui::test]
    async fn at_r20_2_exact_target_session_configuration_successors_invalidate_real_receipts(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        let first_configuration = exact_scene_configuration(35, 21, 1.);
        let first = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(first_configuration, 35 * 21 * 4, cx)
            })
            .expect("first actual pending job");
        let mut next_configuration = exact_scene_configuration(37, 23, 1.5);
        next_configuration.viewport.transform =
            graphite_glam::DAffine2::from_translation(graphite_glam::DVec2::new(1., 2.));
        let second = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(next_configuration, 37 * 23 * 4, cx)
            })
            .expect("changed exact canvas and transform");
        assert!(first.await.is_err());
        let completed = second.await.expect("current successor");
        assert_exact_scene_frame(&completed, 37, 23);
        assert_eq!(
            completed
                .scene
                .configuration()
                .viewport
                .transform
                .to_cols_array()
                .map(f64::to_bits),
            next_configuration
                .viewport
                .transform
                .to_cols_array()
                .map(f64::to_bits)
        );
        assert_eq!(
            completed.scene.configuration().scale.to_bits(),
            1.5_f64.to_bits()
        );
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&completed))
        );
        let successor_configuration = exact_scene_configuration(39, 25, 2.);
        let successor = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(successor_configuration, 39 * 25 * 4, cx)
            })
            .expect("completed receipt successor");
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&completed))
        );
        let current = successor.await.expect("third target");
        assert_exact_scene_frame(&current, 39, 25);
        assert!(current.scene_generation > completed.scene_generation);
        assert!(Arc::ptr_eq(&current.source, &completed.source));
        assert!(Arc::ptr_eq(
            current.result.as_ref().expect("same selected edit"),
            completed.result.as_ref().expect("retained edit")
        ));
        assert_eq!(current.source_revision, completed.source_revision);
        assert_eq!(
            current.scene.configuration().viewport.resolution,
            successor_configuration.viewport.resolution
        );
        assert_eq!(
            current.scene.configuration().scale.to_bits(),
            2_f64.to_bits()
        );
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&completed))
        );
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&current))
        );
        let repeated = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(successor_configuration, 39 * 25 * 4, cx)
            })
            .expect("explicit same target new generation")
            .await
            .expect("repeated exact scene");
        assert_eq!(current.image.as_bytes(0), repeated.image.as_bytes(0));
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&current))
        );
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&repeated))
        );
    }

    #[gpui::test]
    async fn at_r20_2_exact_target_session_cap_and_invalid_configuration_recover_atomically(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        let source = ready.owner.read_with(cx, |owner, _| {
            owner.resource_state().committed().expect("source").clone()
        });
        let result = ready
            .owner
            .read_with(cx, |owner, _| owner.result().expect("edit").clone());
        let source_pixels = source.image().clone();
        let result_pixels = result.pixels().image().clone();
        let result_attributes = result.pixels().attributes().clone();
        let revision = ready
            .owner
            .read_with(cx, |owner, _| owner.resource_state().revision());
        let configuration = exact_scene_configuration(35, 21, 1.5);
        let required = 35 * 21 * 4;
        let completed = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("fresh target")
            .await
            .expect("exact cap initial output");
        assert_exact_scene_frame(&completed, 35, 21);
        for maximum in [0, required - 1] {
            let error = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview_at_target(configuration, maximum, cx)
                })
                .expect("bounded background job")
                .await
                .err()
                .expect("cap rejects output");
            assert_eq!(
                error,
                crate::render::CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum }
                    .to_string()
            );
            assert!(
                !ready
                    .owner
                    .read_with(cx, |owner, _| owner.scene_preview_is_current(&completed))
            );
            ready.owner.read_with(cx, |owner, _| {
                assert!(owner.resource_state().is_fresh());
                assert!(owner.result_is_fresh());
                assert_eq!(owner.resource_state().revision(), revision);
                assert_eq!(owner.operation(), Some(CpuImageEdit::Invert));
                assert!(Arc::ptr_eq(
                    owner.resource_state().committed().expect("same source"),
                    &source
                ));
                assert!(Arc::ptr_eq(owner.result().expect("same result"), &result));
            });
        }
        for (invalid, expected) in [
            (
                exact_scene_configuration(0, 21, 1.),
                graphics_core::render::CpuSvgSceneError::EmptyViewport.to_string(),
            ),
            (
                exact_scene_configuration(35, 21, -0.),
                graphics_core::render::CpuSvgSceneError::InvalidScale.to_string(),
            ),
            (
                exact_scene_configuration(8193, 21, 1.),
                crate::render::CpuSvgPreviewError::RendererSizeLimit.to_string(),
            ),
        ] {
            let error = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview_at_target(invalid, usize::MAX, cx)
                })
                .expect("configuration validated by original scene/converter job")
                .await
                .err()
                .expect("invalid target");
            assert_eq!(error, expected);
            assert!(
                ready
                    .owner
                    .read_with(cx, |owner, _| owner.result_is_fresh())
            );
            assert!(Arc::ptr_eq(
                &source,
                &ready.owner.read_with(cx, |owner, _| owner
                    .resource_state()
                    .committed()
                    .expect("retained source")
                    .clone())
            ));
            assert!(Arc::ptr_eq(
                &result,
                &ready.owner.read_with(cx, |owner, _| owner
                    .result()
                    .expect("retained edit")
                    .clone())
            ));
        }
        let recovered = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("valid exact cap retry")
            .await
            .expect("current output recovery");
        assert_exact_scene_frame(&recovered, 35, 21);
        assert_eq!(recovered.image.as_bytes(0), completed.image.as_bytes(0));
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&recovered))
        );
        assert_eq!(source.image(), &source_pixels);
        assert_eq!(result.pixels().image(), &result_pixels);
        assert_eq!(result.pixels().attributes(), &result_attributes);
        assert_eq!(result.input_identity(), source.identity());
        assert_eq!(result.input_origin(), source.origin());
    }

    #[gpui::test]
    async fn at_r20_2_exact_target_session_cancel_operation_load_revoke_and_release_never_publish(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        let configuration = exact_scene_configuration(35, 21, 1.5);
        let required = 35 * 21 * 4;
        let completed = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("current edit scene")
            .await
            .expect("real visible frame");
        assert_exact_scene_frame(&completed, 35, 21);
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("pending conversion");
        ready.owner.update(cx, |owner, cx| owner.cancel(cx));
        assert!(pending.await.is_err());
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&completed))
        );
        assert_eq!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.diagnostic().cloned()),
            Some(ImageSessionDiagnostic::Resource(
                crate::resources::ImageResourceError::Cancelled
            ))
        );
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("retry before operation transition");
        ready
            .owner
            .update(cx, |owner, cx| owner.set_operation(None, cx));
        assert!(pending.await.is_err());
        cx.run_until_parked();
        let original = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("explicit Original selection")
            .await
            .expect("original recovery");
        assert!(original.result.is_none());
        assert_exact_scene_frame(&original, 35, 21);
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("before source successor");
        let mut successor = load(&ready, "test-supplied-exact-scene-successor");
        successor.transform = [16., 0., 0., 8., 0., 0.];
        ready
            .owner
            .update(cx, |owner, cx| owner.load_encoded_png(successor, cx))
            .expect("authenticated new caller-supplied version");
        assert!(pending.await.is_err());
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&original))
        );
        cx.run_until_parked();
        let current_source = ready.owner.read_with(cx, |owner, _| {
            owner
                .resource_state()
                .committed()
                .expect("new committed source")
                .clone()
        });
        assert_ne!(
            current_source.identity().version_id,
            original.source.identity().version_id
        );
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("before actual host revocation");
        ready
            .owner
            .update(cx, |owner, cx| owner.revoke(current_source.identity(), cx))
            .expect("exact committed source revoke");
        assert!(pending.await.is_err());
        assert!(
            ready
                .owner
                .update(cx, |owner, cx| owner.prepare_scene_preview_at_target(
                    configuration,
                    required,
                    cx
                ))
                .is_err()
        );
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.resource_state().is_revoked())
        );
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&original))
        );
        let released = ready_scene_result(cx);
        let pending = released
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("before session release");
        let weak = released.owner.downgrade();
        drop(released.owner);
        cx.run_until_parked();
        assert!(weak.upgrade().is_none());
        assert!(pending.await.is_err());
    }

    fn scene_png_rgba_rows(image: &gpui::RenderImage) -> Vec<u8> {
        let mut pixels = image
            .as_bytes(0)
            .expect("one captured native frame")
            .to_vec();
        assert_eq!(pixels.len() % 4, 0);
        for pixel in pixels.as_chunks_mut::<4>().0.iter_mut() {
            pixel.swap(0, 2);
        }
        pixels
    }

    #[gpui::test]
    async fn at_r22_1_exact_scene_png_original_and_edit_preserve_frame_target_and_provenance(
        cx: &mut TestAppContext,
    ) {
        use sha2::{Digest, Sha256};
        let ready = ready_scene_result(cx);
        let source = ready.owner.read_with(cx, |owner, _| {
            owner.resource_state().committed().expect("source").clone()
        });
        let pixels_before = source.image().clone();
        let identity_before = source.identity().clone();
        let origin_before = source.origin();
        let descriptor_before = *source.descriptor();
        for operation in [None, Some(CpuImageEdit::Invert)] {
            ready
                .owner
                .update(cx, |owner, cx| owner.set_operation(operation, cx));
            cx.run_until_parked();
            let selected_result = ready
                .owner
                .read_with(cx, |owner, _| owner.result().cloned());
            let result_pixels = selected_result
                .as_ref()
                .map(|result| result.pixels().image().clone());
            let result_attributes = selected_result
                .as_ref()
                .map(|result| result.pixels().attributes().clone());
            let result_digest = selected_result
                .as_ref()
                .map(|result| result.pixels_sha256());
            for scale in [1., 1.5, 2.] {
                let configuration = exact_scene_configuration(35, 21, scale);
                let preview = ready
                    .owner
                    .update(cx, |owner, cx| {
                        owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
                    })
                    .expect("fresh declared target")
                    .await
                    .expect("actual exact scene");
                assert_exact_scene_frame(&preview, 35, 21);
                let frame = preview.image.clone();
                let frame_before = frame.as_bytes(0).expect("captured native pixels").to_vec();
                let scene = preview.scene.clone();
                let expected_rgba = scene_png_rgba_rows(&frame);
                let source_revision = preview.source_revision;
                let receipt = ready
                    .owner
                    .update(cx, |owner, cx| {
                        owner.prepare_scene_png(preview, 35 * 21 * 4, 16 * 1024, cx)
                    })
                    .expect("current exact frame PNG")
                    .await
                    .expect("shared actual PNG codec");
                let png = receipt.png();
                assert_eq!((png.width(), png.height()), (35, 21));
                assert_eq!(png.content_type(), "image/png");
                assert_eq!(png.sample_format(), CpuSampleFormat::Rgba8);
                assert_eq!(png.color_space(), ImageColorSpace::Srgb);
                assert_eq!(png.alpha(), AlphaAssociation::Straight);
                assert_eq!(
                    png.encoded_bytes().get(..8),
                    Some(&[137, 80, 78, 71, 13, 10, 26, 10][..])
                );
                let decoded = image::load_from_memory_with_format(
                    png.encoded_bytes(),
                    image::ImageFormat::Png,
                )
                .expect("actual PNG decoder")
                .to_rgba8();
                assert_eq!(decoded.dimensions(), (35, 21));
                assert_eq!(decoded.as_raw(), &expected_rgba);
                assert_eq!(
                    receipt.frame_pixels_sha256(),
                    <[u8; 32]>::from(Sha256::digest(&expected_rgba))
                );
                assert_eq!(
                    png.content_sha256(),
                    <[u8; 32]>::from(Sha256::digest(png.encoded_bytes()))
                );
                assert!(
                    decoded
                        .pixels()
                        .any(|pixel| pixel.0[3] > 0 && pixel.0[3] < 255)
                );
                let x = (1.5 * scale) as u32;
                let y = (2. * scale) as u32;
                let first_pixel = decoded.get_pixel(x, y).0;
                assert!(first_pixel[3] > 0);
                if operation.is_none() {
                    assert!(
                        first_pixel[0] > first_pixel[1] && first_pixel[0] > first_pixel[2],
                        "original red producer region: {first_pixel:?}"
                    );
                } else {
                    assert!(
                        first_pixel[1] > first_pixel[0] && first_pixel[2] > first_pixel[0],
                        "original Invert cyan region: {first_pixel:?}"
                    );
                }
                assert!(std::ptr::eq(receipt.scene(), scene.as_ref()));
                assert!(std::ptr::eq(receipt.source(), source.as_ref()));
                assert_eq!(receipt.source_revision(), source_revision);
                assert_eq!(
                    receipt.scene().configuration().viewport.resolution,
                    configuration.viewport.resolution
                );
                assert_eq!(
                    receipt
                        .scene()
                        .configuration()
                        .viewport
                        .transform
                        .to_cols_array()
                        .map(f64::to_bits),
                    configuration
                        .viewport
                        .transform
                        .to_cols_array()
                        .map(f64::to_bits)
                );
                assert_eq!(
                    receipt.scene().configuration().scale.to_bits(),
                    scale.to_bits()
                );
                assert_eq!(
                    receipt.scene().configuration().max_svg_bytes,
                    configuration.max_svg_bytes
                );
                assert_eq!(receipt.source().identity(), &identity_before);
                assert_eq!(receipt.source().origin(), origin_before);
                assert_eq!(
                    receipt.source().descriptor().transform.map(f64::to_bits),
                    descriptor_before.transform.map(f64::to_bits)
                );
                match (receipt.result(), selected_result.as_ref()) {
                    (None, None) => assert_eq!(operation, None),
                    (Some(retained), Some(result)) => {
                        assert!(std::ptr::eq(retained, result.as_ref()));
                        assert_eq!(retained.operation(), CpuImageEdit::Invert);
                        assert_eq!(retained.input_identity(), &identity_before);
                        assert_eq!(retained.input_origin(), origin_before);
                        assert_eq!(Some(retained.pixels_sha256()), result_digest);
                        assert_eq!(
                            retained.pixels().image(),
                            result_pixels.as_ref().expect("original edited pixels")
                        );
                        assert_eq!(
                            retained.pixels().attributes(),
                            result_attributes.as_ref().expect("original attributes")
                        );
                    }
                    _ => panic!("source and edited result roles changed during encoding"),
                }
                assert!(
                    ready
                        .owner
                        .read_with(cx, |owner, _| owner.scene_png_is_current(&receipt))
                );
                assert_eq!(frame.as_bytes(0).expect("captured frame"), frame_before);
            }
            let error = match ready
                .owner
                .update(cx, |owner, cx| owner.prepare_png(16 * 1024, cx))
                .expect("unchanged pixel-only route")
                .await
            {
                Err(error) => error,
                Ok(_) => panic!("transformed source still needs a scene"),
            };
            assert_eq!(
                error,
                graphics_core::export::PngExportError::SceneRenderRequired.to_string()
            );
        }
        assert_eq!(source.image(), &pixels_before);
        assert_eq!(*source.descriptor(), descriptor_before);
        assert_eq!(source.identity(), &identity_before);
        assert_eq!(source.origin(), origin_before);
        assert_eq!(
            ready
                .fixture
                .assets
                .lock()
                .expect("canonical service")
                .read_verified(
                    &ready.png.asset_identity,
                    &ready.fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("canonical retained indexed producer"),
            ready.png.encoded
        );
    }

    #[gpui::test]
    async fn at_r22_1_exact_scene_png_pixel_and_encoded_caps_recover_without_mutating_source(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        let configuration = exact_scene_configuration(35, 21, 1.5);
        let required = 35 * 21 * 4;
        let source = ready.owner.read_with(cx, |owner, _| {
            owner.resource_state().committed().expect("source").clone()
        });
        let result = ready.owner.read_with(cx, |owner, _| {
            owner.result().expect("selected edit").clone()
        });
        let source_pixels = source.image().clone();
        let result_pixels = result.pixels().image().clone();
        let result_attributes = result.pixels().attributes().clone();
        let revision = ready
            .owner
            .read_with(cx, |owner, _| owner.resource_state().revision());
        let preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("fresh scene")
            .await
            .expect("exact target frame");
        let seed = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, required, 16 * 1024, cx)
            })
            .expect("initial PNG")
            .await
            .expect("actual codec");
        let seed_bytes = seed.png().encoded_bytes().to_vec();
        let encoded_length = seed_bytes.len();
        for (pixel_cap, encoded_cap) in [
            (0, encoded_length),
            (required - 1, encoded_length),
            (required, 0),
            (required, encoded_length - 1),
        ] {
            let preview = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview_at_target(configuration, required, cx)
                })
                .expect("new same-target receipt")
                .await
                .expect("exact scene unaffected by encode budget");
            let frame = preview.image.clone();
            let before = frame.as_bytes(0).expect("frame bytes").to_vec();
            let error = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_png(preview, pixel_cap, encoded_cap, cx)
                })
                .expect("budget checked by background codec seam")
                .await
                .err()
                .expect("cap failure");
            if pixel_cap < required {
                assert_eq!(
                    error,
                    format!(
                        "PNG rows require {required} bytes, exceeding the {pixel_cap} byte limit"
                    )
                );
            } else {
                assert_eq!(
                    error,
                    graphics_core::export::PngExportError::EncodedSizeExceeded {
                        actual: encoded_length,
                        limit: encoded_cap
                    }
                    .to_string()
                );
            }
            assert_eq!(frame.as_bytes(0).expect("last valid exact frame"), before);
            assert_eq!(seed.png().encoded_bytes(), seed_bytes);
            assert!(
                !ready
                    .owner
                    .read_with(cx, |owner, _| owner.scene_png_is_current(&seed))
            );
            ready.owner.read_with(cx, |owner, _| {
                assert!(owner.resource_state().is_fresh());
                assert!(owner.result_is_fresh());
                assert!(owner.diagnostic().is_none());
                assert_eq!(owner.resource_state().revision(), revision);
                assert_eq!(owner.operation(), Some(CpuImageEdit::Invert));
                assert!(Arc::ptr_eq(
                    owner.resource_state().committed().expect("same source"),
                    &source
                ));
                assert!(Arc::ptr_eq(owner.result().expect("same result"), &result));
            });
        }
        let preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("exact cap retry scene")
            .await
            .expect("scene recovery");
        let recovered = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, required, encoded_length, cx)
            })
            .expect("exact limits")
            .await
            .expect("PNG recovery");
        assert_eq!(recovered.png().encoded_bytes(), seed_bytes);
        assert_eq!(recovered.frame_pixels_sha256(), seed.frame_pixels_sha256());
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&recovered))
        );
        assert_eq!(source.image(), &source_pixels);
        assert_eq!(result.pixels().image(), &result_pixels);
        assert_eq!(result.pixels().attributes(), &result_attributes);
        assert_eq!(recovered.source().identity(), source.identity());
        assert_eq!(
            recovered.result().expect("edited receipt").pixels_sha256(),
            result.pixels_sha256()
        );
    }

    #[gpui::test]
    async fn at_r22_1_exact_scene_png_target_successors_and_legacy_receipts_never_publish(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        let configuration = exact_scene_configuration(35, 21, 1.);
        let preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
            })
            .expect("initial target")
            .await
            .expect("exact scene");
        let completed = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, 35 * 21 * 4, 16 * 1024, cx)
            })
            .expect("exact PNG")
            .await
            .expect("initial current receipt");
        let completed_bytes = completed.png().encoded_bytes().to_vec();
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&completed))
        );
        let preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
            })
            .expect("next scene generation")
            .await
            .expect("new receipt");
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&completed))
        );
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, 35 * 21 * 4, 16 * 1024, cx)
            })
            .expect("pending old target encode");
        let mut successor = exact_scene_configuration(37, 23, 2.);
        successor.viewport.transform =
            graphite_glam::DAffine2::from_translation(graphite_glam::DVec2::new(1., 2.));
        let current = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(successor, 37 * 23 * 4, cx)
            })
            .expect("new exact target config");
        assert!(pending.await.is_err());
        let current = current.await.expect("current successor frame");
        assert_exact_scene_frame(&current, 37, 23);
        let receipt = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(current, 37 * 23 * 4, 16 * 1024, cx)
            })
            .expect("new target PNG")
            .await
            .expect("new current receipt");
        assert_eq!((receipt.png().width(), receipt.png().height()), (37, 23));
        assert_eq!(
            receipt.scene().configuration().scale.to_bits(),
            2_f64.to_bits()
        );
        assert_eq!(
            receipt
                .scene()
                .configuration()
                .viewport
                .transform
                .to_cols_array()
                .map(f64::to_bits),
            successor
                .viewport
                .transform
                .to_cols_array()
                .map(f64::to_bits)
        );
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&receipt))
        );
        assert_eq!(completed.png().encoded_bytes(), completed_bytes);
        let legacy = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(configuration, 1., 70 * 42 * 4, cx)
            })
            .expect("unchanged legacy request")
            .await
            .expect("legacy supersampled frame");
        assert_exact_scene_frame(&legacy, 70, 42);
        assert!(
            ready
                .owner
                .update(cx, |owner, cx| owner.prepare_scene_png(
                    legacy,
                    70 * 42 * 4,
                    16 * 1024,
                    cx
                ))
                .is_err()
        );
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&receipt))
        );
        let stale = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
            })
            .expect("completed exact receipt")
            .await
            .expect("exact scene");
        let fresh = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(successor, 37 * 23 * 4, cx)
            })
            .expect("source unchanged target successor")
            .await
            .expect("fresh target");
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_preview_is_current(&stale))
        );
        assert!(
            ready
                .owner
                .update(cx, |owner, cx| owner.prepare_scene_png(
                    stale,
                    35 * 21 * 4,
                    16 * 1024,
                    cx
                ))
                .is_err()
        );
        let recovery = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(fresh, 37 * 23 * 4, 16 * 1024, cx)
            })
            .expect("fresh exact target recovery")
            .await
            .expect("current PNG");
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&recovery))
        );
    }

    #[gpui::test]
    async fn at_r22_1_exact_scene_png_cancel_edit_source_revoke_and_release_reject_pending_work(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        let configuration = exact_scene_configuration(35, 21, 1.5);
        let required = 35 * 21 * 4;
        let preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("fresh scene")
            .await
            .expect("exact frame");
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, required, 16 * 1024, cx)
            })
            .expect("before canonical cancellation");
        ready.owner.update(cx, |owner, cx| owner.cancel(cx));
        assert!(pending.await.is_err());
        let preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("retry scene")
            .await
            .expect("edit remains valid");
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, required, 16 * 1024, cx)
            })
            .expect("before operation change");
        ready
            .owner
            .update(cx, |owner, cx| owner.set_operation(None, cx));
        assert!(pending.await.is_err());
        cx.run_until_parked();
        let preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("Original scene")
            .await
            .expect("source-only recovery");
        assert!(preview.result.is_none());
        let completed = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, required, 16 * 1024, cx)
            })
            .expect("Original PNG")
            .await
            .expect("current source receipt");
        let original_bytes = completed.png().encoded_bytes().to_vec();
        assert!(completed.result().is_none());
        let preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("before reload scene")
            .await
            .expect("scene");
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, required, 16 * 1024, cx)
            })
            .expect("before source reload");
        let mut successor = load(&ready, "test-supplied-scene-png-successor");
        successor.transform = [16., 0., 0., 8., 0., 0.];
        ready
            .owner
            .update(cx, |owner, cx| owner.load_encoded_png(successor, cx))
            .expect("genuine explicit successor request");
        assert!(pending.await.is_err());
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&completed))
        );
        cx.run_until_parked();
        let source = ready.owner.read_with(cx, |owner, _| {
            owner
                .resource_state()
                .committed()
                .expect("new source")
                .clone()
        });
        assert_ne!(
            source.identity().version_id,
            completed.source().identity().version_id
        );
        let preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("successor scene")
            .await
            .expect("fresh source");
        let pending = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, required, 16 * 1024, cx)
            })
            .expect("before revoke");
        ready
            .owner
            .update(cx, |owner, cx| owner.revoke(source.identity(), cx))
            .expect("actual resource revocation");
        assert!(pending.await.is_err());
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.resource_state().is_revoked())
        );
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&completed))
        );
        assert_eq!(completed.png().encoded_bytes(), original_bytes);
        let released = ready_scene_result(cx);
        let preview = released
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, required, cx)
            })
            .expect("release scene")
            .await
            .expect("real exact frame");
        let pending = released
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(preview, required, 16 * 1024, cx)
            })
            .expect("before session release");
        let weak = released.owner.downgrade();
        drop(released.owner);
        cx.run_until_parked();
        assert!(weak.upgrade().is_none());
        assert!(pending.await.is_err());
    }

    #[gpui::test]
    async fn at_r22_1_exact_scene_png_edit_roundtrip_changes_digest_and_restores_original(
        cx: &mut TestAppContext,
    ) {
        let ready = ready_scene_result(cx);
        ready
            .owner
            .update(cx, |owner, cx| owner.set_operation(None, cx));
        cx.run_until_parked();
        let source = ready.owner.read_with(cx, |owner, _| {
            owner
                .resource_state()
                .committed()
                .expect("authenticated editable source")
                .clone()
        });
        let source_pixels = source.image().clone();
        let source_identity = source.identity().clone();
        let source_origin = source.origin();
        let source_descriptor = *source.descriptor();
        let source_revision = ready
            .owner
            .read_with(cx, |owner, _| owner.resource_state().revision());
        let indexed_bytes = ready
            .fixture
            .assets
            .lock()
            .expect("canonical service")
            .read_verified(
                &ready.png.asset_identity,
                &ready.fixture.input_authorization,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("exact genuine indexed PNG");
        assert_eq!(indexed_bytes, ready.png.encoded);
        let configuration = exact_scene_configuration(37, 23, 1.5);
        let pixel_cap = 37 * 23 * 4;
        let original_preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, pixel_cap, cx)
            })
            .expect("explicit Original exact target")
            .await
            .expect("original source scene");
        assert_exact_scene_frame(&original_preview, 37, 23);
        assert!(original_preview.result.is_none());
        let original_rows = scene_png_rgba_rows(&original_preview.image);
        let original = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(original_preview, pixel_cap, 16 * 1024, cx)
            })
            .expect("Original PNG request")
            .await
            .expect("actual source PNG");
        let original_bytes = original.png().encoded_bytes().to_vec();
        let original_encoded_digest = original.png().content_sha256();
        let original_frame_digest = original.frame_pixels_sha256();
        let original_decoded = image::load_from_memory_with_format(
            original.png().encoded_bytes(),
            image::ImageFormat::Png,
        )
        .expect("original baked PNG decoder")
        .to_rgba8();
        assert_eq!(original_decoded.as_raw(), &original_rows);
        let red = original_decoded.get_pixel(2, 3).0;
        assert!(red[3] > 0 && red[0] > red[1] && red[0] > red[2]);
        assert!(original.result().is_none());
        assert!(std::ptr::eq(original.source(), source.as_ref()));
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&original))
        );

        ready.owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&original))
        );
        cx.run_until_parked();
        let result = ready.owner.read_with(cx, |owner, _| {
            assert!(owner.result_is_fresh());
            owner.result().expect("original Invert result").clone()
        });
        let edited_pixels = result.pixels().image().clone();
        let edited_attributes = result.pixels().attributes().clone();
        let edited_preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, pixel_cap, cx)
            })
            .expect("new exact scene for selected edit")
            .await
            .expect("original edited scene");
        assert_exact_scene_frame(&edited_preview, 37, 23);
        assert!(Arc::ptr_eq(
            edited_preview.result.as_ref().expect("edit receipt"),
            &result
        ));
        let edited_rows = scene_png_rgba_rows(&edited_preview.image);
        let edited = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(edited_preview, pixel_cap, 16 * 1024, cx)
            })
            .expect("new edited PNG")
            .await
            .expect("actual edited target PNG");
        let edited_decoded = image::load_from_memory_with_format(
            edited.png().encoded_bytes(),
            image::ImageFormat::Png,
        )
        .expect("edited PNG decoder")
        .to_rgba8();
        assert_eq!(edited_decoded.dimensions(), (37, 23));
        assert_eq!(edited_decoded.as_raw(), &edited_rows);
        let cyan = edited_decoded.get_pixel(2, 3).0;
        assert!(cyan[3] > 0 && cyan[1] > cyan[0] && cyan[2] > cyan[0]);
        assert_ne!(edited_rows, original_rows);
        assert_ne!(edited.png().encoded_bytes(), original_bytes);
        assert_ne!(edited.png().content_sha256(), original_encoded_digest);
        assert_ne!(edited.frame_pixels_sha256(), original_frame_digest);
        assert!(
            original_rows
                .as_chunks::<4>()
                .0
                .iter()
                .zip(edited_rows.as_chunks::<4>().0.iter())
                .all(|(source, result)| source[3] == result[3])
        );
        assert_eq!(edited.source_revision(), source_revision);
        assert!(std::ptr::eq(edited.source(), source.as_ref()));
        let retained = edited.result().expect("selected edit provenance");
        assert!(std::ptr::eq(retained, result.as_ref()));
        assert_eq!(retained.operation(), CpuImageEdit::Invert);
        assert_eq!(retained.input_identity(), &source_identity);
        assert_eq!(retained.input_origin(), source_origin);
        assert_eq!(retained.pixels_sha256(), result.pixels_sha256());
        assert_eq!(
            edited.scene().configuration().viewport.resolution,
            configuration.viewport.resolution
        );
        assert_eq!(
            edited.scene().configuration().scale.to_bits(),
            configuration.scale.to_bits()
        );
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&edited))
        );
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&original))
        );
        assert_eq!(original.png().encoded_bytes(), original_bytes);

        ready
            .owner
            .update(cx, |owner, cx| owner.set_operation(None, cx));
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&edited))
        );
        cx.run_until_parked();
        let restored_preview = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview_at_target(configuration, pixel_cap, cx)
            })
            .expect("restore Original selection")
            .await
            .expect("source-only exact scene recovery");
        assert!(restored_preview.result.is_none());
        assert_exact_scene_frame(&restored_preview, 37, 23);
        assert_eq!(scene_png_rgba_rows(&restored_preview.image), original_rows);
        let restored = ready
            .owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_png(restored_preview, pixel_cap, original_bytes.len(), cx)
            })
            .expect("exact encoded cap restoration")
            .await
            .expect("restored original PNG");
        assert_eq!(restored.png().encoded_bytes(), original_bytes);
        assert_eq!(restored.png().content_sha256(), original_encoded_digest);
        assert_eq!(restored.frame_pixels_sha256(), original_frame_digest);
        assert_eq!(restored.source_revision(), source_revision);
        assert!(restored.result().is_none());
        assert!(std::ptr::eq(restored.source(), source.as_ref()));
        assert!(
            ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&restored))
        );
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&original))
        );
        assert!(
            !ready
                .owner
                .read_with(cx, |owner, _| owner.scene_png_is_current(&edited))
        );
        ready.owner.read_with(cx, |owner, _| {
            assert_eq!(owner.operation(), None);
            assert!(owner.result().is_none());
            assert!(owner.resource_state().is_fresh());
            assert_eq!(owner.resource_state().revision(), source_revision);
            assert!(Arc::ptr_eq(
                owner
                    .resource_state()
                    .committed()
                    .expect("same editable input"),
                &source
            ));
        });
        assert_eq!(source.image(), &source_pixels);
        assert_eq!(*source.descriptor(), source_descriptor);
        assert_eq!(
            source.descriptor().transform.map(f64::to_bits),
            source_descriptor.transform.map(f64::to_bits)
        );
        assert_eq!(source.identity(), &source_identity);
        assert_eq!(source.origin(), source_origin);
        assert_eq!(result.pixels().image(), &edited_pixels);
        assert_eq!(result.pixels().attributes(), &edited_attributes);
        assert_eq!(
            ready
                .fixture
                .assets
                .lock()
                .expect("canonical service")
                .read_verified(
                    &ready.png.asset_identity,
                    &ready.fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024,
                )
                .expect("immutable indexed producer retained after edit restoration"),
            indexed_bytes
        );
    }

    mod caller_scene_workspace_controls {
        use super::*;
        use crate::graph::ImportedImageItem;
        use crate::image_preview::CpuPreviewError;
        use gpui::VisualTestContext;
        use workspace::item::Item as _;

        async fn workspace_image<'a>(
            ready: &ReadyImage,
            cx: &'a mut TestAppContext,
        ) -> (
            Entity<ImportedImageItem>,
            Entity<workspace::Workspace>,
            &'a mut VisualTestContext,
        ) {
            cx.update(|cx| {
                cx.set_global(db::AppDatabase::test_new());
                let app_state = workspace::AppState::test(cx);
                workspace::AppState::set_global(app_state, cx);
                crate::graph::init(cx);
            });
            let fs = FakeFs::new(cx.executor());
            fs.insert_tree(Path::new("/project"), serde_json::json!({}))
                .await;
            let project = Project::test(fs, [Path::new("/project")], cx).await;
            let (host, window) = cx.add_window_view(|window, cx| {
                workspace::MultiWorkspace::test_new(project, window, cx)
            });
            let workspace = host.read_with(window, |host, _| host.workspace().clone());
            let item = workspace.update_in(window, |workspace, window, cx| {
                let item = cx.new(|cx| {
                    ImportedImageItem::from_session(ready.owner.clone(), 4096, window, cx)
                });
                workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
                item.focus_handle(cx).focus(window, cx);
                item
            });
            window.simulate_resize(gpui::size(gpui::px(900.), gpui::px(700.)));
            window.run_until_parked();
            (item, workspace, window)
        }

        fn paint(window: &mut VisualTestContext) {
            window.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
            });
        }

        fn presented(
            item: &Entity<ImportedImageItem>,
            window: &VisualTestContext,
        ) -> Arc<gpui::RenderImage> {
            item.read_with(window, |item, cx| {
                item.presentation()
                    .read(cx)
                    .image()
                    .expect("native Item image")
                    .clone()
            })
        }

        #[gpui::test]
        async fn at_r20_2_workspace_installed_exact_scene_survives_redraw_and_original_edit_transition(
            cx: &mut TestAppContext,
        ) {
            let ready = ready_scene_result(cx);
            ready
                .owner
                .update(cx, |owner, cx| owner.set_operation(None, cx));
            cx.run_until_parked();
            let configuration = exact_scene_configuration(35, 21, 1.5);
            let preview = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
                })
                .expect("explicit Original scene target")
                .await
                .expect("actual original renderer");
            assert_exact_scene_frame(&preview, 35, 21);
            assert!(preview.result.is_none());
            let original = preview.image.clone();
            let original_bytes = original.as_bytes(0).expect("exact scene frame").to_vec();
            let source = preview.source.clone();
            let before_image = source.image().clone();
            let before_descriptor = *source.descriptor();
            let revision = preview.source_revision;
            let (item, _, window) = workspace_image(&ready, cx).await;
            item.update_in(window, |item, window, cx| {
                item.accept_scene_preview(preview, window, cx)
            })
            .expect("install explicit exact scene");
            for _ in 0..3 {
                paint(window);
                let image = presented(&item, window);
                assert!(Arc::ptr_eq(&image, &original));
                assert_eq!(
                    image.as_bytes(0).expect("displayed scene bytes"),
                    original_bytes
                );
                assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
                assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                        .is_none()
                );
                assert!(window.update(|window, _| window.has_image_atlas_entry(&original)));
                assert!(item.read_with(window, |item, cx| item.presentation().read(cx).is_fresh()));
                item.read_with(window, |item, cx| {
                    let receipt = item
                        .presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .expect("one retained exact receipt");
                    assert!(item.session().read(cx).scene_preview_is_current(receipt));
                    assert!(Arc::ptr_eq(&receipt.image, &original));
                    assert_eq!(
                        receipt.scene.configuration().viewport.resolution,
                        configuration.viewport.resolution
                    );
                    assert_eq!(
                        receipt.scene.configuration().scale.to_bits(),
                        configuration.scale.to_bits()
                    );
                });
            }
            ready.owner.update(window, |owner, cx| {
                owner.set_operation(Some(CpuImageEdit::Invert), cx)
            });
            window.run_until_parked();
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &original));
            assert!(
                window
                    .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                    .is_some()
            );
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_some()
            );
            let edited = ready
                .owner
                .update(window, |owner, cx| {
                    owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
                })
                .expect("explicit edited target")
                .await
                .expect("original edited scene");
            let result = edited
                .result
                .as_ref()
                .expect("Invert source result")
                .clone();
            assert_eq!(result.operation(), CpuImageEdit::Invert);
            assert!(Arc::ptr_eq(&edited.source, &source));
            let image = edited.image.clone();
            let bytes = image.as_bytes(0).expect("edited frame").to_vec();
            assert_ne!(bytes, original_bytes);
            let original_pixel = &original_bytes[(3 * 35 + 2) * 4..(3 * 35 + 2) * 4 + 4];
            let edited_pixel = &bytes[(3 * 35 + 2) * 4..(3 * 35 + 2) * 4 + 4];
            assert!(original_pixel[2] > original_pixel[0]);
            assert!(edited_pixel[0] > edited_pixel[2]);
            item.update_in(window, |item, window, cx| {
                item.accept_scene_preview(edited, window, cx)
            })
            .expect("replace original with current edit scene");
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &image));
            assert_eq!(
                presented(&item, window)
                    .as_bytes(0)
                    .expect("native visible edit"),
                bytes
            );
            assert!(window.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_none()
            );
            assert!(!window.update(|window, _| window.has_image_atlas_entry(&original)));
            ready.owner.read_with(window, |owner, _| {
                assert_eq!(owner.resource_state().revision(), revision);
                assert!(Arc::ptr_eq(
                    owner.result().expect("editable result retained"),
                    &result
                ));
                assert!(Arc::ptr_eq(
                    owner.resource_state().committed().expect("source retained"),
                    &source
                ));
            });
            assert_eq!(source.image(), &before_image);
            assert_eq!(*source.descriptor(), before_descriptor);
        }

        #[gpui::test]
        async fn at_r20_2_workspace_scene_successors_keep_last_valid_until_replacement_and_revoke_hides(
            cx: &mut TestAppContext,
        ) {
            let ready = ready_scene_result(cx);
            let configuration = exact_scene_configuration(35, 21, 1.);
            let preview = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
                })
                .expect("initial exact scene")
                .await
                .expect("native initial result");
            let first = preview.image.clone();
            let first_scene = Arc::downgrade(&preview.scene);
            let (item, _, window) = workspace_image(&ready, cx).await;
            item.update_in(window, |item, window, cx| {
                item.accept_scene_preview(preview, window, cx)
            })
            .expect("install current receipt");
            paint(window);
            let successor = exact_scene_configuration(37, 23, 1.5);
            let next = ready
                .owner
                .update(window, |owner, cx| {
                    owner.prepare_scene_preview_at_target(successor, 37 * 23 * 4, cx)
                })
                .expect("explicit target successor");
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &first));
            assert!(
                window
                    .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                    .is_some()
            );
            assert!(first_scene.upgrade().is_some());
            let next = next.await.expect("new target receipt");
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &first));
            let second = next.image.clone();
            item.update_in(window, |item, window, cx| {
                item.accept_scene_preview(next, window, cx)
            })
            .expect("replace with exact successor");
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &second));
            assert!(first_scene.upgrade().is_none());
            assert!(!window.update(|window, _| window.has_image_atlas_entry(&first)));
            let mut replacement = load(&ready, "explicit-workspace-scene-source-successor");
            replacement.transform = [16., 0., 0., 8., 0., 0.];
            ready
                .owner
                .update(window, |owner, cx| owner.load_encoded_png(replacement, cx))
                .expect("real source refresh");
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &second));
            assert!(
                window
                    .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                    .is_some()
            );
            window.run_until_parked();
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &second));
            let recovered = ready
                .owner
                .update(window, |owner, cx| {
                    owner.prepare_scene_preview_at_target(successor, 37 * 23 * 4, cx)
                })
                .expect("fresh successor source")
                .await
                .expect("actual recovery scene");
            let source = recovered.source.clone();
            let image = recovered.image.clone();
            let retained_scene = Arc::downgrade(&recovered.scene);
            item.update_in(window, |item, window, cx| {
                item.accept_scene_preview(recovered, window, cx)
            })
            .expect("install fresh source receipt");
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &image));
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_none()
            );
            ready
                .owner
                .update(window, |owner, cx| owner.revoke(source.identity(), cx))
                .expect("genuine current resource revocation");
            window.run_until_parked();
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-IMPORTED-REVOKED").is_some());
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            assert!(item.read_with(window, |item, cx| {
                item.presentation().read(cx).image().is_none()
            }));
            assert!(retained_scene.upgrade().is_none());
            assert!(item.read_with(window, |item, cx| {
                item.presentation()
                    .read(cx)
                    .retained_scene_preview()
                    .is_none()
            }));
            assert!(!window.update(|window, _| window.has_image_atlas_entry(&image)));
            assert!(
                !image
                    .as_bytes(0)
                    .expect("immutable captured frame")
                    .is_empty()
            );
        }

        #[gpui::test]
        async fn at_r20_2_workspace_scene_rejects_foreign_legacy_and_oversize_receipts_with_alert_recovery(
            cx: &mut TestAppContext,
        ) {
            let ready = ready_scene_result(cx);
            let configuration = exact_scene_configuration(35, 21, 1.);
            let preview = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
                })
                .expect("current scene")
                .await
                .expect("native scene");
            let image = preview.image.clone();
            let foreign = ready_scene_result(cx);
            let foreign_receipt = foreign
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview_at_target(configuration, 35 * 21 * 4, cx)
                })
                .expect("other genuine session")
                .await
                .expect("foreign current receipt");
            let (item, _, window) = workspace_image(&ready, cx).await;
            item.update_in(window, |item, window, cx| {
                item.accept_scene_preview(preview, window, cx)
            })
            .expect("install current scene");
            paint(window);
            assert_eq!(
                item.update_in(window, |item, window, cx| {
                    item.accept_scene_preview(foreign_receipt, window, cx)
                }),
                Err(CpuPreviewError::ResultNotFresh)
            );
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &image));
            assert!(
                window
                    .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                    .is_some()
            );
            assert_eq!(
                item.read_with(window, |item, _| item
                    .preview_diagnostic()
                    .map(str::to_owned)),
                Some(CpuPreviewError::ResultNotFresh.to_string())
            );
            let legacy = ready
                .owner
                .update(window, |owner, cx| {
                    owner.prepare_scene_preview(scene_configuration(), 1., 16 * 8 * 4 * 4, cx)
                })
                .expect("real legacy renderer path")
                .await
                .expect("legacy receipt");
            assert_eq!(
                item.update_in(window, |item, window, cx| {
                    item.accept_scene_preview(legacy, window, cx)
                }),
                Err(CpuPreviewError::InvalidPixelLayout)
            );
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &image));
            assert_eq!(
                item.read_with(window, |item, _| item
                    .preview_diagnostic()
                    .map(str::to_owned)),
                Some(CpuPreviewError::InvalidPixelLayout.to_string())
            );
            let large = ready
                .owner
                .update(window, |owner, cx| {
                    owner.prepare_scene_preview_at_target(
                        exact_scene_configuration(41, 27, 1.),
                        41 * 27 * 4,
                        cx,
                    )
                })
                .expect("explicit target above Item cap")
                .await
                .expect("valid larger native receipt");
            assert_eq!(
                large.image.as_bytes(0).expect("larger actual bytes").len(),
                41 * 27 * 4
            );
            assert_eq!(
                item.update_in(window, |item, window, cx| {
                    item.accept_scene_preview(large, window, cx)
                }),
                Err(CpuPreviewError::PreviewBudgetExceeded)
            );
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &image));
            assert!(
                window
                    .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                    .is_some()
            );
            assert_eq!(
                item.read_with(window, |item, _| item
                    .preview_diagnostic()
                    .map(str::to_owned)),
                Some(CpuPreviewError::PreviewBudgetExceeded.to_string())
            );
            let recovered = ready
                .owner
                .update(window, |owner, cx| {
                    owner.prepare_scene_preview_at_target(
                        exact_scene_configuration(32, 32, 1.),
                        4096,
                        cx,
                    )
                })
                .expect("exact Item-cap valid retry")
                .await
                .expect("fresh recovery");
            let replacement = recovered.image.clone();
            assert_eq!(
                replacement
                    .as_bytes(0)
                    .expect("exact-cap recovery bytes")
                    .len(),
                4096
            );
            item.update_in(window, |item, window, cx| {
                item.accept_scene_preview(recovered, window, cx)
            })
            .expect("valid receipt clears rejection");
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &replacement));
            assert!(
                window
                    .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                    .is_none()
            );
            assert!(item.read_with(window, |item, _| item.preview_diagnostic().is_none()));
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_none()
            );
        }

        #[gpui::test]
        async fn at_r20_2_workspace_scene_split_retains_one_receipt_and_retires_both_window_atlases(
            cx: &mut TestAppContext,
        ) {
            let ready = ready_scene_result(cx);
            let preview = ready
                .owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview_at_target(
                        exact_scene_configuration(35, 21, 1.5),
                        35 * 21 * 4,
                        cx,
                    )
                })
                .expect("one explicit shared target")
                .await
                .expect("native receipt");
            let image = preview.image.clone();
            let bytes = image.as_bytes(0).expect("shared frame bytes").to_vec();
            let weak_scene = Arc::downgrade(&preview.scene);
            let source = preview.source.clone();
            let revision = preview.source_revision;
            let result = preview
                .result
                .as_ref()
                .expect("original edit result")
                .clone();
            let mut second_context = cx.clone();
            let (item, workspace, window) = workspace_image(&ready, cx).await;
            item.update_in(window, |item, window, cx| {
                item.accept_scene_preview(preview, window, cx)
            })
            .expect("one installed receipt");
            let (project, app_state) = workspace.read_with(window, |workspace, _| {
                (workspace.project().clone(), workspace.app_state().clone())
            });
            let (second_host, second_window) = second_context.add_window_view(|window, cx| {
                let workspace =
                    cx.new(|cx| workspace::Workspace::new(None, project, app_state, window, cx));
                workspace::MultiWorkspace::new(workspace, window, cx)
            });
            let second_workspace =
                second_host.read_with(second_window, |host, _| host.workspace().clone());
            let split = item
                .update_in(second_window, |item, window, cx| {
                    item.clone_on_split(None, window, cx)
                })
                .await
                .expect("existing native split");
            second_workspace.update_in(second_window, |workspace, window, cx| {
                workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx);
                split.focus_handle(cx).focus(window, cx);
            });
            let (session_id, presentation_id, weak_presentation, focus) =
                item.read_with(window, |item, cx| {
                    (
                        item.session().entity_id(),
                        item.presentation().entity_id(),
                        item.presentation().downgrade(),
                        item.focus_handle(cx),
                    )
                });
            split.read_with(second_window, |item, cx| {
                assert_eq!(item.session().entity_id(), session_id);
                assert_eq!(item.presentation().entity_id(), presentation_id);
                assert_ne!(item.focus_handle(cx), focus);
            });
            for _ in 0..2 {
                paint(window);
                paint(second_window);
                assert!(Arc::ptr_eq(&presented(&item, window), &image));
                assert!(Arc::ptr_eq(&presented(&split, second_window), &image));
                assert_eq!(
                    presented(&split, second_window)
                        .as_bytes(0)
                        .expect("split exact frame"),
                    bytes
                );
                assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
                assert!(second_window.update(|window, _| window.has_image_atlas_entry(&image)));
                assert!(weak_scene.upgrade().is_some());
                item.read_with(window, |item, cx| {
                    let receipt = item
                        .presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .expect("shared retained receipt");
                    assert!(item.session().read(cx).scene_preview_is_current(receipt));
                    assert!(Arc::ptr_eq(&receipt.image, &image));
                    assert_eq!(receipt.source_revision, revision);
                });
                split.read_with(second_window, |item, cx| {
                    let receipt = item
                        .presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .expect("same split receipt");
                    assert!(item.session().read(cx).scene_preview_is_current(receipt));
                    assert!(Arc::ptr_eq(&receipt.image, &image));
                });
                ready.owner.read_with(window, |owner, _| {
                    assert_eq!(owner.resource_state().revision(), revision);
                    assert!(Arc::ptr_eq(
                        owner.resource_state().committed().expect("same source"),
                        &source
                    ));
                    assert!(Arc::ptr_eq(
                        owner.result().expect("same source edit"),
                        &result
                    ));
                });
            }
            let first_id = item.entity_id();
            workspace
                .update_in(window, |workspace, window, cx| {
                    workspace.active_pane().clone().update(cx, |pane, cx| {
                        pane.close_item_by_id(first_id, workspace::SaveIntent::Close, window, cx)
                    })
                })
                .await
                .expect("actual native first Item close");
            assert!(workspace.read_with(window, |workspace, cx| {
                workspace.active_item_as::<ImportedImageItem>(cx).is_none()
            }));
            window.update(|_, _| drop(item));
            paint(window);
            window.run_until_parked();
            paint(second_window);
            assert!(weak_presentation.upgrade().is_some());
            assert!(weak_scene.upgrade().is_some());
            assert!(Arc::ptr_eq(&presented(&split, second_window), &image));
            let split_id = split.entity_id();
            second_workspace
                .update_in(second_window, |workspace, window, cx| {
                    workspace.active_pane().clone().update(cx, |pane, cx| {
                        pane.close_item_by_id(split_id, workspace::SaveIntent::Close, window, cx)
                    })
                })
                .await
                .expect("actual native final split close");
            assert!(second_workspace.read_with(second_window, |workspace, cx| {
                workspace.active_item_as::<ImportedImageItem>(cx).is_none()
            }));
            second_window.update(|_, _| drop(split));
            paint(window);
            paint(second_window);
            second_window.run_until_parked();
            window.run_until_parked();
            assert!(weak_presentation.upgrade().is_none());
            assert!(weak_scene.upgrade().is_none());
            assert!(!window.update(|window, _| window.has_image_atlas_entry(&image)));
            assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&image)));
            assert_eq!(
                image
                    .as_bytes(0)
                    .expect("captured immutable frame after retirement"),
                bytes
            );
        }

        mod requested_scene_workspace_controls {
            use super::*;
            use crate::graph::{CancelImageLoad, InvertImage};

            async fn configured_workspace_image<'a>(
                ready: &ReadyImage,
                configuration: graphics_core::render::CpuSvgSceneConfiguration,
                cx: &'a mut TestAppContext,
            ) -> (
                Entity<ImportedImageItem>,
                Entity<workspace::Workspace>,
                &'a mut VisualTestContext,
            ) {
                let (pixel_item, workspace, window) = workspace_image(ready, cx).await;
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREPARE-SCENE")
                        .is_none()
                );
                workspace
                    .update_in(window, |workspace, window, cx| {
                        workspace.active_pane().clone().update(cx, |pane, cx| {
                            pane.close_item_by_id(
                                pixel_item.entity_id(),
                                workspace::SaveIntent::Close,
                                window,
                                cx,
                            )
                        })
                    })
                    .await
                    .expect("close unchanged pixel-only fixture Item");
                window.update(|_, _| drop(pixel_item));
                let item = workspace.update_in(window, |workspace, window, cx| {
                    let item = cx.new(|cx| {
                        ImportedImageItem::from_session_with_scene_configuration(
                            ready.owner.clone(),
                            4096,
                            configuration,
                            window,
                            cx,
                        )
                    });
                    workspace.add_item_to_active_pane(
                        Box::new(item.clone()),
                        None,
                        true,
                        window,
                        cx,
                    );
                    item.focus_handle(cx).focus(window, cx);
                    item
                });
                paint(window);
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREPARE-SCENE")
                        .is_some()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_none()
                );
                assert!(item.read_with(window, |item, cx| {
                    item.presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .is_none()
                }));
                (item, workspace, window)
            }

            fn click_scene(window: &mut VisualTestContext) {
                let bounds = window
                    .debug_bounds("GRAPHITE-IMPORTED-PREPARE-SCENE")
                    .expect("actual configured scene button");
                window.simulate_click(bounds.center(), gpui::Modifiers::default());
                window.run_until_parked();
                paint(window);
            }

            fn begin_scene_without_pumping(window: &mut VisualTestContext) {
                let position = window
                    .debug_bounds("GRAPHITE-IMPORTED-PREPARE-SCENE")
                    .expect("native scene button")
                    .center();
                window.update(|window, cx| {
                    window.dispatch_event(
                        gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                            position,
                            button: gpui::MouseButton::Left,
                            modifiers: gpui::Modifiers::default(),
                            click_count: 1,
                            first_mouse: false,
                        }),
                        cx,
                    );
                    window.dispatch_event(
                        gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                            position,
                            button: gpui::MouseButton::Left,
                            modifiers: gpui::Modifiers::default(),
                            click_count: 1,
                        }),
                        cx,
                    );
                });
            }

            #[gpui::test]
            async fn at_r20_2_configured_workspace_button_prepares_original_scene_and_repeats_after_native_edit(
                cx: &mut TestAppContext,
            ) {
                let ready = ready_scene_result(cx);
                ready
                    .owner
                    .update(cx, |owner, cx| owner.set_operation(None, cx));
                cx.run_until_parked();
                let configuration = exact_scene_configuration(35, 21, 1.5);
                let source = ready.owner.read_with(cx, |owner, _| {
                    owner
                        .resource_state()
                        .committed()
                        .expect("real source")
                        .clone()
                });
                let source_image = source.image().clone();
                let source_descriptor = *source.descriptor();
                let source_origin = source.origin();
                let revision = ready
                    .owner
                    .read_with(cx, |owner, _| owner.resource_state().revision());
                let (item, _, window) = configured_workspace_image(&ready, configuration, cx).await;
                click_scene(window);
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_none()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                        .is_none()
                );
                assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
                assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
                let original = presented(&item, window);
                let original_bytes = original.as_bytes(0).expect("native source frame").to_vec();
                item.read_with(window, |item, cx| {
                    let receipt = item
                        .presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .expect("button installed receipt");
                    assert!(item.session().read(cx).scene_preview_is_current(receipt));
                    assert!(Arc::ptr_eq(&receipt.image, &original));
                    assert!(Arc::ptr_eq(&receipt.source, &source));
                    assert!(receipt.result.is_none());
                    assert_eq!(receipt.source_revision, revision);
                    assert_exact_scene_frame(receipt, 35, 21);
                    assert_eq!(
                        receipt.scene.configuration().scale.to_bits(),
                        configuration.scale.to_bits()
                    );
                    assert_eq!(
                        receipt
                            .scene
                            .configuration()
                            .viewport
                            .transform
                            .to_cols_array()
                            .map(f64::to_bits),
                        configuration
                            .viewport
                            .transform
                            .to_cols_array()
                            .map(f64::to_bits)
                    );
                });
                let original_pixel = &original_bytes[(3 * 35 + 2) * 4..(3 * 35 + 2) * 4 + 4];
                assert!(original_pixel[2] > original_pixel[0] && original_pixel[3] > 0);
                let edit_focus = item.read_with(window, |item, cx| item.focus_handle(cx));
                window.update(|window, cx| edit_focus.dispatch_action(&InvertImage, window, cx));
                window.run_until_parked();
                paint(window);
                assert!(Arc::ptr_eq(&presented(&item, window), &original));
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                        .is_some()
                );
                click_scene(window);
                let edited = presented(&item, window);
                assert!(!Arc::ptr_eq(&edited, &original));
                let edited_bytes = edited.as_bytes(0).expect("original Invert scene frame");
                assert_ne!(edited_bytes, original_bytes);
                let edited_pixel = &edited_bytes[(3 * 35 + 2) * 4..(3 * 35 + 2) * 4 + 4];
                assert!(edited_pixel[0] > edited_pixel[2] && edited_pixel[3] > 0);
                assert_eq!(edited_pixel[3], original_pixel[3]);
                assert!(window.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_none()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                        .is_none()
                );
                item.read_with(window, |item, cx| {
                    let receipt = item
                        .presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .expect("edit installed by button");
                    let result = receipt.result.as_ref().expect("original Invert result");
                    assert_eq!(result.operation(), CpuImageEdit::Invert);
                    assert!(Arc::ptr_eq(&receipt.source, &source));
                    assert!(item.session().read(cx).scene_preview_is_current(receipt));
                    assert!(Arc::ptr_eq(
                        item.session().read(cx).result().expect("editable result"),
                        result
                    ));
                    assert_eq!(receipt.source_revision, revision);
                });
                assert_eq!(source.image(), &source_image);
                assert_eq!(*source.descriptor(), source_descriptor);
                assert_eq!(source.origin(), source_origin);
                assert!(!window.update(|window, _| window.has_image_atlas_entry(&original)));
                assert!(window.update(|window, _| window.has_image_atlas_entry(&edited)));
            }

            #[gpui::test]
            async fn at_r20_2_configured_workspace_scene_failure_retry_preserves_real_source_and_last_valid_pixels(
                cx: &mut TestAppContext,
            ) {
                let ready = ready_scene_result(cx);
                let configuration = exact_scene_configuration(35, 21, 1.);
                let (item, _, window) = configured_workspace_image(&ready, configuration, cx).await;
                click_scene(window);
                let last_valid = presented(&item, window);
                let source = ready.owner.read_with(window, |owner, _| {
                    owner.resource_state().committed().expect("source").clone()
                });
                let source_image = source.image().clone();
                let result = ready
                    .owner
                    .read_with(window, |owner, _| owner.result().expect("edit").clone());
                let revision = ready
                    .owner
                    .read_with(window, |owner, _| owner.resource_state().revision());
                let indexed_bytes = ready
                    .fixture
                    .assets
                    .lock()
                    .expect("real indexed service")
                    .read_verified(
                        &ready.png.asset_identity,
                        &ready.fixture.input_authorization,
                        &sim_tensor::CancellationToken::default(),
                        16 * 1024,
                    )
                    .expect("real source bytes");
                let mut denied_digest = load(&ready, "explicit-request-failed-digest");
                denied_digest.transform = [16., 0., 0., 8., 0., 0.];
                denied_digest.identity.content_sha256 = [0; 32];
                ready
                    .owner
                    .update(window, |owner, cx| {
                        owner.load_encoded_png(denied_digest, cx)
                    })
                    .expect("actual read/decode failure request");
                begin_scene_without_pumping(window);
                paint(window);
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                        .is_some()
                );
                assert_eq!(
                    item.read_with(window, |item, _| item
                        .preview_diagnostic()
                        .map(str::to_owned)),
                    Some("Image source is not current".into())
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_none()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                        .is_some()
                );
                assert!(Arc::ptr_eq(&presented(&item, window), &last_valid));
                window.run_until_parked();
                paint(window);
                ready.owner.read_with(window, |owner, _| {
                    assert!(!owner.resource_state().is_fresh());
                    assert_eq!(owner.resource_state().revision(), revision);
                    assert!(Arc::ptr_eq(
                        owner
                            .resource_state()
                            .committed()
                            .expect("original retained"),
                        &source
                    ));
                    assert!(Arc::ptr_eq(
                        owner.result().expect("last valid edit"),
                        &result
                    ));
                    assert!(owner.diagnostic().is_some());
                });
                click_scene(window);
                assert!(Arc::ptr_eq(&presented(&item, window), &last_valid));
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                        .is_some()
                );
                let mut recovery = load(&ready, "explicit-request-correct-digest-recovery");
                recovery.transform = [16., 0., 0., 8., 0., 0.];
                ready
                    .owner
                    .update(window, |owner, cx| owner.load_encoded_png(recovery, cx))
                    .expect("genuine successor input");
                window.run_until_parked();
                paint(window);
                assert!(Arc::ptr_eq(&presented(&item, window), &last_valid));
                click_scene(window);
                let recovered = presented(&item, window);
                assert!(!Arc::ptr_eq(&recovered, &last_valid));
                assert_eq!(recovered.as_bytes(0), last_valid.as_bytes(0));
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                        .is_none()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_none()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                        .is_none()
                );
                assert!(item.read_with(window, |item, cx| {
                    let receipt = item
                        .presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .expect("recovered receipt");
                    item.session().read(cx).scene_preview_is_current(receipt)
                }));
                assert_eq!(source.image(), &source_image);
                assert_eq!(
                    ready
                        .fixture
                        .assets
                        .lock()
                        .expect("canonical service after retry")
                        .read_verified(
                            &ready.png.asset_identity,
                            &ready.fixture.input_authorization,
                            &sim_tensor::CancellationToken::default(),
                            16 * 1024
                        )
                        .expect("unchanged indexed PNG"),
                    indexed_bytes
                );
            }

            #[gpui::test]
            async fn at_r20_2_configured_workspace_scene_pending_duplicate_cancel_and_retry_never_publish_cancelled_frame(
                cx: &mut TestAppContext,
            ) {
                let ready = ready_scene_result(cx);
                let configuration = exact_scene_configuration(35, 21, 1.5);
                let (item, _, window) = configured_workspace_image(&ready, configuration, cx).await;
                click_scene(window);
                let last_valid = presented(&item, window);
                let generation = item.read_with(window, |item, cx| {
                    item.presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .expect("initial receipt")
                        .scene_generation
                });
                let source = ready.owner.read_with(window, |owner, _| {
                    owner.resource_state().committed().expect("source").clone()
                });
                let result = ready
                    .owner
                    .read_with(window, |owner, _| owner.result().expect("result").clone());
                begin_scene_without_pumping(window);
                paint(window);
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_some()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                        .is_some()
                );
                assert!(Arc::ptr_eq(&presented(&item, window), &last_valid));
                begin_scene_without_pumping(window);
                paint(window);
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_some()
                );
                let cancel_focus = item.read_with(window, |item, cx| item.focus_handle(cx));
                window.update(|window, cx| {
                    cancel_focus.dispatch_action(&CancelImageLoad, window, cx)
                });
                paint(window);
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_none()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                        .is_some()
                );
                assert_eq!(
                    item.read_with(window, |item, _| item
                        .preview_diagnostic()
                        .map(str::to_owned)),
                    Some("Scene preview preparation was cancelled".into())
                );
                window.run_until_parked();
                paint(window);
                assert!(Arc::ptr_eq(&presented(&item, window), &last_valid));
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                        .is_some()
                );
                ready.owner.read_with(window, |owner, _| {
                    assert!(owner.resource_state().is_fresh());
                    assert!(Arc::ptr_eq(
                        owner.resource_state().committed().expect("same source"),
                        &source
                    ));
                    assert!(Arc::ptr_eq(owner.result().expect("same result"), &result));
                });
                click_scene(window);
                let recovered = presented(&item, window);
                assert!(!Arc::ptr_eq(&recovered, &last_valid));
                assert_eq!(recovered.as_bytes(0), last_valid.as_bytes(0));
                item.read_with(window, |item, cx| {
                    let receipt = item
                        .presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .expect("explicit retry receipt");
                    assert_eq!(receipt.scene_generation, generation + 2);
                    assert!(item.session().read(cx).scene_preview_is_current(receipt));
                });
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_none()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-PREVIEW-ERROR")
                        .is_none()
                );
                assert!(!window.update(|window, _| window.has_image_atlas_entry(&last_valid)));
            }

            #[gpui::test]
            async fn at_r20_2_configured_workspace_scene_split_redraw_creates_no_request_and_pending_item_drop_cannot_publish(
                cx: &mut TestAppContext,
            ) {
                let ready = ready_scene_result(cx);
                let configuration = exact_scene_configuration(35, 21, 1.5);
                let mut second_context = cx.clone();
                let (item, workspace, window) =
                    configured_workspace_image(&ready, configuration, cx).await;
                click_scene(window);
                let image = presented(&item, window);
                let bytes = image.as_bytes(0).expect("shared frame").to_vec();
                let (scene_generation, weak_scene, weak_presentation, session_id, presentation_id) =
                    item.read_with(window, |item, cx| {
                        let presentation = item.presentation().read(cx);
                        let receipt = presentation
                            .retained_scene_preview()
                            .expect("configured accepted target");
                        (
                            receipt.scene_generation,
                            Arc::downgrade(&receipt.scene),
                            item.presentation().downgrade(),
                            item.session().entity_id(),
                            item.presentation().entity_id(),
                        )
                    });
                let source = ready.owner.read_with(window, |owner, _| {
                    owner.resource_state().committed().expect("source").clone()
                });
                let result = ready
                    .owner
                    .read_with(window, |owner, _| owner.result().expect("result").clone());
                let weak_session = ready.owner.downgrade();
                let (project, app_state) = workspace.read_with(window, |workspace, _| {
                    (workspace.project().clone(), workspace.app_state().clone())
                });
                let (second_host, second_window) = second_context.add_window_view(|window, cx| {
                    let workspace = cx
                        .new(|cx| workspace::Workspace::new(None, project, app_state, window, cx));
                    workspace::MultiWorkspace::new(workspace, window, cx)
                });
                let second_workspace =
                    second_host.read_with(second_window, |host, _| host.workspace().clone());
                let split = item
                    .update_in(second_window, |item, window, cx| {
                        item.clone_on_split(None, window, cx)
                    })
                    .await
                    .expect("native split");
                second_workspace.update_in(second_window, |workspace, window, cx| {
                    workspace.add_item_to_active_pane(
                        Box::new(split.clone()),
                        None,
                        true,
                        window,
                        cx,
                    );
                    split.focus_handle(cx).focus(window, cx);
                });
                for _ in 0..2 {
                    paint(window);
                    paint(second_window);
                    assert!(
                        second_window
                            .debug_bounds("GRAPHITE-IMPORTED-PREPARE-SCENE")
                            .is_some()
                    );
                    assert!(
                        second_window
                            .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                            .is_none()
                    );
                    assert!(Arc::ptr_eq(&presented(&item, window), &image));
                    assert!(Arc::ptr_eq(&presented(&split, second_window), &image));
                    split.read_with(second_window, |item, cx| {
                        assert_eq!(item.session().entity_id(), session_id);
                        assert_eq!(item.presentation().entity_id(), presentation_id);
                        let receipt = item
                            .presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .expect("no split-triggered request");
                        assert_eq!(receipt.scene_generation, scene_generation);
                        assert!(item.session().read(cx).scene_preview_is_current(receipt));
                        assert_eq!(
                            receipt.scene.configuration().viewport.resolution,
                            configuration.viewport.resolution
                        );
                        assert_eq!(
                            receipt.scene.configuration().scale.to_bits(),
                            configuration.scale.to_bits()
                        );
                    });
                    assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
                    assert!(second_window.update(|window, _| window.has_image_atlas_entry(&image)));
                }
                let weak_item = item.downgrade();
                begin_scene_without_pumping(window);
                paint(window);
                assert!(
                    window
                        .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                        .is_some()
                );
                let item_id = item.entity_id();
                workspace.update_in(window, |workspace, window, cx| {
                    workspace.active_pane().clone().update(cx, |pane, cx| {
                        pane.remove_item(item_id, false, true, window, cx)
                    });
                });
                window.update(|_, _| drop(item));
                paint(window);
                window.run_until_parked();
                paint(second_window);
                assert!(weak_item.upgrade().is_none());
                assert!(weak_session.upgrade().is_some());
                assert!(weak_presentation.upgrade().is_some());
                assert!(weak_scene.upgrade().is_some());
                assert!(Arc::ptr_eq(&presented(&split, second_window), &image));
                assert_eq!(
                    presented(&split, second_window)
                        .as_bytes(0)
                        .expect("no late replacement"),
                    bytes
                );
                assert!(
                    second_window
                        .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                        .is_some()
                );
                ready.owner.read_with(second_window, |owner, _| {
                    assert!(owner.resource_state().is_fresh());
                    assert!(Arc::ptr_eq(
                        owner
                            .resource_state()
                            .committed()
                            .expect("external owner retained source"),
                        &source
                    ));
                    assert!(Arc::ptr_eq(
                        owner.result().expect("retained result"),
                        &result
                    ));
                });
                click_scene(second_window);
                let recovered = presented(&split, second_window);
                assert!(!Arc::ptr_eq(&recovered, &image));
                assert_eq!(
                    recovered.as_bytes(0).expect("same source/config recovery"),
                    bytes
                );
                assert!(weak_scene.upgrade().is_none());
                assert!(!window.update(|window, _| window.has_image_atlas_entry(&image)));
                assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&image)));
                let recovered_scene = split.read_with(second_window, |item, cx| {
                    let receipt = item
                        .presentation()
                        .read(cx)
                        .retained_scene_preview()
                        .expect("split explicit request");
                    assert_eq!(receipt.scene_generation, scene_generation + 2);
                    assert!(item.session().read(cx).scene_preview_is_current(receipt));
                    Arc::downgrade(&receipt.scene)
                });
                second_workspace
                    .update_in(second_window, |workspace, window, cx| {
                        workspace.active_pane().clone().update(cx, |pane, cx| {
                            pane.close_item_by_id(
                                split.entity_id(),
                                workspace::SaveIntent::Close,
                                window,
                                cx,
                            )
                        })
                    })
                    .await
                    .expect("normal final native close");
                second_window.update(|_, _| drop(split));
                paint(window);
                paint(second_window);
                second_window.run_until_parked();
                assert!(weak_presentation.upgrade().is_none());
                assert!(recovered_scene.upgrade().is_none());
                assert!(!window.update(|window, _| window.has_image_atlas_entry(&recovered)));
                assert!(
                    !second_window.update(|window, _| window.has_image_atlas_entry(&recovered))
                );
                assert_eq!(
                    recovered
                        .as_bytes(0)
                        .expect("captured pixels remain immutable"),
                    bytes
                );
            }

            mod root_scene_opener_controls {
                use super::*;
                use crate::interop::{GraphiteAssetBindingError, GraphiteAssetImageRequest};
                use sim_runtime::assets::AssetNamespace;

                fn request(
                    png: &IndexedRequestPng,
                    transform: [f64; 6],
                ) -> GraphiteAssetImageRequest {
                    GraphiteAssetImageRequest {
                        asset_identity: png.asset_identity.clone(),
                        identity: png.image_identity.clone(),
                        transform,
                        encoded_budget: EncodedPngBudget {
                            max_encoded_bytes: 16 * 1024,
                            max_pixel_bytes: 4096,
                            max_metadata_bytes: 4096,
                        },
                        max_result_bytes: 4096,
                    }
                }

                #[gpui::test]
                async fn at_r28_1_root_input_scene_opener_preserves_admission_and_explicit_button_target(
                    cx: &mut TestAppContext,
                ) {
                    let ready = ready_image(cx);
                    let record = ready
                        .fixture
                        .assets
                        .lock()
                        .expect("indexed owner")
                        .record(&ready.png.asset_identity)
                        .expect("actual Input record");
                    let configuration = exact_scene_configuration(35, 21, 1.5);
                    let (fixture_item, workspace, window) = workspace_image(&ready, cx).await;
                    let count = workspace.read_with(window, |workspace, cx| {
                        workspace.active_pane().read(cx).items_len()
                    });
                    assert!(matches!(
                        workspace.update_in(window, |workspace, window, cx| {
                            crate::open_asset_image_with_scene_configuration(
                                request(&ready.png, [16., 0., 0., 8., 0., 0.]),
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        }),
                        Err(GraphiteAssetBindingError::Inactive)
                    ));
                    window
                        .update(|_, cx| {
                            crate::register_asset_binding(
                                ready.fixture.assets.clone(),
                                ready.fixture.input_authorization.clone(),
                                cx,
                            )
                        })
                        .expect("supplied genuine Input reader");
                    let mut foreign = request(&ready.png, [16., 0., 0., 8., 0., 0.]);
                    foreign.asset_identity.profile_id = "foreign-profile".into();
                    assert!(matches!(
                        workspace.update_in(window, |workspace, window, cx| {
                            crate::open_asset_image_with_scene_configuration(
                                foreign,
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        }),
                        Err(GraphiteAssetBindingError::ProfileMismatch)
                    ));
                    let mut output = request(&ready.png, [16., 0., 0., 8., 0., 0.]);
                    output.asset_identity.namespace = AssetNamespace::Output;
                    assert!(matches!(
                        workspace.update_in(window, |workspace, window, cx| {
                            crate::open_asset_image_with_scene_configuration(
                                output,
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        }),
                        Err(GraphiteAssetBindingError::InputNamespaceRequired)
                    ));
                    assert_eq!(
                        workspace.read_with(window, |workspace, cx| workspace
                            .active_pane()
                            .read(cx)
                            .items_len()),
                        count
                    );
                    workspace
                        .update_in(window, |workspace, window, cx| {
                            workspace.active_pane().clone().update(cx, |pane, cx| {
                                pane.close_item_by_id(
                                    fixture_item.entity_id(),
                                    workspace::SaveIntent::Close,
                                    window,
                                    cx,
                                )
                            })
                        })
                        .await
                        .expect("normal fixture close");
                    window.update(|_, _| drop(fixture_item));
                    let ordinary = workspace
                        .update_in(window, |workspace, window, cx| {
                            crate::open_asset_image(
                                request(&ready.png, [1., 0., 0., 1., 0., 0.]),
                                4096,
                                workspace,
                                window,
                                cx,
                            )
                        })
                        .expect("unchanged ordinary root opener");
                    window.run_until_parked();
                    paint(window);
                    assert!(
                        window
                            .debug_bounds("GRAPHITE-IMPORTED-PREPARE-SCENE")
                            .is_none()
                    );
                    assert!(ordinary.read_with(window, |item, cx| {
                        item.presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .is_none()
                    }));
                    assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
                    workspace
                        .update_in(window, |workspace, window, cx| {
                            workspace.active_pane().clone().update(cx, |pane, cx| {
                                pane.close_item_by_id(
                                    ordinary.entity_id(),
                                    workspace::SaveIntent::Close,
                                    window,
                                    cx,
                                )
                            })
                        })
                        .await
                        .expect("normal ordinary Item close");
                    window.update(|_, _| drop(ordinary));
                    let item = workspace
                        .update_in(window, |workspace, window, cx| {
                            crate::open_asset_image_with_scene_configuration(
                                request(&ready.png, [16., 0., 0., 8., 0., 0.]),
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        })
                        .expect("explicit root caller configuration");
                    window.run_until_parked();
                    paint(window);
                    assert_eq!(
                        workspace.read_with(window, |workspace, cx| workspace
                            .active_item_as::<ImportedImageItem>(cx)
                            .expect("active image")
                            .entity_id()),
                        item.entity_id()
                    );
                    assert!(window.update(|window, cx| {
                        item.read(cx).focus_handle(cx).contains_focused(window, cx)
                    }));
                    assert!(
                        window
                            .debug_bounds("GRAPHITE-IMPORTED-PREPARE-SCENE")
                            .is_some()
                    );
                    assert!(
                        window
                            .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                            .is_none()
                    );
                    assert!(item.read_with(window, |item, cx| {
                        item.presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .is_none()
                    }));
                    let session = item.read_with(window, |item, _| item.session().clone());
                    let source = session.read_with(window, |session, _| {
                        assert!(session.resource_state().is_fresh());
                        assert_eq!(session.operation(), None);
                        session
                            .resource_state()
                            .committed()
                            .expect("authenticated source")
                            .clone()
                    });
                    let image_before = source.image().clone();
                    click_scene(window);
                    let original = presented(&item, window);
                    let bytes = original.as_bytes(0).expect("exact native frame").to_vec();
                    let pixel = &bytes[(3 * 35 + 2) * 4..(3 * 35 + 2) * 4 + 4];
                    assert!(pixel[2] > pixel[0] && pixel[3] > 0);
                    item.read_with(window, |item, cx| {
                        let receipt = item
                            .presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .expect("root button receipt");
                        assert_eq!(receipt.scene_generation, 1);
                        assert_exact_scene_frame(receipt, 35, 21);
                        assert_eq!(
                            receipt.scene.configuration().scale.to_bits(),
                            configuration.scale.to_bits()
                        );
                        assert_eq!(
                            receipt
                                .scene
                                .configuration()
                                .viewport
                                .transform
                                .to_cols_array()
                                .map(f64::to_bits),
                            configuration
                                .viewport
                                .transform
                                .to_cols_array()
                                .map(f64::to_bits)
                        );
                        assert_eq!(
                            receipt.scene.configuration().viewport.quality,
                            configuration.viewport.quality
                        );
                        assert_eq!(
                            receipt.scene.configuration().max_svg_bytes,
                            configuration.max_svg_bytes
                        );
                        assert!(receipt.result.is_none());
                        assert!(Arc::ptr_eq(&receipt.source, &source));
                        assert!(item.session().read(cx).scene_preview_is_current(receipt));
                    });
                    let focus = item.read_with(window, |item, cx| item.focus_handle(cx));
                    window.update(|window, cx| focus.dispatch_action(&InvertImage, window, cx));
                    window.run_until_parked();
                    paint(window);
                    assert!(Arc::ptr_eq(&presented(&item, window), &original));
                    assert!(
                        window
                            .debug_bounds("GRAPHITE-IMPORTED-LAST-VALID")
                            .is_some()
                    );
                    click_scene(window);
                    let edited = presented(&item, window);
                    assert!(!Arc::ptr_eq(&edited, &original));
                    let edited_bytes = edited.as_bytes(0).expect("edited native scene");
                    let edited_pixel = &edited_bytes[(3 * 35 + 2) * 4..(3 * 35 + 2) * 4 + 4];
                    assert!(edited_pixel[0] > edited_pixel[2]);
                    assert_eq!(edited_pixel[3], pixel[3]);
                    item.read_with(window, |item, cx| {
                        let receipt = item
                            .presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .expect("edited root receipt");
                        assert_eq!(receipt.scene_generation, 2);
                        assert_eq!(
                            receipt.result.as_ref().expect("original edit").operation(),
                            CpuImageEdit::Invert
                        );
                        assert!(Arc::ptr_eq(&receipt.source, &source));
                        assert!(item.session().read(cx).scene_preview_is_current(receipt));
                    });
                    assert_eq!(source.image(), &image_before);
                    let service = ready.fixture.assets.lock().expect("unchanged host owner");
                    assert_eq!(service.record(&ready.png.asset_identity), Some(record));
                    assert_eq!(
                        service
                            .read_verified(
                                &ready.png.asset_identity,
                                &ready.fixture.input_authorization,
                                &sim_tensor::CancellationToken::default(),
                                16 * 1024
                            )
                            .expect("unchanged indexed producer bytes"),
                        ready.png.encoded
                    );
                }

                #[gpui::test]
                async fn at_r28_1_committed_output_scene_opener_requires_genuine_receipt_and_output_grant(
                    cx: &mut TestAppContext,
                ) {
                    let ready = ready_image(cx);
                    let profile = ready.fixture.input_authorization.profile_id();
                    let writer = sim_runtime::authorize_native_output_committer(profile)
                        .expect("fixture publisher grant");
                    let policy = sim_runtime::PermissionPolicy::native_runtime_services(profile)
                        .expect("existing host policy");
                    let reader = policy
                        .authorize_one(
                            sim_runtime::NATIVE_API_ASSET_READER_SUBJECT,
                            sim_runtime::Capability::Asset {
                                namespace: "output".into(),
                                action: sim_runtime::AssetOperation::Read,
                            },
                        )
                        .expect("separate sealed Output Read grant");
                    let roots = ready
                        .fixture
                        .assets
                        .lock()
                        .expect("real indexed owner")
                        .roots()
                        .clone();
                    let mut publisher =
                        sim_runtime::OutputCommitter::open(roots).expect("real existing publisher");
                    let proposal = sim_runtime::OutputProposal::new(
                        "60000000-0000-0000-0000-000000000002"
                            .parse()
                            .expect("fixture UUID"),
                        AssetNamespace::Output,
                        "graphite-configured-scene-source",
                        "png",
                        0,
                        ready.png.descriptor.width,
                        ready.png.descriptor.height,
                        ready.png.encoded.clone(),
                    )
                    .expect("actual producer PNG proposal");
                    let receipts = publisher
                        .commit_proposal_batch_and_register(
                            &[proposal],
                            "2026-10-09T12:00:00+00:00"
                                .parse()
                                .expect("fixture timestamp"),
                            &mut ready
                                .fixture
                                .assets
                                .lock()
                                .expect("real registration transaction"),
                            &writer,
                            &sim_tensor::CancellationToken::default(),
                        )
                        .expect("actual committed transaction");
                    assert_eq!(receipts.len(), 1);
                    let receipt = receipts.into_iter().next().expect("genuine sealed receipt");
                    let receipt_before = receipt.clone();
                    let identity = receipt.operation().identity.clone();
                    let before = ready
                        .fixture
                        .assets
                        .lock()
                        .expect("indexed owner")
                        .record(&identity)
                        .expect("registered Output record");
                    let output_request = || {
                        let mut output = request(&ready.png, [16., 0., 0., 8., 0., 0.]);
                        output.asset_identity = identity.clone();
                        output.identity.asset_id = identity
                            .to_reference()
                            .expect("canonical Output reference")
                            .into();
                        output.identity.version_id =
                            "explicit-fixture-output-label-not-durable-version-proof".into();
                        output
                    };
                    let configuration = exact_scene_configuration(35, 21, 1.5);
                    let (fixture_item, workspace, window) = workspace_image(&ready, cx).await;
                    let count = workspace.read_with(window, |workspace, cx| {
                        workspace.active_pane().read(cx).items_len()
                    });
                    window
                        .update(|_, cx| {
                            crate::register_asset_binding(
                                ready.fixture.assets.clone(),
                                ready.fixture.input_authorization.clone(),
                                cx,
                            )
                        })
                        .expect("Input-only grant binding");
                    assert!(matches!(
                        workspace.update_in(window, |workspace, window, cx| {
                            crate::open_committed_asset_image_with_scene_configuration(
                                output_request(),
                                &receipt,
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        }),
                        Err(GraphiteAssetBindingError::MissingOutputAuthorization)
                    ));
                    assert!(matches!(
                        workspace.update_in(window, |workspace, window, cx| {
                            crate::open_asset_image_with_scene_configuration(
                                output_request(),
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        }),
                        Err(GraphiteAssetBindingError::InputNamespaceRequired)
                    ));
                    window
                        .update(|_, cx| {
                            crate::register_asset_binding_with_output_reader(
                                ready.fixture.assets.clone(),
                                ready.fixture.input_authorization.clone(),
                                Some(reader.clone()),
                                cx,
                            )
                        })
                        .expect("separately supplied Output Read authority");
                    let mut wrong_digest = output_request();
                    wrong_digest.identity.content_sha256[0] ^= 1;
                    let mut wrong_reference = output_request();
                    wrong_reference.identity.asset_id = "caller-relabelled-source".into();
                    let mut wrong_identity = output_request();
                    wrong_identity.asset_identity = ready.png.asset_identity.clone();
                    for invalid in [wrong_digest, wrong_reference, wrong_identity] {
                        assert!(matches!(
                            workspace.update_in(window, |workspace, window, cx| {
                                crate::open_committed_asset_image_with_scene_configuration(
                                    invalid,
                                    &receipt,
                                    4096,
                                    configuration,
                                    workspace,
                                    window,
                                    cx,
                                )
                            }),
                            Err(GraphiteAssetBindingError::InvalidOutputReceipt)
                        ));
                        assert_eq!(
                            workspace.read_with(window, |workspace, cx| workspace
                                .active_pane()
                                .read(cx)
                                .items_len()),
                            count
                        );
                    }
                    let mut foreign = output_request();
                    foreign.asset_identity.profile_id = "foreign-output-profile".into();
                    assert!(matches!(
                        workspace.update_in(window, |workspace, window, cx| {
                            crate::open_committed_asset_image_with_scene_configuration(
                                foreign,
                                &receipt,
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        }),
                        Err(GraphiteAssetBindingError::ProfileMismatch)
                    ));
                    assert_eq!(
                        workspace.read_with(window, |workspace, cx| workspace
                            .active_pane()
                            .read(cx)
                            .items_len()),
                        count
                    );
                    workspace
                        .update_in(window, |workspace, window, cx| {
                            workspace.active_pane().clone().update(cx, |pane, cx| {
                                pane.close_item_by_id(
                                    fixture_item.entity_id(),
                                    workspace::SaveIntent::Close,
                                    window,
                                    cx,
                                )
                            })
                        })
                        .await
                        .expect("close fixture Item");
                    window.update(|_, _| drop(fixture_item));
                    let item = workspace
                        .update_in(window, |workspace, window, cx| {
                            crate::open_committed_asset_image_with_scene_configuration(
                                output_request(),
                                &receipt,
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        })
                        .expect("genuine receipt opens explicitly configured Item");
                    window.run_until_parked();
                    paint(window);
                    assert!(
                        window
                            .debug_bounds("GRAPHITE-IMPORTED-PREPARE-SCENE")
                            .is_some()
                    );
                    assert!(
                        window
                            .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                            .is_none()
                    );
                    assert!(item.read_with(window, |item, cx| {
                        item.presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .is_none()
                    }));
                    let session = item.read_with(window, |item, _| item.session().clone());
                    let source = session.read_with(window, |session, _| {
                        assert!(session.resource_state().is_fresh());
                        session
                            .resource_state()
                            .committed()
                            .expect("decoded actual Output bytes")
                            .clone()
                    });
                    assert_eq!(
                        source.identity().asset_id.as_ref(),
                        identity.to_reference().expect("receipt reference")
                    );
                    assert_eq!(
                        source.identity().content_sha256,
                        ready.png.image_identity.content_sha256
                    );
                    let input_before = source.image().clone();
                    click_scene(window);
                    let original = presented(&item, window);
                    let bytes = original
                        .as_bytes(0)
                        .expect("Output original frame")
                        .to_vec();
                    assert!(bytes[(3 * 35 + 2) * 4 + 2] > bytes[(3 * 35 + 2) * 4]);
                    item.read_with(window, |item, cx| {
                        let preview = item
                            .presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .expect("configured Output scene");
                        assert_exact_scene_frame(preview, 35, 21);
                        assert_eq!(preview.scene_generation, 1);
                        assert!(preview.result.is_none());
                        assert!(Arc::ptr_eq(&preview.source, &source));
                        assert_eq!(
                            preview.scene.configuration().scale.to_bits(),
                            configuration.scale.to_bits()
                        );
                        assert!(item.session().read(cx).scene_preview_is_current(preview));
                    });
                    let focus = item.read_with(window, |item, cx| item.focus_handle(cx));
                    window.update(|window, cx| focus.dispatch_action(&InvertImage, window, cx));
                    window.run_until_parked();
                    paint(window);
                    assert!(Arc::ptr_eq(&presented(&item, window), &original));
                    click_scene(window);
                    let edited = presented(&item, window);
                    assert_ne!(edited.as_bytes(0).expect("Output edit scene"), bytes);
                    item.read_with(window, |item, cx| {
                        let preview = item
                            .presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .expect("edited Output scene");
                        let result = preview.result.as_ref().expect("original edit result");
                        assert_eq!(result.operation(), CpuImageEdit::Invert);
                        assert_eq!(result.input_identity(), source.identity());
                        assert_eq!(result.input_origin(), source.origin());
                        assert!(Arc::ptr_eq(&preview.source, &source));
                    });
                    assert_eq!(source.image(), &input_before);
                    assert_eq!(receipt, receipt_before);
                    let service = ready.fixture.assets.lock().expect("same canonical owner");
                    assert_eq!(service.record(&identity), Some(before));
                    assert_eq!(
                        service
                            .read_verified(
                                &identity,
                                &reader,
                                &sim_tensor::CancellationToken::default(),
                                16 * 1024
                            )
                            .expect("immutable committed bytes"),
                        ready.png.encoded
                    );
                    drop(service);
                    window
                        .update(|_, cx| crate::clear_asset_binding(cx))
                        .expect("invalidate Output consumer");
                    window.run_until_parked();
                    paint(window);
                    assert!(
                        session
                            .read_with(window, |session, _| session.resource_state().is_revoked())
                    );
                    assert!(item.read_with(window, |item, cx| {
                        item.presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .is_none()
                    }));
                    assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
                    assert!(!window.update(|window, _| window.has_image_atlas_entry(&edited)));
                    assert_eq!(receipt, receipt_before);
                }

                #[gpui::test]
                async fn at_r28_1_root_scene_binding_replacement_revokes_shared_split_without_implicit_requests(
                    cx: &mut TestAppContext,
                ) {
                    let ready = ready_image(cx);
                    let configuration = exact_scene_configuration(35, 21, 1.5);
                    let (fixture_item, workspace, window) = workspace_image(&ready, cx).await;
                    let binding = window
                        .update(|_, cx| {
                            crate::register_asset_binding(
                                ready.fixture.assets.clone(),
                                ready.fixture.input_authorization.clone(),
                                cx,
                            )
                        })
                        .expect("actual Input service registration");
                    workspace
                        .update_in(window, |workspace, window, cx| {
                            workspace.active_pane().clone().update(cx, |pane, cx| {
                                pane.close_item_by_id(
                                    fixture_item.entity_id(),
                                    workspace::SaveIntent::Close,
                                    window,
                                    cx,
                                )
                            })
                        })
                        .await
                        .expect("normal fixture close");
                    window.update(|_, _| drop(fixture_item));
                    let item = workspace
                        .update_in(window, |workspace, window, cx| {
                            crate::open_asset_image_with_scene_configuration(
                                request(&ready.png, [16., 0., 0., 8., 0., 0.]),
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        })
                        .expect("root configured Input Item");
                    window.run_until_parked();
                    paint(window);
                    assert!(item.read_with(window, |item, cx| {
                        item.presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .is_none()
                    }));
                    let split = item
                        .update_in(window, |item, window, cx| {
                            item.clone_on_split(None, window, cx)
                        })
                        .await
                        .expect("existing native Item split");
                    workspace.update_in(window, |workspace, window, cx| {
                        workspace.add_item_to_active_pane(
                            Box::new(split.clone()),
                            None,
                            true,
                            window,
                            cx,
                        )
                    });
                    paint(window);
                    assert!(split.read_with(window, |split, cx| {
                        split
                            .presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .is_none()
                    }));
                    assert!(
                        window
                            .debug_bounds("GRAPHITE-IMPORTED-SCENE-PENDING")
                            .is_none()
                    );
                    assert!(
                        window
                            .debug_bounds("GRAPHITE-IMPORTED-PREPARE-SCENE")
                            .is_some()
                    );
                    click_scene(window);
                    let image = presented(&split, window);
                    assert!(Arc::ptr_eq(&presented(&item, window), &image));
                    let (session, presentation_id, generation) =
                        split.read_with(window, |split, cx| {
                            let preview = split
                                .presentation()
                                .read(cx)
                                .retained_scene_preview()
                                .expect("split button receipt");
                            assert_eq!(preview.scene_generation, 1);
                            (
                                split.session().clone(),
                                split.presentation().entity_id(),
                                preview.scene_generation,
                            )
                        });
                    for _ in 0..2 {
                        paint(window);
                        assert!(Arc::ptr_eq(&presented(&split, window), &image));
                        item.read_with(window, |item, cx| {
                            assert_eq!(item.session().entity_id(), session.entity_id());
                            assert_eq!(item.presentation().entity_id(), presentation_id);
                            let preview = item
                                .presentation()
                                .read(cx)
                                .retained_scene_preview()
                                .expect("one shared receipt");
                            assert_eq!(preview.scene_generation, generation);
                            assert!(item.session().read(cx).scene_preview_is_current(preview));
                            assert_eq!(
                                preview.scene.configuration().viewport.resolution,
                                configuration.viewport.resolution
                            );
                        });
                    }
                    let replacement =
                        host_reader_fixture().expect("same-profile replacement host service");
                    let replacement_png =
                        indexed_request_png(&replacement, 4).expect("actual replacement producer");
                    assert_eq!(
                        replacement.input_authorization.profile_id(),
                        ready.fixture.input_authorization.profile_id()
                    );
                    assert!(!Arc::ptr_eq(&replacement.assets, &ready.fixture.assets));
                    let successor = window
                        .update(|_, cx| {
                            crate::register_asset_binding(
                                replacement.assets.clone(),
                                replacement.input_authorization.clone(),
                                cx,
                            )
                        })
                        .expect("exact service replacement invalidates old leases");
                    assert_ne!(successor.entity_id(), binding.entity_id());
                    window.run_until_parked();
                    paint(window);
                    assert!(!binding.read_with(window, |binding, _| binding.is_active()));
                    assert!(
                        session
                            .read_with(window, |session, _| session.resource_state().is_revoked())
                    );
                    for retired in [&item, &split] {
                        retired.read_with(window, |item, cx| {
                            assert!(item.presentation().read(cx).image().is_none());
                            assert!(
                                item.presentation()
                                    .read(cx)
                                    .retained_scene_preview()
                                    .is_none()
                            );
                        });
                    }
                    assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
                    assert!(!window.update(|window, _| window.has_image_atlas_entry(&image)));
                    let recovered = workspace
                        .update_in(window, |workspace, window, cx| {
                            crate::open_asset_image_with_scene_configuration(
                                request(&replacement_png, [16., 0., 0., 8., 0., 0.]),
                                4096,
                                configuration,
                                workspace,
                                window,
                                cx,
                            )
                        })
                        .expect("explicit request through new current binding");
                    window.run_until_parked();
                    paint(window);
                    assert!(recovered.read_with(window, |item, cx| {
                        item.presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .is_none()
                    }));
                    click_scene(window);
                    let recovered_image = presented(&recovered, window);
                    assert!(!Arc::ptr_eq(&recovered_image, &image));
                    assert_eq!(
                        recovered_image.as_bytes(0).expect("new source scene"),
                        image.as_bytes(0).expect("retained immutable old frame")
                    );
                    recovered.read_with(window, |item, cx| {
                        let preview = item
                            .presentation()
                            .read(cx)
                            .retained_scene_preview()
                            .expect("new binding accepted scene");
                        assert_eq!(preview.scene_generation, 1);
                        assert!(item.session().read(cx).scene_preview_is_current(preview));
                        assert_eq!(preview.source.identity(), &replacement_png.image_identity);
                    });
                    assert!(
                        session
                            .read_with(window, |session, _| session.resource_state().is_revoked())
                    );
                }
            }
        }
    }
}

pub(crate) mod artboard_workspace_controls {
    use super::*;
    use crate::graph::GraphiteArtboardItem;
    use crate::session::GraphiteArtboardSession;
    use gpui::{DevicePixels, Entity, VisualTestContext, size};
    use graphics_core::evaluate::{
        CpuSvgArtboardDocument, CpuSvgArtboardParameters, CpuSvgEvaluationBudget,
    };
    use graphite_core_types::Color;
    use graphite_graph_craft::document::{NodeId, value::DVec2};
    use graphite_std::application_io::{ExportFormat, RenderConfig, TimingInformation};
    use graphite_std::transform::Footprint;
    use graphite_std::vector_types::vector::style::RenderMode;
    use std::sync::Arc;
    use workspace::item::Item as _;

    const PIXEL_BYTES: usize = 31 * 17 * 4;

    pub(crate) fn parameters() -> CpuSvgArtboardParameters {
        CpuSvgArtboardParameters {
            location: DVec2::new(2., 2.),
            dimensions: DVec2::new(6., 4.),
            background: Color::from_gamma_srgb_channels(1., 0., 0., 0.5),
            clip: true,
        }
    }

    fn configuration() -> RenderConfig {
        RenderConfig {
            viewport: Footprint {
                transform: graphite_glam::DAffine2::from_translation(DVec2::ONE),
                resolution: graphite_glam::UVec2::new(31, 17),
                quality: Footprint::DEFAULT.quality,
            },
            scale: 1.,
            time: TimingInformation {
                time: 0.,
                animation_time: std::time::Duration::ZERO,
            },
            pointer: DVec2::ZERO,
            render_mode: RenderMode::Normal,
            export_format: ExportFormat::Svg,
            for_export: true,
            for_eyedropper: false,
        }
    }

    pub(crate) async fn native_artboard(
        cx: &mut TestAppContext,
        initial: CpuSvgArtboardParameters,
        max_preview_bytes: usize,
    ) -> (
        Entity<GraphiteArtboardItem>,
        Entity<workspace::Workspace>,
        &mut VisualTestContext,
    ) {
        init(cx);
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
            crate::graph::init(cx);
        });
        let document = CpuSvgArtboardDocument::new(
            91,
            NodeId(7001),
            initial,
            configuration(),
            CpuSvgEvaluationBudget {
                max_viewport_pixels: 31 * 17,
                max_output_bytes: 64 * 1024,
            },
        )
        .expect("genuine source template and explicit caller configuration");
        let session = cx.new(|_| GraphiteArtboardSession::new(document));
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (host, window) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = host.read_with(window, |host, _| host.workspace().clone());
        let item = workspace.update_in(window, |workspace, window, cx| {
            let item = cx.new(|cx| {
                GraphiteArtboardItem::from_session(session, max_preview_bytes, window, cx)
            });
            workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
            item.focus_handle(cx).focus(window, cx);
            item
        });
        window.update(|_, cx| {
            let bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/default-linux.json",
                cx,
            )
            .expect("actual application keymap");
            cx.bind_keys(bindings);
        });
        window.simulate_resize(gpui::size(gpui::px(1400.), gpui::px(1400.)));
        window.run_until_parked();
        paint(window);
        (item, workspace, window)
    }

    pub(crate) fn paint(window: &mut VisualTestContext) {
        window.update(|window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        });
    }

    pub(crate) fn click(window: &mut VisualTestContext, selector: &'static str) {
        let position = window
            .debug_bounds(selector)
            .expect("actual visible control")
            .center();
        window.simulate_click(position, gpui::Modifiers::default());
        window.run_until_parked();
        paint(window);
    }

    fn begin_click(window: &mut VisualTestContext, selector: &'static str) {
        let position = window
            .debug_bounds(selector)
            .expect("actual native control")
            .center();
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position,
                    button: gpui::MouseButton::Left,
                    modifiers: gpui::Modifiers::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
                cx,
            );
            window.dispatch_event(
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position,
                    button: gpui::MouseButton::Left,
                    modifiers: gpui::Modifiers::default(),
                    click_count: 1,
                }),
                cx,
            );
        });
    }

    fn edit(
        item: &Entity<GraphiteArtboardItem>,
        window: &mut VisualTestContext,
        index: usize,
        text: &str,
    ) -> Entity<editor::Editor> {
        let editor = item.read_with(window, |item, _| item.artboard_editors[index].clone());
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input(text);
        window.run_until_parked();
        paint(window);
        editor
    }

    fn commit(window: &mut VisualTestContext) {
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
    }

    fn presented(
        item: &Entity<GraphiteArtboardItem>,
        window: &VisualTestContext,
    ) -> Arc<gpui::RenderImage> {
        item.read_with(window, |item, cx| {
            item.presentation()
                .read(cx)
                .image()
                .expect("actual retained frame")
                .clone()
        })
    }

    fn pixel(image: &gpui::RenderImage, x: usize, y: usize) -> [u8; 4] {
        let width = image.size(0).width.0 as usize;
        image
            .as_bytes(0)
            .expect("native single frame")
            .get((y * width + x) * 4..(y * width + x + 1) * 4)
            .expect("source-derived interior")
            .try_into()
            .expect("BGRA pixel")
    }

    fn assert_pixel(image: &gpui::RenderImage, expected: [u8; 4]) {
        for (actual, expected) in pixel(image, 5, 5).into_iter().zip(expected) {
            assert!(
                actual.abs_diff(expected) <= 1,
                "native interior {actual} versus {expected}"
            );
        }
    }

    #[gpui::test]
    async fn at_r09_3_artboard_workspace_explicit_prepare_original_pixels_and_metadata(
        cx: &mut TestAppContext,
    ) {
        let (item, workspace, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
        let session = item.read_with(window, |item, _| item.session().clone());
        let network_before =
            session.read_with(window, |session, _| session.document().network().clone());
        assert_eq!(
            workspace.read_with(window, |workspace, cx| {
                workspace
                    .active_item_as::<GraphiteArtboardItem>(cx)
                    .expect("Pane owns Artboard Item")
                    .entity_id()
            }),
            item.entity_id()
        );
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-EMPTY").is_some());
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PREPARE").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        session.read_with(window, |session, _| {
            assert!(!session.is_pending());
            assert!(session.preview().is_none());
            assert!(session.current_output().is_none());
            assert_eq!(session.revision(), 0);
        });
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let image = presented(&item, window);
        assert_eq!(image.size(0), size(DevicePixels(31), DevicePixels(17)));
        assert_eq!(image.as_bytes(0).expect("frame bytes").len(), PIXEL_BYTES);
        assert_pixel(&image, [0, 0, 255, 128]);
        assert_eq!(pixel(&image, 0, 0), [0, 0, 0, 0]);
        assert_eq!(pixel(&image, 30, 16), [0, 0, 0, 0]);
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PENDING").is_none());
        let preview = session.read_with(window, |session, _| {
            let preview = session
                .preview()
                .expect("genuine successful receipt")
                .clone();
            assert!(session.preview_is_current(&preview));
            assert!(session.document().owns_current_output(preview.output()));
            assert!(!session.is_pending());
            assert_eq!(session.parameters(), parameters());
            assert_eq!(session.configuration(), configuration());
            assert_eq!(session.document().network(), &network_before);
            assert_eq!(preview.output().output().metadata.backgrounds.len(), 1);
            assert_eq!(
                preview.output().output().metadata.backgrounds[0].location,
                parameters().location
            );
            assert_eq!(
                preview.output().output().metadata.backgrounds[0].dimensions,
                parameters().dimensions
            );
            assert!(
                preview
                    .output()
                    .output()
                    .metadata
                    .click_targets
                    .contains_key(&NodeId(7001))
            );
            assert!(
                preview
                    .output()
                    .output()
                    .metadata
                    .clip_targets
                    .contains(&NodeId(7001))
            );
            assert!(Arc::ptr_eq(preview.image(), &image));
            preview
        });
        let output_before = preview.output().output().clone();
        for _ in 0..2 {
            paint(window);
            assert!(Arc::ptr_eq(&presented(&item, window), &image));
            assert!(session.read_with(window, |session, _| session.preview_is_current(&preview)));
            assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
        }
        assert_eq!(preview.output().output(), &output_before);
    }

    #[gpui::test]
    async fn at_r09_4_artboard_native_integer_fields_draft_commit_escape_and_finite_recovery(
        cx: &mut TestAppContext,
    ) {
        let mut initial = parameters();
        initial.location.x = 2.25;
        initial.dimensions = DVec2::new(6.5, 4.25);
        let (item, _, window) = native_artboard(cx, initial, PIXEL_BYTES).await;
        let session = item.read_with(window, |item, _| item.session().clone());
        let width = item.read_with(window, |item, _| item.artboard_editors[2].clone());
        assert_eq!(width.read_with(window, |editor, cx| editor.text(cx)), "6.5");
        let network_before =
            session.read_with(window, |session, _| session.document().network().clone());
        let x = edit(&item, window, 0, "1 + 2.4");
        assert_eq!(
            session.read_with(window, |session, _| session.parameters()),
            initial
        );
        assert_eq!(
            session.read_with(window, |session, _| session.revision()),
            0
        );
        commit(window);
        let mut expected = initial;
        expected.location.x = 3.;
        assert_eq!(
            session.read_with(window, |session, _| session.parameters()),
            expected
        );
        assert_eq!(
            session.read_with(window, |session, _| session.revision()),
            1
        );
        assert_eq!(width.read_with(window, |editor, cx| editor.text(cx)), "6.5");
        assert!(!window.update(|window, cx| x.read(cx).is_focused(window)));
        assert!(window.update(|window, cx| item.read(cx).focus_handle(cx).is_focused(window)));
        paint(window);
        assert_eq!(
            session.read_with(window, |session, _| session.revision()),
            1
        );
        let committed_network =
            session.read_with(window, |session, _| session.document().network().clone());
        assert_ne!(committed_network, network_before);
        let x = edit(&item, window, 0, "not a number");
        commit(window);
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
        assert_eq!(
            session.read_with(window, |session, _| session.parameters()),
            expected
        );
        assert_eq!(
            session.read_with(window, |session, _| session.revision()),
            1
        );
        assert_eq!(
            session.read_with(window, |session, _| session.document().network().clone()),
            committed_network
        );
        x.update_in(window, |editor, window, cx| {
            editor.focus_handle(cx).focus(window, cx)
        });
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        assert_eq!(x.read_with(window, |editor, cx| editor.text(cx)), "3");
        assert!(!window.update(|window, cx| x.read(cx).is_focused(window)));
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
        edit(&item, window, 2, "1e30");
        commit(window);
        assert_eq!(
            session.read_with(window, |session, _| session.parameters().dimensions.x),
            9_007_199_254_740_992.
        );
        edit(&item, window, 2, "-3");
        commit(window);
        edit(&item, window, 3, "0");
        commit(window);
        expected.dimensions = DVec2::new(-3., 0.);
        assert_eq!(
            session.read_with(window, |session, _| session.parameters()),
            expected
        );
        assert_eq!(
            session.read_with(window, |session, _| session.configuration()),
            configuration()
        );
        let revision = session.read_with(window, |session, _| session.revision());
        edit(&item, window, 4, "NaN, 0, 0, 1");
        commit(window);
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
        assert_eq!(
            session.read_with(window, |session, _| session.revision()),
            revision
        );
        assert_eq!(
            session.read_with(window, |session, _| session.parameters()),
            expected
        );
        edit(&item, window, 4, "0, 0, 1, 0.5");
        commit(window);
        expected.background = Color::from_gamma_srgb_channels(0., 0., 1., 0.5);
        assert_eq!(
            session.read_with(window, |session, _| session.parameters()),
            expected
        );
        assert_eq!(
            session.read_with(window, |session, _| session.revision()),
            revision + 1
        );
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
    }

    #[gpui::test]
    async fn at_r20_2_artboard_native_background_clip_edits_preserve_source_and_reprepare(
        cx: &mut TestAppContext,
    ) {
        let (item, _, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
        let session = item.read_with(window, |item, _| item.session().clone());
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let original = presented(&item, window);
        let old_preview = session.read_with(window, |session, _| {
            session.preview().expect("old receipt").clone()
        });
        let old_output = old_preview.output().output().clone();
        let old_network =
            session.read_with(window, |session, _| session.document().network().clone());
        edit(&item, window, 4, "0, 0, 1, 0.5");
        commit(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_some()
        );
        assert!(Arc::ptr_eq(&presented(&item, window), &original));
        assert!(!session.read_with(window, |session, _| {
            session.preview_is_current(&old_preview)
        }));
        assert!(!session.read_with(window, |session, _| session.is_pending()));
        let revision = session.read_with(window, |session, _| session.revision());
        click(window, "GRAPHITE-ARTBOARD-CLIP");
        assert_eq!(
            session.read_with(window, |session, _| session.revision()),
            revision + 1
        );
        session.read_with(window, |session, _| {
            assert!(!session.parameters().clip);
            assert_eq!(session.parameters().location, parameters().location);
            assert_eq!(session.parameters().dimensions, parameters().dimensions);
            assert_eq!(session.configuration(), configuration());
        });
        assert!(Arc::ptr_eq(&presented(&item, window), &original));
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let blue = presented(&item, window);
        assert!(!Arc::ptr_eq(&blue, &original));
        assert_ne!(blue.as_bytes(0), original.as_bytes(0));
        assert_pixel(&blue, [255, 0, 0, 128]);
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_none()
        );
        session.read_with(window, |session, _| {
            let preview = session.preview().expect("new source result");
            assert!(session.preview_is_current(preview));
            assert!(
                !preview
                    .output()
                    .output()
                    .metadata
                    .clip_targets
                    .contains(&NodeId(7001))
            );
            assert_eq!(preview.output().revision(), revision + 1);
            assert_eq!(preview.output().configuration(), configuration());
            assert_ne!(session.document().network(), &old_network);
        });
        assert_eq!(old_preview.output().output(), &old_output);
        assert_pixel(&original, [0, 0, 255, 128]);
    }

    #[gpui::test]
    async fn at_r20_2_artboard_native_cap_failure_recovery_pending_cancel_and_retry(
        cx: &mut TestAppContext,
    ) {
        let (item, _, window) = native_artboard(cx, parameters(), PIXEL_BYTES - 1).await;
        let session = item.read_with(window, |item, _| item.session().clone());
        let network = session.read_with(window, |session, _| session.document().network().clone());
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        session.read_with(window, |session, _| {
            assert!(!session.is_pending());
            assert!(matches!(session.diagnostic(), Some(crate::session::ArtboardSessionError::Conversion(error)) if matches!(error.as_ref(), crate::render::CpuSvgPreviewError::PreviewBudgetExceeded { .. })));
            assert_eq!(session.revision(), 0);
            assert_eq!(session.parameters(), parameters());
            assert_eq!(session.document().network(), &network);
        });
        let mut recovery = configuration();
        recovery.viewport.resolution.x = 30;
        session
            .update(window, |session, cx| {
                session.apply(session.revision(), session.parameters(), recovery, cx)
            })
            .expect("explicit smaller caller target, no inferred window policy");
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let recovered = presented(&item, window);
        assert_eq!(recovered.size(0), size(DevicePixels(30), DevicePixels(17)));
        assert_pixel(&recovered, [0, 0, 255, 128]);
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
        let preview = session.read_with(window, |session, _| {
            session.preview().expect("recovered").clone()
        });
        begin_click(window, "GRAPHITE-ARTBOARD-PREPARE");
        assert!(session.read_with(window, |session, _| session.is_pending()));
        begin_click(window, "GRAPHITE-ARTBOARD-CANCEL");
        assert!(!session.read_with(window, |session, _| session.is_pending()));
        window.run_until_parked();
        paint(window);
        assert!(!session.read_with(window, |session, _| session.preview_is_current(&preview)));
        assert!(Arc::ptr_eq(&presented(&item, window), &recovered));
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_some()
        );
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
        click(window, "GRAPHITE-ARTBOARD-CANCEL");
        assert!(Arc::ptr_eq(&presented(&item, window), &recovered));
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let retry = presented(&item, window);
        assert!(!Arc::ptr_eq(&retry, &recovered));
        assert_eq!(retry.as_bytes(0), recovered.as_bytes(0));
        assert!(session.read_with(window, |session, _| {
            session.preview_is_current(session.preview().expect("retry receipt"))
        }));
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        assert_eq!(
            session.read_with(window, |session, _| session.configuration()),
            recovery
        );
    }

    #[gpui::test]
    async fn at_r09_3_artboard_workspace_split_local_drafts_shared_preview_and_native_final_release(
        cx: &mut TestAppContext,
    ) {
        let mut second_context = cx.clone();
        let (item, workspace, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
        let weak_session = item.read_with(window, |item, _| item.session().downgrade());
        let (project, app_state) = workspace.read_with(window, |workspace, _| {
            (workspace.project().clone(), workspace.app_state().clone())
        });
        let (second_host, second_window) = second_context.add_window_view(|window, cx| {
            let workspace =
                cx.new(|cx| workspace::Workspace::new(None, project, app_state, window, cx));
            workspace::MultiWorkspace::new(workspace, window, cx)
        });
        second_window.update(|_, cx| {
            let bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/default-linux.json",
                cx,
            )
            .expect("actual default application keymap");
            cx.bind_keys(bindings);
        });
        let second_workspace =
            second_host.read_with(second_window, |host, _| host.workspace().clone());
        let split = item
            .update_in(second_window, |item, window, cx| {
                item.clone_on_split(None, window, cx)
            })
            .await
            .expect("native split");
        second_workspace.update_in(second_window, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx);
            split.focus_handle(cx).focus(window, cx);
        });
        second_window.simulate_resize(gpui::size(gpui::px(1400.), gpui::px(1400.)));
        second_window.run_until_parked();
        paint(window);
        paint(second_window);
        let (session_id, presentation_id, focus, weak_presentation, first_editor) =
            item.read_with(window, |item, cx| {
                (
                    item.session().entity_id(),
                    item.presentation().entity_id(),
                    item.focus_handle(cx),
                    item.presentation().downgrade(),
                    item.artboard_editors[0].entity_id(),
                )
            });
        split.read_with(second_window, |split, cx| {
            assert_eq!(split.session().entity_id(), session_id);
            assert_eq!(split.presentation().entity_id(), presentation_id);
            assert_ne!(split.focus_handle(cx), focus);
            assert_ne!(split.artboard_editors[0].entity_id(), first_editor);
            assert!(split.session().read(cx).preview().is_none());
            assert!(!split.session().read(cx).is_pending());
        });
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        paint(second_window);
        let image = presented(&item, window);
        let bytes = image.as_bytes(0).expect("shared frame").to_vec();
        let generation = item.read_with(window, |item, cx| {
            item.session()
                .read(cx)
                .preview()
                .expect("receipt")
                .generation()
        });
        assert!(Arc::ptr_eq(&presented(&split, second_window), &image));
        let split_editor = edit(&split, second_window, 0, "invalid local draft");
        assert!(Arc::ptr_eq(&presented(&split, second_window), &image));
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        assert!(
            second_window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_some()
        );
        for _ in 0..2 {
            paint(second_window);
            paint(window);
            item.read_with(window, |item, cx| {
                let session = item.session().read(cx);
                assert_eq!(
                    session.preview().expect("same receipt").generation(),
                    generation
                );
                assert!(session.preview_is_current(session.preview().expect("same receipt")));
                assert!(item.presentation().read(cx).is_fresh());
                assert_eq!(session.parameters(), parameters());
            });
        }
        edit(&item, window, 0, "4");
        commit(window);
        assert_eq!(
            split_editor.read_with(second_window, |editor, cx| editor.text(cx)),
            "invalid local draft"
        );
        second_window.simulate_keystrokes("escape");
        second_window.run_until_parked();
        paint(second_window);
        assert_eq!(
            split_editor.read_with(second_window, |editor, cx| editor.text(cx)),
            "4"
        );
        assert!(!second_window.update(|window, cx| split_editor.read(cx).is_focused(window)));
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        paint(second_window);
        let replacement = presented(&item, window);
        assert!(!Arc::ptr_eq(&replacement, &image));
        assert!(Arc::ptr_eq(&presented(&split, second_window), &replacement));
        assert!(window.update(|window, _| window.has_image_atlas_entry(&replacement)));
        assert!(second_window.update(|window, _| window.has_image_atlas_entry(&replacement)));
        let weak_item = item.downgrade();
        let first_id = item.entity_id();
        workspace
            .update_in(window, |workspace, window, cx| {
                workspace.active_pane().clone().update(cx, |pane, cx| {
                    pane.close_item_by_id(first_id, workspace::SaveIntent::Close, window, cx)
                })
            })
            .await
            .expect("actual native first close clears Toolbar owner");
        window.update(|_, _| drop(item));
        paint(window);
        window.run_until_parked();
        paint(second_window);
        assert!(weak_item.upgrade().is_none());
        assert!(weak_session.upgrade().is_some());
        assert!(weak_presentation.upgrade().is_some());
        assert!(Arc::ptr_eq(&presented(&split, second_window), &replacement));
        let split_id = split.entity_id();
        second_workspace
            .update_in(second_window, |workspace, window, cx| {
                workspace.active_pane().clone().update(cx, |pane, cx| {
                    pane.close_item_by_id(split_id, workspace::SaveIntent::Close, window, cx)
                })
            })
            .await
            .expect("actual native final close");
        second_window.update(|_, _| drop(split));
        paint(window);
        paint(second_window);
        second_window.run_until_parked();
        window.run_until_parked();
        assert!(weak_session.upgrade().is_none());
        assert!(weak_presentation.upgrade().is_none());
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&replacement)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&replacement)));
        assert_eq!(image.as_bytes(0).expect("immutable previous frame"), bytes);
    }
    #[gpui::test]
    async fn at_r09_3_root_artboard_opener_preserves_supplied_document_and_prepares_visible_original_frame(
        cx: &mut TestAppContext,
    ) {
        let (fixture_item, workspace, window) =
            native_artboard(cx, parameters(), PIXEL_BYTES).await;
        let fixture_id = fixture_item.entity_id();
        workspace
            .update_in(window, |workspace, window, cx| {
                workspace.active_pane().clone().update(cx, |pane, cx| {
                    pane.close_item_by_id(fixture_id, workspace::SaveIntent::Close, window, cx)
                })
            })
            .await
            .expect("close existing fixture Item through native Pane");
        window.update(|_, _| drop(fixture_item));
        paint(window);
        window.run_until_parked();
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_pane()
                .read(cx)
                .items_len()),
            0
        );
        let mut caller_configuration = configuration();
        caller_configuration.time.time = 3.25;
        caller_configuration.pointer = DVec2::new(4., 6.);
        let document = CpuSvgArtboardDocument::new(
            439,
            NodeId(7001),
            parameters(),
            caller_configuration,
            CpuSvgEvaluationBudget {
                max_viewport_pixels: 31 * 17,
                max_output_bytes: 64 * 1024,
            },
        )
        .expect("caller supplies genuine original document and explicit context");
        let network_before = document.network().clone();
        let parameters_before = document.parameters();
        let revision_before = document.revision();
        let document_id = document.document_id_encoding();
        let node_id = document.node_id();
        let item = workspace.update_in(window, |workspace, window, cx| {
            crate::open_artboard_document(document, PIXEL_BYTES, workspace, window, cx)
        });
        window.run_until_parked();
        paint(window);
        assert_eq!(
            workspace.read_with(window, |workspace, cx| {
                workspace
                    .active_item_as::<GraphiteArtboardItem>(cx)
                    .expect("root opened active Item")
                    .entity_id()
            }),
            item.entity_id()
        );
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_pane()
                .read(cx)
                .items_len()),
            1
        );
        assert!(window.update(|window, cx| item.read(cx).focus_handle(cx).is_focused(window)));
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PREPARE").is_some());
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-EMPTY").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        let session = item.read_with(window, |item, _| item.session().clone());
        session.read_with(window, |session, _| {
            assert!(!session.is_pending());
            assert!(session.preview().is_none());
            assert!(session.current_output().is_none());
            assert_eq!(session.revision(), revision_before);
            assert_eq!(session.parameters(), parameters_before);
            assert_eq!(session.configuration(), caller_configuration);
            assert_eq!(session.document().document_id_encoding(), document_id);
            assert_eq!(session.document().node_id(), node_id);
            assert_eq!(session.document().network(), &network_before);
        });
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let image = presented(&item, window);
        assert_eq!(image.size(0), size(DevicePixels(31), DevicePixels(17)));
        assert_eq!(
            image.as_bytes(0).expect("native exact frame").len(),
            PIXEL_BYTES
        );
        assert_pixel(&image, [0, 0, 255, 128]);
        assert_eq!(pixel(&image, 0, 0), [0, 0, 0, 0]);
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PENDING").is_none());
        session.read_with(window, |session, _| {
            let preview = session.preview().expect("genuine compiled accepted result");
            assert!(session.preview_is_current(preview));
            assert!(session.document().owns_current_output(preview.output()));
            assert!(Arc::ptr_eq(preview.image(), &image));
            assert_eq!(preview.output().document_id_encoding(), document_id);
            assert_eq!(preview.output().node_id(), node_id);
            assert_eq!(preview.output().revision(), revision_before);
            assert_eq!(preview.output().configuration(), caller_configuration);
            assert_eq!(preview.output().output().metadata.backgrounds.len(), 1);
            assert_eq!(
                preview.output().output().metadata.backgrounds[0].location,
                parameters_before.location
            );
            assert_eq!(
                preview.output().output().metadata.backgrounds[0].dimensions,
                parameters_before.dimensions
            );
            assert!(
                preview
                    .output()
                    .output()
                    .metadata
                    .click_targets
                    .contains_key(&node_id)
            );
            assert!(
                preview
                    .output()
                    .output()
                    .metadata
                    .clip_targets
                    .contains(&node_id)
            );
            assert_eq!(session.parameters(), parameters_before);
            assert_eq!(session.configuration(), caller_configuration);
            assert_eq!(session.document().network(), &network_before);
        });
    }
    #[gpui::test]
    async fn at_r22_1_artboard_workspace_png_button_prepares_visible_exact_frame(
        cx: &mut TestAppContext,
    ) {
        use sha2::Digest as _;
        let (item, _, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PREPARE-PNG")
                .is_none()
        );
        let session = item.read_with(window, |item, _| item.session().clone());
        let network = session.read_with(window, |session, _| session.document().network().clone());
        item.update(window, |item, cx| {
            item.configure_png_limits(PIXEL_BYTES, 64 * 1024, cx)
        });
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PREPARE-PNG")
                .is_some()
        );
        assert!(item.read_with(window, |item, _| item.prepared_png().is_none()));
        assert!(!item.read_with(window, |item, _| item.png_is_pending()));
        assert!(session.read_with(window, |session, _| session.preview().is_none()));
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let image = presented(&item, window);
        assert_pixel(&image, [0, 0, 255, 128]);
        let preview = session.read_with(window, |session, _| {
            session.preview().expect("current preview").clone()
        });
        let output = preview.output().output().clone();
        let output_lease = preview.output().output_lease();
        begin_click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        assert!(item.read_with(window, |item, _| item.png_is_pending()));
        assert!(item.read_with(window, |item, _| item.prepared_png().is_none()));
        window.run_until_parked();
        paint(window);
        assert!(!item.read_with(window, |item, _| item.png_is_pending()));
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                .is_some()
        );
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PNG-ERROR").is_none());
        let prepared = item.read_with(window, |item, _| {
            item.prepared_png().expect("button result").clone()
        });
        assert!(session.read_with(window, |session, _| session.png_is_current(&prepared)));
        let png = prepared.png();
        assert_eq!((png.width(), png.height()), (31, 17));
        assert_eq!(
            png.sample_format(),
            graphics_core::resources::CpuSampleFormat::Rgba8
        );
        assert_eq!(
            png.color_space(),
            graphics_core::resources::ImageColorSpace::Srgb
        );
        assert_eq!(
            png.alpha(),
            graphics_core::resources::AlphaAssociation::Straight
        );
        assert_eq!(png.content_type(), "image/png");
        let rows: Vec<u8> = image
            .as_bytes(0)
            .expect("actual frame")
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[2], pixel[1], pixel[0], pixel[3]])
            .collect();
        let decoded =
            image::load_from_memory_with_format(png.encoded_bytes(), image::ImageFormat::Png)
                .expect("actual native PNG")
                .into_rgba8();
        assert_eq!(decoded.dimensions(), (31, 17));
        assert_eq!(decoded.as_raw(), &rows);
        let row_digest: [u8; 32] = sha2::Sha256::digest(&rows).into();
        let png_digest: [u8; 32] = sha2::Sha256::digest(png.encoded_bytes()).into();
        assert_eq!(prepared.pixels_sha256(), row_digest);
        assert_eq!(png.content_sha256(), png_digest);
        assert_ne!(row_digest, png_digest);
        assert!(Arc::ptr_eq(
            &prepared.source_output().output_lease(),
            &output_lease
        ));
        assert_eq!(prepared.source_output().output(), &output);
        assert_eq!(prepared.source_output().output().metadata, output.metadata);
        assert_eq!(prepared.source_output().configuration(), configuration());
        assert_eq!(prepared.source_output().document_id_encoding(), 91);
        assert_eq!(prepared.source_output().node_id(), NodeId(7001));
        assert_eq!(prepared.source_output().revision(), 0);
        assert!(Arc::ptr_eq(&presented(&item, window), &image));
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let repeated = item.read_with(window, |item, _| {
            item.prepared_png().expect("repeat PNG").clone()
        });
        assert!(!session.read_with(window, |session, _| session.png_is_current(&prepared)));
        assert!(session.read_with(window, |session, _| session.png_is_current(&repeated)));
        assert_eq!(repeated.png().encoded_bytes(), png.encoded_bytes());
        session.read_with(window, |session, _| {
            assert_eq!(session.document().network(), &network);
            assert_eq!(session.parameters(), parameters());
            assert_eq!(session.revision(), 0);
            assert!(session.preview_is_current(&preview));
        });
    }

    #[gpui::test]
    async fn at_r22_1_artboard_workspace_png_limits_error_cancel_and_retry_keep_preview(
        cx: &mut TestAppContext,
    ) {
        let (item, _, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
        let session = item.read_with(window, |item, _| item.session().clone());
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let image = presented(&item, window);
        let preview = session.read_with(window, |session, _| {
            session.preview().expect("genuine frame").clone()
        });
        let network = session.read_with(window, |session, _| session.document().network().clone());
        for (pixel_limit, encoded_limit) in [
            (0, 64 * 1024),
            (PIXEL_BYTES - 1, 64 * 1024),
            (PIXEL_BYTES, 0),
        ] {
            item.update(window, |item, cx| {
                item.configure_png_limits(pixel_limit, encoded_limit, cx)
            });
            paint(window);
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PNG-ERROR").is_some());
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                    .is_none()
            );
            assert!(!item.read_with(window, |item, _| item.png_is_pending()));
            assert!(item.read_with(window, |item, _| item.prepared_png().is_none()));
            assert!(Arc::ptr_eq(&presented(&item, window), &image));
            assert!(session.read_with(window, |session, _| session.preview_is_current(&preview)));
            assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        }
        item.update(window, |item, cx| {
            item.configure_png_limits(PIXEL_BYTES, 64 * 1024, cx)
        });
        paint(window);
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let first = item.read_with(window, |item, _| {
            item.prepared_png().expect("recovery").clone()
        });
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PNG-ERROR").is_none());
        let encoded_size = first.png().encoded_bytes().len();
        item.update(window, |item, cx| {
            item.configure_png_limits(PIXEL_BYTES, encoded_size - 1, cx)
        });
        paint(window);
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PNG-ERROR").is_some());
        assert!(Arc::ptr_eq(&presented(&item, window), &image));
        item.update(window, |item, cx| {
            item.configure_png_limits(PIXEL_BYTES, encoded_size, cx)
        });
        paint(window);
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let exact = item.read_with(window, |item, _| {
            item.prepared_png().expect("exact limits").clone()
        });
        assert_eq!(exact.png().encoded_bytes(), first.png().encoded_bytes());
        click(window, "GRAPHITE-ARTBOARD-CANCEL-PNG");
        assert!(!session.read_with(window, |session, _| session.png_is_current(&exact)));
        assert!(session.read_with(window, |session, _| session.preview_is_current(&preview)));
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                .is_some()
        );
        assert!(Arc::ptr_eq(&presented(&item, window), &image));
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let exact = item.read_with(window, |item, _| {
            item.prepared_png()
                .expect("PNG-only cancellation recovery")
                .clone()
        });
        assert!(session.read_with(window, |session, _| session.png_is_current(&exact)));
        assert_eq!(exact.png().encoded_bytes(), first.png().encoded_bytes());
        begin_click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        assert!(item.read_with(window, |item, _| item.png_is_pending()));
        begin_click(window, "GRAPHITE-ARTBOARD-CANCEL-PNG");
        assert!(!item.read_with(window, |item, _| item.png_is_pending()));
        assert!(session.read_with(window, |session, _| session.preview_is_current(&preview)));
        assert!(!session.read_with(window, |session, _| session.png_is_current(&exact)));
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PNG-ERROR").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                .is_some()
        );
        assert!(Arc::ptr_eq(&presented(&item, window), &image));
        assert_eq!(
            item.read_with(window, |item, _| item
                .prepared_png()
                .expect("retained old bytes")
                .png()
                .encoded_bytes()
                .to_vec()),
            exact.png().encoded_bytes()
        );
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PNG-ERROR").is_none());
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                .is_some()
        );
        session.read_with(window, |session, _| {
            assert_eq!(session.document().network(), &network);
            assert_eq!(session.parameters(), parameters());
            assert_eq!(session.revision(), 0);
            assert!(session.preview_is_current(&preview));
        });
    }

    #[gpui::test]
    async fn at_r22_1_artboard_workspace_png_edit_split_and_pending_release(
        cx: &mut TestAppContext,
    ) {
        let (item, workspace, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
        item.update(window, |item, cx| {
            item.configure_png_limits(PIXEL_BYTES, 64 * 1024, cx)
        });
        paint(window);
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let original = item.read_with(window, |item, _| {
            item.prepared_png().expect("source PNG").clone()
        });
        let image = presented(&item, window);
        let session = item.read_with(window, |item, _| item.session().clone());
        edit(&item, window, 4, "0,0,1,0.75");
        commit(window);
        assert!(!session.read_with(window, |session, _| session.png_is_current(&original)));
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_some()
        );
        assert!(Arc::ptr_eq(&presented(&item, window), &image));
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let edited_image = presented(&item, window);
        assert_pixel(&edited_image, [255, 0, 0, 191]);
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let edited = item.read_with(window, |item, _| {
            item.prepared_png().expect("edited PNG").clone()
        });
        assert!(session.read_with(window, |session, _| session.png_is_current(&edited)));
        assert_ne!(
            original.png().content_sha256(),
            edited.png().content_sha256()
        );
        assert_ne!(original.pixels_sha256(), edited.pixels_sha256());
        assert_eq!(original.source_output().revision(), 0);
        assert_eq!(edited.source_output().revision(), 1);
        assert_eq!(
            original.source_output().configuration(),
            edited.source_output().configuration()
        );
        let generation = session.read_with(window, |session, _| {
            session.preview().expect("current frame").generation()
        });
        begin_click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        assert!(item.read_with(window, |item, _| item.png_is_pending()));
        let clone = item.update_in(window, |item, window, cx| {
            item.clone_on_split(None, window, cx)
        });
        let split = clone.await.expect("same owner split");
        workspace.update_in(window, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx);
        });
        paint(window);
        assert!(item.read_with(window, |item, _| item.png_is_pending()));
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PREPARE-PNG")
                .is_some()
        );
        assert!(!split.read_with(window, |item, _| item.png_is_pending()));
        assert!(split.read_with(window, |item, _| item.prepared_png().is_none()));
        split.read_with(window, |split, cx| {
            assert_eq!(split.session(), &session);
            assert_eq!(split.presentation(), item.read(cx).presentation());
            assert_eq!(
                split
                    .session()
                    .read(cx)
                    .preview()
                    .expect("same generation")
                    .generation(),
                generation
            );
        });
        assert!(Arc::ptr_eq(&presented(&split, window), &edited_image));
        assert!(!session.read_with(window, |session, _| session.png_is_current(&edited)));
        begin_click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        assert!(split.read_with(window, |item, _| item.png_is_pending()));
        workspace.update_in(window, |workspace, window, cx| {
            assert!(workspace.activate_item(&item, true, true, window, cx));
        });
        paint(window);
        assert!(item.read_with(window, |item, _| item.png_is_pending()));
        begin_click(window, "GRAPHITE-ARTBOARD-CANCEL-PNG");
        assert!(!item.read_with(window, |item, _| item.png_is_pending()));
        assert!(split.read_with(window, |item, _| item.png_is_pending()));
        item.update(window, |item, cx| {
            item.configure_png_limits(PIXEL_BYTES, 64 * 1024 + 1, cx)
        });
        assert!(split.read_with(window, |item, _| item.png_is_pending()));
        workspace.update_in(window, |workspace, window, cx| {
            assert!(workspace.activate_item(&split, true, true, window, cx));
        });
        window.run_until_parked();
        paint(window);
        assert!(!split.read_with(window, |item, _| item.png_is_pending()));
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                .is_some()
        );
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PNG-ERROR").is_none());
        let split_png = split.read_with(window, |item, _| {
            item.prepared_png()
                .expect("newer split request survived old view cancellation and limits")
                .clone()
        });
        assert!(session.read_with(window, |session, _| session.png_is_current(&split_png)));
        assert_eq!(
            split_png.png().encoded_bytes(),
            edited.png().encoded_bytes()
        );
        assert!(Arc::ptr_eq(&presented(&split, window), &edited_image));
        assert_eq!(
            session.read_with(window, |session, _| session
                .preview()
                .expect("no scene restart")
                .generation()),
            generation
        );
        let first_id = item.entity_id();
        workspace
            .update_in(window, |workspace, window, cx| {
                workspace.active_pane().clone().update(cx, |pane, cx| {
                    pane.close_item_by_id(first_id, workspace::SaveIntent::Close, window, cx)
                })
            })
            .await
            .expect("normal first native close");
        let weak_item = item.downgrade();
        window.update(|_, _| drop(item));
        paint(window);
        window.run_until_parked();
        assert!(weak_item.upgrade().is_none());
        let weak_session = session.downgrade();
        let weak_split = split.downgrade();
        let weak_presentation = split.read_with(window, |item, _| item.presentation().downgrade());
        drop(session);
        begin_click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        assert!(split.read_with(window, |item, _| item.png_is_pending()));
        let split_id = split.entity_id();
        workspace.update_in(window, |workspace, window, cx| {
            workspace.active_pane().clone().update(cx, |pane, cx| {
                pane.remove_item(split_id, false, true, window, cx)
            });
        });
        window.update(|_, _| drop(split));
        paint(window);
        window.run_until_parked();
        paint(window);
        assert!(weak_split.upgrade().is_none());
        assert!(weak_session.upgrade().is_none());
        assert!(weak_presentation.upgrade().is_none());
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&edited_image)));
        assert_eq!(original.source_output().revision(), 0);
        assert_eq!(edited.source_output().revision(), 1);
        assert_eq!(original.png().width(), 31);
        assert_eq!(edited.png().height(), 17);
    }
    #[gpui::test]
    async fn at_r22_1_artboard_workspace_png_rejects_local_drafts_and_recovers_explicitly(
        cx: &mut TestAppContext,
    ) {
        let (item, _, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
        item.update(window, |item, cx| {
            item.configure_png_limits(PIXEL_BYTES, 64 * 1024, cx)
        });
        paint(window);
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let session = item.read_with(window, |item, _| item.session().clone());
        let canonical_background = item.read_with(window, |item, cx| {
            item.artboard_editors[4].read(cx).text(cx)
        });
        let image = presented(&item, window);
        let original = item.read_with(window, |item, _| {
            item.prepared_png().expect("genuine committed PNG").clone()
        });
        let output = original.source_output().output().clone();
        let output_lease = original.source_output().output_lease();
        let network = session.read_with(window, |session, _| session.document().network().clone());
        let generation = session.read_with(window, |session, _| {
            session.preview().expect("committed frame").generation()
        });

        let location = edit(&item, window, 0, "4");
        assert_eq!(
            location.read_with(window, |editor, cx| editor.text(cx)),
            "4"
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                .is_none()
        );
        assert!(!item.read_with(window, |item, _| item.png_is_pending()));
        assert!(session.read_with(window, |session, _| session.png_is_current(&original)));
        session.read_with(window, |session, _| {
            assert_eq!(session.parameters(), parameters());
            assert_eq!(session.revision(), 0);
            assert_eq!(session.document().network(), &network);
            assert_eq!(
                session
                    .preview()
                    .expect("no draft-triggered job")
                    .generation(),
                generation
            );
        });
        assert!(Arc::ptr_eq(&presented(&item, window), &image));
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        assert_eq!(
            location.read_with(window, |editor, cx| editor.text(cx)),
            "2"
        );
        assert!(!window.update(|window, cx| location.read(cx).is_focused(window)));
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                .is_none()
        );
        assert!(session.read_with(window, |session, _| session.png_is_current(&original)));

        edit(&item, window, 4, "invalid gamma RGBA");
        commit(window);
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                .is_none()
        );
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        assert!(!item.read_with(window, |item, _| item.png_is_pending()));
        assert!(session.read_with(window, |session, _| session.png_is_current(&original)));
        item.read_with(window, |item, _| {
            assert_eq!(
                item.prepared_png()
                    .expect("unchanged committed bytes")
                    .png()
                    .encoded_bytes(),
                original.png().encoded_bytes()
            );
        });
        session.read_with(window, |session, _| {
            assert_eq!(session.parameters(), parameters());
            assert_eq!(session.revision(), 0);
            assert_eq!(session.configuration(), configuration());
            assert_eq!(session.document().network(), &network);
            assert_eq!(
                session
                    .preview()
                    .expect("no invalid-input job")
                    .generation(),
                generation
            );
        });
        assert!(Arc::ptr_eq(&presented(&item, window), &image));

        let background = edit(&item, window, 4, "invalid gamma RGBA");
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        assert_eq!(
            background.read_with(window, |editor, cx| editor.text(cx)),
            canonical_background
        );
        assert!(!window.update(|window, cx| background.read(cx).is_focused(window)));
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                .is_none()
        );
        assert!(!item.read_with(window, |item, _| item.png_is_pending()));
        assert!(session.read_with(window, |session, _| session.png_is_current(&original)));
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let recovered = item.read_with(window, |item, _| {
            item.prepared_png()
                .expect("explicit recovered PNG action")
                .clone()
        });
        assert!(!session.read_with(window, |session, _| session.png_is_current(&original)));
        assert!(session.read_with(window, |session, _| session.png_is_current(&recovered)));
        assert_eq!(
            recovered.png().encoded_bytes(),
            original.png().encoded_bytes()
        );

        edit(&item, window, 4, "0,0,1,0.75");
        commit(window);
        assert!(!session.read_with(window, |session, _| session.png_is_current(&recovered)));
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                .is_some()
        );
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        assert!(!item.read_with(window, |item, _| item.png_is_pending()));
        assert_eq!(
            item.read_with(window, |item, _| item
                .prepared_png()
                .expect("no stale-source preparation")
                .png()
                .encoded_bytes()
                .to_vec()),
            recovered.png().encoded_bytes()
        );
        assert!(Arc::ptr_eq(&presented(&item, window), &image));
        assert_eq!(
            session.read_with(window, |session, _| session.revision()),
            1
        );
        assert_eq!(
            session.read_with(window, |session, _| session
                .preview()
                .expect("no implicit scene job")
                .generation()),
            generation
        );
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let edited_image = presented(&item, window);
        assert_pixel(&edited_image, [255, 0, 0, 191]);
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let edited = item.read_with(window, |item, _| {
            item.prepared_png()
                .expect("explicit edited scene PNG")
                .clone()
        });
        assert!(session.read_with(window, |session, _| session.png_is_current(&edited)));
        assert_ne!(
            edited.png().content_sha256(),
            original.png().content_sha256()
        );
        assert_ne!(edited.pixels_sha256(), original.pixels_sha256());
        assert_eq!(edited.source_output().revision(), 1);
        assert_eq!(edited.source_output().configuration(), configuration());
        assert_eq!(original.source_output().output(), &output);
        assert_eq!(original.source_output().output().metadata, output.metadata);
        assert!(Arc::ptr_eq(
            &original.source_output().output_lease(),
            &output_lease
        ));
        assert_eq!(original.source_output().revision(), 0);
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                .is_some()
        );
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-PNG-ERROR").is_none());
    }

    #[gpui::test]
    async fn at_r08_artboard_asset_workspace_bridge_edits_bakes_and_retires_revoked_display(
        cx: &mut TestAppContext,
    ) {
        let fixture = crate::resources::tests::host_reader_fixture().expect("real asset service");
        let png =
            crate::resources::tests::indexed_request_png(&fixture, 4).expect("real indexed PNG");
        let (existing, workspace, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
        window.update(|_, cx| {
            crate::register_asset_binding(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                cx,
            )
            .expect("actual global Input binding")
        });
        let mut request = crate::interop::artboard_binding_controls::request(&png);
        request.placement = graphite_glam::DAffine2::from_scale_angle_translation(
            DVec2::new(18., 8.),
            0.,
            DVec2::new(2., 2.),
        );
        let acquisition = workspace
            .update_in(window, |_, window, cx| {
                crate::open_asset_artboard(request, PIXEL_BYTES, window, cx)
            })
            .expect("explicit root acquisition");
        let item = acquisition.await.expect("authorized active Item");
        window.run_until_parked();
        paint(window);
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_item_as::<GraphiteArtboardItem>(cx)
                .expect("new active Item")
                .entity_id()),
            item.entity_id()
        );
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_pane()
                .read(cx)
                .items_len()),
            2
        );
        let session = item.read_with(window, |item, _| item.session().clone());
        let source = session.read_with(window, |session, _| {
            session
                .document()
                .encoded_png_input()
                .expect("immutable input")
                .source()
                .clone()
        });
        assert_eq!(source.identity(), &png.image_identity);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        item.update(window, |item, cx| {
            item.configure_png_limits(PIXEL_BYTES, 64 * 1024, cx)
        });
        paint(window);
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let original = presented(&item, window);
        assert!(window.update(|window, _| window.has_image_atlas_entry(&original)));
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let original_png = item.read_with(window, |item, _| {
            item.prepared_png()
                .expect("real prepared source PNG")
                .clone()
        });
        assert!(session.read_with(window, |session, _| session.png_is_current(&original_png)));
        edit(&item, window, 2, "16");
        commit(window);
        assert_eq!(
            session.read_with(window, |session, _| session.parameters().dimensions.x),
            16.
        );
        assert!(!session.read_with(window, |session, _| session.png_is_current(&original_png)));
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let edited = presented(&item, window);
        assert_ne!(original.as_bytes(0), edited.as_bytes(0));
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let edited_png = item.read_with(window, |item, _| {
            item.prepared_png().expect("new baked frame").clone()
        });
        assert_ne!(
            original_png.png().encoded_bytes(),
            edited_png.png().encoded_bytes()
        );
        assert_eq!(edited_png.source_output().parameters().dimensions.x, 16.);
        assert!(Arc::ptr_eq(
            edited_png
                .source_output()
                .encoded_png_input()
                .expect("captured provenance")
                .source(),
            &source
        ));
        window.update(|_, cx| crate::clear_asset_binding(cx).expect("host revocation"));
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&edited)));
        assert!(item.read_with(window, |item, _| item.prepared_png().is_none()));
        assert!(!session.read_with(window, |session, _| session.png_is_current(&edited_png)));
        session.read_with(window, |session, _| {
            assert!(session.resource_is_revoked());
            assert!(session.current_output().is_none());
            assert_eq!(session.parameters().dimensions.x, 16.);
            assert!(Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("editable source retained")
                    .source(),
                &source
            ));
        });
        assert!(!existing.read_with(window, |item, cx| {
            item.session().read(cx).resource_is_revoked()
        }));
    }
    mod artboard_png_proposal_controls {
        use super::*;
        use crate::session::PreparedArtboardPngAsset;
        use crate::storage::{ArtboardProposalError, PreparedArtboardPngProposal};
        use sha2::Digest as _;
        use sim_runtime::{OutputCommitError, OutputProposal, assets::AssetNamespace};

        fn proposal(prepared: &PreparedArtboardPngAsset) -> OutputProposal {
            OutputProposal::new(
                "70000000-0000-0000-0000-000000000001"
                    .parse()
                    .expect("caller proposal ID"),
                AssetNamespace::Output,
                "graphite/captured-artboard",
                "png",
                7,
                prepared.png().width(),
                prepared.png().height(),
                prepared.png().encoded_bytes().to_vec(),
            )
            .expect("caller-owned proposal, no publication")
        }

        async fn bake(
            session: &Entity<GraphiteArtboardSession>,
            window: &mut VisualTestContext,
        ) -> PreparedArtboardPngAsset {
            let preview = session
                .update(window, |session, cx| {
                    session.request_preview(PIXEL_BYTES, cx)
                })
                .expect("genuine original-network request")
                .await
                .expect("accepted native frame");
            assert!(
                preview
                    .image()
                    .as_bytes(0)
                    .expect("single frame")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|pixel| pixel[3] != 0)
            );
            session
                .update(window, |session, cx| {
                    session.prepare_png(preview, PIXEL_BYTES, 64 * 1024, cx)
                })
                .expect("current native frame")
                .await
                .expect("actual shared PNG encoder")
        }

        fn digest_hex(digest: [u8; 32]) -> String {
            digest.iter().map(|byte| format!("{byte:02x}")).collect()
        }

        #[gpui::test]
        async fn at_r22_2_artboard_proposal_captures_exact_encoded_source_and_wire_metadata(
            cx: &mut TestAppContext,
        ) {
            let (_item, _workspace, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
            let fixture =
                crate::resources::tests::host_reader_fixture().expect("real indexed service");
            let indexed = crate::resources::tests::indexed_request_png(&fixture, 4)
                .expect("actual producer PNG");
            let binding = window
                .update(|_, cx| {
                    crate::interop::GraphiteAssetBinding::new(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        cx,
                    )
                })
                .expect("genuine Input grant");
            let mut request = crate::interop::artboard_binding_controls::request(&indexed);
            request.document_id_encoding = u64::MAX;
            request.parameters.location = DVec2::new(-0., 0.);
            request.parameters.background = Color::from_gamma_srgb_channels(0.25, 0.5, 0.75, 0.5);
            request.placement =
                graphite_glam::DAffine2::from_cols_array(&[18., -0., 0., 8., 2., 2.]);
            request.configuration.viewport.transform =
                graphite_glam::DAffine2::from_cols_array(&[1., -0., 0., 1., -0., 0.]);
            request.configuration.viewport.quality =
                graphite_core_types::transform::RenderQuality::Scale(0.5);
            request.configuration.time.time = -0.;
            request.configuration.time.animation_time = std::time::Duration::new(2, 345);
            request.configuration.pointer = DVec2::new(-0., 7.);
            let expected_parameters = request.parameters;
            let expected_configuration = request.configuration;
            let expected_placement = request
                .placement
                .to_cols_array()
                .map(|value| format!("{:016x}", value.to_bits()));
            let session = binding
                .update(window, |binding, cx| binding.open_artboard(request, cx))
                .expect("authorized acquisition")
                .await
                .expect("verified encoded checkpoint");
            let (network, revision, source) = session.read_with(window, |session, _| {
                (
                    session.document().network().clone(),
                    session.revision(),
                    session
                        .document()
                        .encoded_png_input()
                        .expect("retained seal")
                        .source()
                        .clone(),
                )
            });
            let prepared = bake(&session, window).await;
            let original = proposal(&prepared);
            let original_before = original.clone();
            let wrapped = session
                .read_with(window, |session, _| {
                    PreparedArtboardPngProposal::new(session, prepared.clone(), original)
                })
                .expect("typed proposal boundary");
            assert!(session.read_with(window, |session, _| wrapped.is_current(session)));
            assert_eq!(original_before.projection_metadata(), &[] as &[u8]);
            assert_eq!(
                wrapped.proposal().proposal_id(),
                original_before.proposal_id()
            );
            assert_eq!(wrapped.proposal().namespace(), AssetNamespace::Output);
            assert_eq!(
                wrapped.proposal().filename_prefix(),
                original_before.filename_prefix()
            );
            assert_eq!(wrapped.proposal().batch_index(), 7);
            assert_eq!(wrapped.proposal().extension(), "png");
            assert_eq!(
                (wrapped.proposal().width(), wrapped.proposal().height()),
                (31, 17)
            );
            assert_eq!(wrapped.proposal().content(), prepared.png().encoded_bytes());
            assert_eq!(
                wrapped.prepared().png().encoded_bytes(),
                prepared.png().encoded_bytes()
            );
            assert!(Arc::ptr_eq(
                &wrapped.prepared().source_output().output_lease(),
                &prepared.source_output().output_lease()
            ));
            let decoded = image::load_from_memory_with_format(
                wrapped.proposal().content(),
                image::ImageFormat::Png,
            )
            .expect("actual PNG")
            .into_rgba8();
            let mut rows = session.read_with(window, |session, _| {
                session
                    .preview()
                    .expect("current frame")
                    .image()
                    .as_bytes(0)
                    .expect("native frame")
                    .to_vec()
            });
            for pixel in rows.as_chunks_mut::<4>().0.iter_mut() {
                pixel.swap(0, 2);
            }
            assert_eq!(decoded.as_raw(), &rows);
            assert_eq!(
                prepared.pixels_sha256(),
                <[u8; 32]>::from(sha2::Sha256::digest(&rows))
            );
            assert_eq!(
                prepared.png().content_sha256(),
                <[u8; 32]>::from(sha2::Sha256::digest(wrapped.proposal().content()))
            );
            let metadata: serde_json::Value =
                serde_json::from_slice(wrapped.proposal().projection_metadata())
                    .expect("bounded JSON metadata");
            assert_eq!(metadata["schema"], "sim.graphite.artboard-png-proposal.v1");
            assert!(
                metadata["qualification"]
                    .as_str()
                    .expect("qualification")
                    .contains("not durable project versions")
            );
            assert!(
                metadata["qualification"]
                    .as_str()
                    .expect("qualification")
                    .contains("not embedded PNG metadata")
            );
            assert!(
                metadata["qualification"]
                    .as_str()
                    .expect("qualification")
                    .contains("not a publication lease")
            );
            assert_eq!(
                metadata["source_revision"],
                graphics_core::resources::GRAPHITE_IMAGE_SOURCE_REVISION
            );
            assert_eq!(metadata["document_id_encoding"], u64::MAX.to_string());
            assert_eq!(
                metadata["node_id"],
                prepared.source_output().node_id().0.to_string()
            );
            assert_eq!(metadata["revision"], revision.to_string());
            assert_eq!(
                metadata["input"],
                serde_json::json!({
                    "asset_label": indexed.image_identity.asset_id.as_ref(),
                    "version_label": indexed.image_identity.version_id.as_ref(),
                    "encoded_sha256": digest_hex(indexed.image_identity.content_sha256),
                    "width": source.width(), "height": source.height(), "placement_f64_bits": expected_placement,
                })
            );
            assert_eq!(
                metadata["artboard"],
                serde_json::json!({
                    "location_f64_bits": ["8000000000000000", "0000000000000000"],
                    "dimensions_f64_bits": ["403f000000000000", "4031000000000000"],
                    "background_linear_rgba_f32_bits": ([expected_parameters.background.r(), expected_parameters.background.g(), expected_parameters.background.b(), expected_parameters.background.a()].map(|value| format!("{:08x}", value.to_bits()))),
                    "clip": true,
                })
            );
            assert_eq!(
                metadata["render"],
                serde_json::json!({
                    "viewport_transform_f64_bits": ["3ff0000000000000", "8000000000000000", "0000000000000000", "3ff0000000000000", "8000000000000000", "0000000000000000"],
                    "viewport_resolution": [31,17], "quality": {"variant":"Scale", "f32_bits":"3f000000"},
                    "scale_f64_bits": ["3ff0000000000000"], "time_f64_bits": ["8000000000000000"],
                    "animation_seconds": "2", "animation_nanoseconds": 345,
                    "pointer_f64_bits": ["8000000000000000", "401c000000000000"],
                    "render_mode": serde_json::to_value(expected_configuration.render_mode).expect("source enum"),
                    "export_format": serde_json::to_value(expected_configuration.export_format).expect("source enum"),
                    "for_export": true, "for_eyedropper": false,
                })
            );
            assert_eq!(
                metadata["output"],
                serde_json::json!({
                    "content_type":"image/png", "width":31, "height":17,
                    "encoded_bytes": prepared.png().encoded_bytes().len().to_string(),
                    "encoded_sha256": digest_hex(prepared.png().content_sha256()),
                    "pixels_sha256": digest_hex(prepared.pixels_sha256()),
                    "sample_format":"rgba8", "color_space":"srgb", "alpha":"straight", "ownership":"immutable-cpu-bytes",
                })
            );
            session.read_with(window, |session, _| {
                assert_eq!(session.document().network(), &network);
                assert_eq!(session.revision(), revision);
                assert_eq!(session.parameters(), expected_parameters);
                assert!(Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("editable source")
                        .source(),
                    &source
                ));
            });
            let mut edited = expected_parameters;
            edited.dimensions.x = 29.;
            assert!(
                session
                    .update(window, |session, cx| session.apply(
                        revision,
                        edited,
                        expected_configuration,
                        cx
                    ))
                    .expect("transactional original edit")
            );
            assert!(!session.read_with(window, |session, _| wrapped.is_current(session)));
            assert!(matches!(
                session.read_with(window, |session, _| PreparedArtboardPngProposal::new(
                    session,
                    prepared.clone(),
                    original_before.clone()
                )),
                Err(ArtboardProposalError::SourceNotCurrent)
            ));
            session.read_with(window, |session, _| {
                assert_eq!(session.parameters(), edited);
                assert_eq!(session.revision(), revision + 1);
                assert!(Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("editable input after edit")
                        .source(),
                    &source
                ));
            });
            assert_eq!(
                wrapped.prepared().source_output().parameters(),
                expected_parameters
            );
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(
                    wrapped.proposal().projection_metadata()
                )
                .expect("immutable metadata"),
                metadata
            );
            assert!(Arc::ptr_eq(
                wrapped
                    .prepared()
                    .source_output()
                    .encoded_png_input()
                    .expect("captured input")
                    .source(),
                &source
            ));
            let refreshed = bake(&session, window).await;
            let successor = session
                .read_with(window, |session, _| {
                    PreparedArtboardPngProposal::new(
                        session,
                        refreshed.clone(),
                        proposal(&refreshed),
                    )
                })
                .expect("fresh successor");
            let successor_metadata: serde_json::Value =
                serde_json::from_slice(successor.proposal().projection_metadata())
                    .expect("new metadata");
            assert_eq!(successor_metadata["revision"], (revision + 1).to_string());
            assert_ne!(
                successor_metadata["artboard"]["dimensions_f64_bits"],
                metadata["artboard"]["dimensions_f64_bits"]
            );
            assert_eq!(successor_metadata["input"], metadata["input"]);
            let bytes = fixture
                .assets
                .lock()
                .expect("actual service")
                .read_verified(
                    &indexed.asset_identity,
                    &fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024,
                )
                .expect("unchanged indexed input");
            assert_eq!(bytes, indexed.encoded);
        }

        #[gpui::test]
        async fn at_r22_2_artboard_proposal_rejects_payload_metadata_and_foreign_stale_cancelled_revoked_sources(
            cx: &mut TestAppContext,
        ) {
            let (item, _workspace, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
            let session = item.read_with(window, |item, _| item.session().clone());
            let network =
                session.read_with(window, |session, _| session.document().network().clone());
            let prepared = bake(&session, window).await;
            let original = proposal(&prepared);
            let with_metadata = original
                .clone()
                .with_projection_metadata(b"{\"other\":true}".to_vec())
                .expect("existing foreign provenance");
            let mut wrong_bytes = original.content().to_vec();
            wrong_bytes[0] ^= 1;
            for (namespace, extension, width, height, bytes) in [
                (
                    AssetNamespace::Output,
                    "jpg",
                    31,
                    17,
                    original.content().to_vec(),
                ),
                (
                    AssetNamespace::Temporary,
                    "png",
                    31,
                    17,
                    original.content().to_vec(),
                ),
                (
                    AssetNamespace::Output,
                    "png",
                    30,
                    17,
                    original.content().to_vec(),
                ),
                (
                    AssetNamespace::Output,
                    "png",
                    31,
                    16,
                    original.content().to_vec(),
                ),
                (AssetNamespace::Output, "png", 31, 17, wrong_bytes),
            ] {
                let mismatched = OutputProposal::new(
                    original.proposal_id(),
                    namespace,
                    original.filename_prefix(),
                    extension,
                    original.batch_index(),
                    width,
                    height,
                    bytes,
                )
                .expect("valid host proposal with mismatched payload");
                assert!(matches!(
                    session.read_with(window, |session, _| PreparedArtboardPngProposal::new(
                        session,
                        prepared.clone(),
                        mismatched
                    )),
                    Err(ArtboardProposalError::PayloadMismatch)
                ));
            }
            assert!(matches!(
                session.read_with(window, |session, _| PreparedArtboardPngProposal::new(
                    session,
                    prepared.clone(),
                    with_metadata.clone()
                )),
                Err(ArtboardProposalError::ExistingMetadata)
            ));
            assert_eq!(with_metadata.projection_metadata(), b"{\"other\":true}");
            let foreign_document = session
                .read_with(window, |session, _| {
                    CpuSvgArtboardDocument::new(
                        session.document().document_id_encoding(),
                        session.document().node_id(),
                        session.parameters(),
                        session.configuration(),
                        CpuSvgEvaluationBudget {
                            max_viewport_pixels: 31 * 17,
                            max_output_bytes: 64 * 1024,
                        },
                    )
                })
                .expect("same public labels, distinct genuine owner");
            let foreign = window.new(|_| GraphiteArtboardSession::new(foreign_document));
            let foreign_prepared = bake(&foreign, window).await;
            assert!(matches!(
                session.read_with(window, |session, _| PreparedArtboardPngProposal::new(
                    session,
                    foreign_prepared.clone(),
                    proposal(&foreign_prepared)
                )),
                Err(ArtboardProposalError::SourceNotCurrent)
            ));
            let wrapped = session
                .read_with(window, |session, _| {
                    PreparedArtboardPngProposal::new(session, prepared.clone(), original.clone())
                })
                .expect("rejections preserve current receipt");
            assert!(session.read_with(window, |session, _| wrapped.is_current(session)));
            session.read_with(window, |session, _| {
                assert_eq!(session.document().network(), &network)
            });
            let metadata: serde_json::Value =
                serde_json::from_slice(wrapped.proposal().projection_metadata())
                    .expect("resource-free captured metadata");
            assert!(metadata["input"].is_null());
            session.update(window, |session, cx| {
                session.cancel(cx);
            });
            assert!(!session.read_with(window, |session, _| wrapped.is_current(session)));
            assert!(matches!(
                session.read_with(window, |session, _| PreparedArtboardPngProposal::new(
                    session,
                    prepared.clone(),
                    original.clone()
                )),
                Err(ArtboardProposalError::SourceNotCurrent)
            ));
            let recovered = bake(&session, window).await;
            let recovered_proposal = session
                .read_with(window, |session, _| {
                    PreparedArtboardPngProposal::new(
                        session,
                        recovered.clone(),
                        proposal(&recovered),
                    )
                })
                .expect("explicit recovery");
            session.update(window, |session, cx| session.revoke_resource(cx));
            assert!(
                !session.read_with(window, |session, _| recovered_proposal.is_current(session))
            );
            assert!(matches!(
                session.read_with(window, |session, _| PreparedArtboardPngProposal::new(
                    session,
                    recovered.clone(),
                    proposal(&recovered)
                )),
                Err(ArtboardProposalError::SourceNotCurrent)
            ));
            session.read_with(window, |session, _| {
                assert!(session.resource_is_revoked());
                assert_eq!(session.document().network(), &network);
                assert_eq!(session.parameters(), parameters());
            });
            assert_eq!(wrapped.proposal().content(), original.content());
            assert_eq!(
                wrapped.prepared().source_output().parameters(),
                parameters()
            );
        }

        #[gpui::test]
        async fn at_r22_2_artboard_proposal_metadata_cap_preserves_encoded_document_and_recovers(
            cx: &mut TestAppContext,
        ) {
            let (_item, _workspace, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
            let fixture = crate::resources::tests::host_reader_fixture().expect("actual fixture");
            let indexed = crate::resources::tests::indexed_request_png(&fixture, 3)
                .expect("actual producer bytes");
            let mut identity = indexed.image_identity.clone();
            identity.version_id = "caller-label-".repeat(2048).into();
            let source = Arc::new(
                graphics_core::resources::prepare_encoded_png_source(
                    identity.clone(),
                    Arc::from(indexed.encoded.clone()),
                    graphics_core::resources::EncodedPngBudget {
                        max_encoded_bytes: 16 * 1024,
                        max_pixel_bytes: 4096,
                        max_metadata_bytes: 4096,
                    },
                )
                .expect("envelope seal; caller labels are not durable versions"),
            );
            let input = graphics_core::evaluate::CpuSvgEncodedPngInput::new(
                source.clone(),
                graphite_glam::DAffine2::IDENTITY,
            )
            .expect("finite placement");
            let document = CpuSvgArtboardDocument::new_with_encoded_png(
                91,
                NodeId(7001),
                parameters(),
                configuration(),
                CpuSvgEvaluationBudget {
                    max_viewport_pixels: 31 * 17,
                    max_output_bytes: 64 * 1024,
                },
                input,
            )
            .expect("genuine source profile");
            let network = document.network().clone();
            let session = window.new(|_| GraphiteArtboardSession::new(document));
            let prepared = bake(&session, window).await;
            let original = proposal(&prepared);
            assert!(
                matches!(session.read_with(window, |session, _| PreparedArtboardPngProposal::new(session, prepared.clone(), original.clone())), Err(ArtboardProposalError::Host(OutputCommitError::ProjectionMetadataTooLarge { actual, limit })) if actual > limit && limit == 16*1024)
            );
            assert!(session.read_with(window, |session, _| session.png_is_current(&prepared)));
            session.read_with(window, |session, _| {
                assert_eq!(session.document().network(), &network);
                assert_eq!(session.revision(), 0);
                assert_eq!(session.parameters(), parameters());
                assert!(Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("retained input")
                        .source(),
                    &source
                ));
            });
            assert_eq!(source.identity(), &identity);
            assert!(original.projection_metadata().is_empty());
            assert_eq!(original.content(), prepared.png().encoded_bytes());
            let recovery_source = Arc::new(
                graphics_core::resources::prepare_encoded_png_source(
                    indexed.image_identity.clone(),
                    Arc::from(indexed.encoded.clone()),
                    graphics_core::resources::EncodedPngBudget {
                        max_encoded_bytes: 16 * 1024,
                        max_pixel_bytes: 4096,
                        max_metadata_bytes: 4096,
                    },
                )
                .expect("short caller label seal"),
            );
            let recovery_document = CpuSvgArtboardDocument::new_with_encoded_png(
                91,
                NodeId(7001),
                parameters(),
                configuration(),
                CpuSvgEvaluationBudget {
                    max_viewport_pixels: 31 * 17,
                    max_output_bytes: 64 * 1024,
                },
                graphics_core::evaluate::CpuSvgEncodedPngInput::new(
                    recovery_source,
                    graphite_glam::DAffine2::IDENTITY,
                )
                .expect("placement"),
            )
            .expect("new explicit owner");
            let recovery = window.new(|_| GraphiteArtboardSession::new(recovery_document));
            let recovered = bake(&recovery, window).await;
            let wrapped = recovery
                .read_with(window, |session, _| {
                    PreparedArtboardPngProposal::new(
                        session,
                        recovered.clone(),
                        proposal(&recovered),
                    )
                })
                .expect("bounded metadata recovery");
            assert!(wrapped.proposal().projection_metadata().len() <= 16 * 1024);
            assert_eq!(wrapped.proposal().content(), original.content());
            assert!(!recovery.read_with(window, |_session, _| {
                wrapped
                    .prepared()
                    .source_output()
                    .encoded_png_input()
                    .expect("separate input")
                    .source()
                    .identity()
                    .version_id
                    == identity.version_id
            }));
            assert_eq!(source.identity(), &identity);
            assert_eq!(
                indexed.encoded,
                fixture
                    .assets
                    .lock()
                    .expect("actual service")
                    .read_verified(
                        &indexed.asset_identity,
                        &fixture.input_authorization,
                        &sim_tensor::CancellationToken::default(),
                        16 * 1024
                    )
                    .expect("input unchanged")
            );
        }
    }
    mod artboard_image_placement_controls {
        use super::*;
        use crate::interop::GraphiteAssetBinding;
        use crate::resources::tests::{HostReaderFixture, IndexedRequestPng};
        use crate::session::ArtboardSessionError;
        use crate::storage::{ArtboardProposalError, PreparedArtboardPngProposal};
        use graphics_core::evaluate::CpuSvgDocumentError;
        use graphite_glam::DAffine2;

        async fn placed_artboard(
            cx: &mut TestAppContext,
        ) -> (
            Entity<GraphiteArtboardItem>,
            Entity<workspace::Workspace>,
            Entity<GraphiteAssetBinding>,
            HostReaderFixture,
            IndexedRequestPng,
            &mut VisualTestContext,
        ) {
            let (_fixture_item, workspace, window) =
                native_artboard(cx, parameters(), PIXEL_BYTES).await;
            for selector in [
                "GRAPHITE-ARTBOARD-IMAGE-XX",
                "GRAPHITE-ARTBOARD-IMAGE-XY",
                "GRAPHITE-ARTBOARD-IMAGE-YX",
                "GRAPHITE-ARTBOARD-IMAGE-YY",
                "GRAPHITE-ARTBOARD-IMAGE-X",
                "GRAPHITE-ARTBOARD-IMAGE-Y",
            ] {
                assert!(window.debug_bounds(selector).is_none());
            }
            let plain_session = _fixture_item.read_with(window, |item, _| item.session().clone());
            let plain_network =
                plain_session.read_with(window, |session, _| session.document().network().clone());
            assert!(matches!(
                plain_session.update(window, |session, cx| session.apply_image_placement(
                    session.revision(),
                    DAffine2::IDENTITY,
                    cx
                )),
                Err(ArtboardSessionError::MissingEncodedPngInput)
            ));
            plain_session.read_with(window, |session, _| {
                assert_eq!(session.revision(), 0);
                assert!(session.image_placement().is_none());
                assert_eq!(session.parameters(), parameters());
                assert_eq!(session.configuration(), configuration());
                assert_eq!(session.document().network(), &plain_network);
            });
            let fixture =
                crate::resources::tests::host_reader_fixture().expect("actual indexed host");
            let png = crate::resources::tests::indexed_request_png(&fixture, 3)
                .expect("actual RGB PNG producer");
            let binding = window
                .update(|_, cx| {
                    crate::register_asset_binding(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        cx,
                    )
                })
                .expect("genuine Input binding");
            let mut request = crate::interop::artboard_binding_controls::request(&png);
            request.placement = DAffine2::from_cols_array(&[12., 0., 0., 8., 2., 2.]);
            let acquisition = workspace
                .update_in(window, |_, window, cx| {
                    crate::open_asset_artboard(request, PIXEL_BYTES, window, cx)
                })
                .expect("explicit host caller");
            let item = acquisition.await.expect("verified source checkpoint");
            item.update(window, |item, cx| {
                item.configure_png_limits(PIXEL_BYTES, 64 * 1024, cx)
            });
            window.run_until_parked();
            paint(window);
            (item, workspace, binding, fixture, png, window)
        }

        fn png_proposal(
            session: &Entity<GraphiteArtboardSession>,
            item: &Entity<GraphiteArtboardItem>,
            window: &VisualTestContext,
        ) -> PreparedArtboardPngProposal {
            let prepared = item.read_with(window, |item, _| {
                item.prepared_png().expect("genuine Item PNG").clone()
            });
            let proposal = sim_runtime::OutputProposal::new(
                "71000000-0000-0000-0000-000000000001"
                    .parse()
                    .expect("caller proposal ID"),
                sim_runtime::assets::AssetNamespace::Output,
                "graphite/image-placement",
                "png",
                0,
                prepared.png().width(),
                prepared.png().height(),
                prepared.png().encoded_bytes().to_vec(),
            )
            .expect("existing proposal constructor without publication");
            session
                .read_with(window, |session, _| {
                    PreparedArtboardPngProposal::new(session, prepared, proposal)
                })
                .expect("current captured provenance")
        }

        fn columns(
            session: &Entity<GraphiteArtboardSession>,
            window: &VisualTestContext,
        ) -> [u64; 6] {
            session.read_with(window, |session, _| {
                session
                    .image_placement()
                    .expect("encoded image placement")
                    .to_cols_array()
                    .map(f64::to_bits)
            })
        }

        fn verify_input(fixture: &HostReaderFixture, png: &IndexedRequestPng) {
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("original indexed service")
                    .read_verified(
                        &png.asset_identity,
                        &fixture.input_authorization,
                        &sim_tensor::CancellationToken::default(),
                        16 * 1024,
                    )
                    .expect("unchanged authenticated bytes"),
                png.encoded
            );
        }

        #[gpui::test]
        async fn at_r23_1_bound_artboard_matrix_edit_commits_ordered_source_and_reprepares_png_proposal(
            cx: &mut TestAppContext,
        ) {
            let (item, _workspace, _binding, fixture, png, window) = placed_artboard(cx).await;
            let session = item.read_with(window, |item, _| item.session().clone());
            for selector in [
                "GRAPHITE-ARTBOARD-IMAGE-XX",
                "GRAPHITE-ARTBOARD-IMAGE-XY",
                "GRAPHITE-ARTBOARD-IMAGE-YX",
                "GRAPHITE-ARTBOARD-IMAGE-YY",
                "GRAPHITE-ARTBOARD-IMAGE-X",
                "GRAPHITE-ARTBOARD-IMAGE-Y",
            ] {
                assert!(window.debug_bounds(selector).is_some());
            }
            let (network_before, source, artboard, configuration_before) =
                session.read_with(window, |session, _| {
                    (
                        session.document().network().clone(),
                        session
                            .document()
                            .encoded_png_input()
                            .expect("source")
                            .source()
                            .clone(),
                        session.parameters(),
                        session.configuration(),
                    )
                });
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            let original = presented(&item, window);
            let preview_before = session.read_with(window, |session, _| {
                session.preview().expect("original actual receipt").clone()
            });
            let output_before = preview_before.output().output().clone();
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            let original_proposal = png_proposal(&session, &item, window);
            let original_metadata: serde_json::Value =
                serde_json::from_slice(original_proposal.proposal().projection_metadata())
                    .expect("captured metadata");
            let mut expected = [12., 0., 0., 8., 2., 2.];
            for (component, expression, value) in [
                (0, "21 / 2", 10.5),
                (1, "1 / 4", 0.25),
                (2, "-1 / 2", -0.5),
                (3, "25 / 4", 6.25),
                (4, "33 / 2", 16.5),
                (5, "13 / 4", 3.25),
            ] {
                let revision = session.read_with(window, |session, _| session.revision());
                edit(&item, window, component + 5, expression);
                assert_eq!(columns(&session, window), expected.map(f64::to_bits));
                assert_eq!(
                    session.read_with(window, |session, _| session.revision()),
                    revision
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                        .is_some()
                );
                commit(window);
                expected[component] = value;
                assert_eq!(columns(&session, window), expected.map(f64::to_bits));
                assert_eq!(
                    session.read_with(window, |session, _| session.revision()),
                    revision + 1
                );
                assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
                assert!(
                    window
                        .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                        .is_some()
                );
                assert!(Arc::ptr_eq(&presented(&item, window), &original));
                session.read_with(window, |session, _| {
                    assert_eq!(session.parameters(), artboard);
                    assert_eq!(session.configuration(), configuration_before);
                    assert!(Arc::ptr_eq(
                        session
                            .document()
                            .encoded_png_input()
                            .expect("same owned source")
                            .source(),
                        &source
                    ));
                });
            }
            assert!(!session.read_with(window, |session, _| original_proposal.is_current(session)));
            assert!(matches!(
                session.read_with(window, |session, _| PreparedArtboardPngProposal::new(
                    session,
                    original_proposal.prepared().clone(),
                    original_proposal.proposal().clone()
                )),
                Err(ArtboardProposalError::SourceNotCurrent)
            ));
            assert_eq!(
                original_proposal.prepared().source_output().output(),
                &output_before
            );
            assert_eq!(
                original_proposal
                    .prepared()
                    .source_output()
                    .encoded_png_input()
                    .expect("historical placement")
                    .placement()
                    .to_cols_array()
                    .map(f64::to_bits),
                [12., 0., 0., 8., 2., 2.].map(f64::to_bits)
            );
            assert_ne!(
                session.read_with(window, |session, _| session.document().network().clone()),
                network_before
            );
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            let moved = presented(&item, window);
            assert_ne!(moved.as_bytes(0), original.as_bytes(0));
            assert_eq!(pixel(&moved, 3, 4), [255, 255, 255, 255]);
            assert_eq!(pixel(&moved, 30, 16), [255, 255, 255, 255]);
            assert!(
                moved
                    .as_bytes(0)
                    .expect("actual native frame")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|pixel| *pixel != [255, 255, 255, 255])
            );
            let current = session.read_with(window, |session, _| {
                session.preview().expect("new original output").clone()
            });
            let graphite_graph_craft::document::value::RenderOutputType::Svg { svg, image_data } =
                &current.output().output().data
            else {
                panic!("original SVG export");
            };
            assert!(image_data.is_empty());
            assert!(svg.contains("matrix(10.5,0.25,-0.5,6.25,16.5,3.25)"));
            assert_eq!(
                current.output().output().metadata.backgrounds,
                output_before.metadata.backgrounds
            );
            assert_eq!(
                current
                    .output()
                    .encoded_png_input()
                    .expect("captured exact placement")
                    .placement()
                    .to_cols_array()
                    .map(f64::to_bits),
                expected.map(f64::to_bits)
            );
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            let moved_proposal = png_proposal(&session, &item, window);
            assert_ne!(
                moved_proposal.proposal().content(),
                original_proposal.proposal().content()
            );
            assert_ne!(
                moved_proposal.prepared().png().content_sha256(),
                original_proposal.prepared().png().content_sha256()
            );
            assert_ne!(
                moved_proposal.prepared().pixels_sha256(),
                original_proposal.prepared().pixels_sha256()
            );
            let decoded = image::load_from_memory_with_format(
                moved_proposal.proposal().content(),
                image::ImageFormat::Png,
            )
            .expect("actual baked PNG")
            .into_rgba8();
            let mut expected_rows = moved.as_bytes(0).expect("one native frame").to_vec();
            for pixel in expected_rows.as_chunks_mut::<4>().0.iter_mut() {
                pixel.swap(0, 2);
            }
            assert_eq!(decoded.as_raw(), &expected_rows);
            let moved_metadata: serde_json::Value =
                serde_json::from_slice(moved_proposal.proposal().projection_metadata())
                    .expect("new captured placement");
            assert_eq!(
                moved_metadata["input"]["asset_label"],
                original_metadata["input"]["asset_label"]
            );
            assert_eq!(
                moved_metadata["input"]["version_label"],
                original_metadata["input"]["version_label"]
            );
            assert_eq!(
                moved_metadata["input"]["encoded_sha256"],
                original_metadata["input"]["encoded_sha256"]
            );
            assert_eq!(
                moved_metadata["input"]["placement_f64_bits"],
                serde_json::json!(expected.map(|value| format!("{:016x}", value.to_bits())))
            );
            assert_eq!(moved_metadata["revision"], "6");
            assert!(Arc::ptr_eq(
                moved_proposal
                    .prepared()
                    .source_output()
                    .encoded_png_input()
                    .expect("retained seal")
                    .source(),
                &source
            ));
            let moved_network =
                session.read_with(window, |session, _| session.document().network().clone());
            edit(&item, window, 5, "10 + 0.5");
            commit(window);
            assert_eq!(columns(&session, window), expected.map(f64::to_bits));
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                6
            );
            assert!(session.read_with(window, |session, _| moved_proposal.is_current(session)));
            assert!(Arc::ptr_eq(&presented(&item, window), &moved));
            session.read_with(window, |session, _| {
                assert_eq!(session.document().network(), &moved_network)
            });
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                    .is_some()
            );
            verify_input(&fixture, &png);
        }

        #[gpui::test]
        async fn at_r23_1_placement_invalid_drafts_and_bitwise_transactions_preserve_source_and_cancel_stale_work(
            cx: &mut TestAppContext,
        ) {
            let (item, _workspace, _binding, fixture, png, window) = placed_artboard(cx).await;
            let session = item.read_with(window, |item, _| item.session().clone());
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            let original = presented(&item, window);
            let proposal = png_proposal(&session, &item, window);
            let network =
                session.read_with(window, |session, _| session.document().network().clone());
            let initial = columns(&session, window);
            for text in ["not a number", "NaN", "∞"] {
                let editor = edit(&item, window, 5, text);
                commit(window);
                assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
                assert_eq!(columns(&session, window), initial);
                assert_eq!(
                    session.read_with(window, |session, _| session.revision()),
                    0
                );
                assert!(session.read_with(window, |session, _| proposal.is_current(session)));
                assert!(
                    window
                        .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                        .is_some()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                        .is_some()
                );
                assert!(Arc::ptr_eq(&presented(&item, window), &original));
                editor.update_in(window, |editor, window, cx| {
                    editor.focus_handle(cx).focus(window, cx)
                });
                window.simulate_keystrokes("escape");
                window.run_until_parked();
                paint(window);
                assert_eq!(editor.read_with(window, |editor, cx| editor.text(cx)), "12");
                assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
                assert!(
                    window
                        .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                        .is_some()
                );
            }
            let largest = f64::MAX * 0.75;
            for placement in [
                DAffine2::from_cols_array(&[f64::NAN, 0., 0., 1., 0., 0.]),
                DAffine2::from_cols_array(&[1., 0., 0., 1., f64::INFINITY, 0.]),
                DAffine2::from_cols_array(&[largest, 0., largest, 1., 0., 0.]),
            ] {
                assert!(matches!(
                    session.update(window, |session, cx| session.apply_image_placement(
                        session.revision(),
                        placement,
                        cx
                    )),
                    Err(ArtboardSessionError::Document(
                        CpuSvgDocumentError::InvalidPlacement
                    ))
                ));
                assert_eq!(columns(&session, window), initial);
                assert_eq!(
                    session.read_with(window, |session, _| session.revision()),
                    0
                );
                assert!(session.read_with(window, |session, _| proposal.is_current(session)));
                session.read_with(window, |session, _| {
                    assert_eq!(session.document().network(), &network)
                });
            }
            let pending = session
                .update(window, |session, cx| {
                    session.request_preview(PIXEL_BYTES, cx)
                })
                .expect("actual pending successor");
            let mut signed = [12., -0., 0., 8., 2., 2.];
            assert!(
                session
                    .update(window, |session, cx| session.apply_image_placement(
                        0,
                        DAffine2::from_cols_array(&signed),
                        cx
                    ))
                    .expect("exact negative-zero source edit")
            );
            assert_eq!(columns(&session, window), signed.map(f64::to_bits));
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            assert!(
                !session
                    .update(window, |session, cx| session.apply_image_placement(
                        1,
                        DAffine2::from_cols_array(&signed),
                        cx
                    ))
                    .expect("same-bit no-op")
            );
            assert!(pending.await.is_err());
            assert!(!session.read_with(window, |session, _| proposal.is_current(session)));
            assert!(matches!(
                session.update(window, |session, cx| session.apply_image_placement(
                    0,
                    DAffine2::from_cols_array(&signed),
                    cx
                )),
                Err(ArtboardSessionError::Document(
                    CpuSvgDocumentError::StaleRevision
                ))
            ));
            let retry = session
                .update(window, |session, cx| {
                    session.request_preview(PIXEL_BYTES, cx)
                })
                .expect("retained negative-zero source");
            assert!(
                !session
                    .update(window, |session, cx| session.apply_image_placement(
                        1,
                        DAffine2::from_cols_array(&signed),
                        cx
                    ))
                    .expect("no-op preserves pending work")
            );
            let retained = retry.await.expect("same-bit pending request survives");
            assert!(session.read_with(window, |session, _| session.preview_is_current(&retained)));
            assert_eq!(
                retained
                    .output()
                    .encoded_png_input()
                    .expect("exact captured sign")
                    .placement()
                    .to_cols_array()
                    .map(f64::to_bits),
                signed.map(f64::to_bits)
            );
            signed[1] = 0.;
            assert!(
                session
                    .update(window, |session, cx| session.apply_image_placement(
                        1,
                        DAffine2::from_cols_array(&signed),
                        cx
                    ))
                    .expect("positive-zero is a distinct source edit")
            );
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                2
            );
            assert!(!session.read_with(window, |session, _| session.preview_is_current(&retained)));
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            assert_eq!(columns(&session, window), signed.map(f64::to_bits));
            let current = session.read_with(window, |session, _| {
                session.preview().expect("recovered frame").clone()
            });
            assert!(session.read_with(window, |session, _| session.preview_is_current(&current)));
            let finite_large = DAffine2::from_cols_array(&[largest, 0., 0., 8., 2., 2.]);
            assert!(
                session
                    .update(window, |session, cx| session.apply_image_placement(
                        2,
                        finite_large,
                        cx
                    ))
                    .expect("finite individual axis remains admitted")
            );
            let network_before_overflow =
                session.read_with(window, |session, _| session.document().network().clone());
            let recovery_image = presented(&item, window);
            let overflowing_component = graphics_core::numeric_input::evaluate_number_input(
                "1.3482698511467367e308",
                graphics_core::numeric_input::NumberInputConstraints::default(),
            )
            .expect("finite original numeric expression");
            assert!(overflowing_component.is_finite());
            assert!((largest + overflowing_component).is_infinite());
            let overflowing_editor = edit(&item, window, 7, "1.3482698511467367e308");
            commit(window);
            assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                    .is_some()
            );
            assert_eq!(
                columns(&session, window),
                finite_large.to_cols_array().map(f64::to_bits)
            );
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                3
            );
            session.read_with(window, |session, _| {
                assert_eq!(session.document().network(), &network_before_overflow)
            });
            assert!(Arc::ptr_eq(&presented(&item, window), &recovery_image));
            overflowing_editor.update_in(window, |editor, window, cx| {
                editor.focus_handle(cx).focus(window, cx)
            });
            window.simulate_keystrokes("escape");
            window.run_until_parked();
            paint(window);
            assert_eq!(
                overflowing_editor.read_with(window, |editor, cx| editor.text(cx)),
                "0"
            );
            edit(&item, window, 5, "12");
            commit(window);
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                4
            );
            assert_eq!(columns(&session, window), signed.map(f64::to_bits));
            assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            let recovered = session.read_with(window, |session, _| {
                session
                    .preview()
                    .expect("finite recovery after rejected corner overflow")
                    .clone()
            });
            assert!(session.read_with(window, |session, _| session.preview_is_current(&recovered)));
            verify_input(&fixture, &png);
        }

        #[gpui::test]
        async fn at_r23_1_placement_split_drafts_remain_local_and_revoked_metadata_edits_cannot_revive_output(
            cx: &mut TestAppContext,
        ) {
            let (item, workspace, binding, fixture, png, window) = placed_artboard(cx).await;
            let session = item.read_with(window, |item, _| item.session().clone());
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            let image = presented(&item, window);
            let old_proposal = png_proposal(&session, &item, window);
            let source = session.read_with(window, |session, _| {
                session
                    .document()
                    .encoded_png_input()
                    .expect("editable input")
                    .source()
                    .clone()
            });
            let cloned = item.update_in(window, |item, window, cx| {
                item.clone_on_split(None, window, cx)
            });
            let split = cloned.await.expect("native Item split");
            workspace.update_in(window, |workspace, window, cx| {
                workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx)
            });
            paint(window);
            split.read_with(window, |split, cx| {
                assert_eq!(split.session(), &session);
                assert_eq!(split.presentation(), item.read(cx).presentation());
                assert_ne!(
                    split.artboard_editors[9].entity_id(),
                    item.read(cx).artboard_editors[9].entity_id()
                );
                assert!(!session.read(cx).is_pending());
                assert_eq!(session.read(cx).revision(), 0);
            });
            assert!(Arc::ptr_eq(&presented(&split, window), &image));
            workspace.update_in(window, |workspace, window, cx| {
                assert!(workspace.activate_item(&item, true, true, window, cx));
            });
            paint(window);
            let draft = edit(&item, window, 9, "uncommitted draft");
            let original_bits = columns(&session, window);
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                0
            );
            assert_eq!(columns(&session, window), original_bits);
            let updated = DAffine2::from_cols_array(&[12., 0., 0., 8., 16., 2.]);
            assert!(
                session
                    .update(window, |session, cx| session
                        .apply_image_placement(0, updated, cx))
                    .expect("external shared-session placement edit")
            );
            paint(window);
            assert_eq!(
                draft.read_with(window, |editor, cx| editor.text(cx)),
                "uncommitted draft"
            );
            assert!(window.update(|window, cx| draft.read(cx).is_focused(window)));
            assert_eq!(
                split.read_with(window, |split, cx| split.artboard_editors[9]
                    .read(cx)
                    .text(cx)),
                "16"
            );
            window.simulate_keystrokes("enter");
            window.run_until_parked();
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            assert_eq!(
                columns(&session, window),
                updated.to_cols_array().map(f64::to_bits)
            );
            workspace.update_in(window, |workspace, window, cx| {
                assert!(workspace.activate_item(&split, true, true, window, cx));
            });
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
            assert_eq!(
                split.read_with(window, |split, cx| split.artboard_editors[9]
                    .read(cx)
                    .text(cx)),
                "16"
            );
            assert_eq!(
                draft.read_with(window, |editor, cx| editor.text(cx)),
                "uncommitted draft"
            );
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            let moved = presented(&split, window);
            assert_ne!(image.as_bytes(0), moved.as_bytes(0));
            assert!(!session.read_with(window, |session, _| old_proposal.is_current(session)));
            workspace.update_in(window, |workspace, window, cx| {
                assert!(workspace.activate_item(&item, true, true, window, cx));
            });
            paint(window);
            draft.update_in(window, |editor, window, cx| {
                editor.focus_handle(cx).focus(window, cx)
            });
            window.simulate_keystrokes("escape");
            window.run_until_parked();
            paint(window);
            assert_eq!(draft.read_with(window, |editor, cx| editor.text(cx)), "16");
            assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
            assert!(!window.update(|window, cx| draft.read(cx).is_focused(window)));
            assert!(Arc::ptr_eq(&presented(&item, window), &moved));
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            let current_proposal = png_proposal(&session, &item, window);
            binding
                .update(window, |binding, cx| binding.invalidate(cx))
                .expect("real host lease invalidation");
            window.run_until_parked();
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            assert!(!window.update(|window, _| window.has_image_atlas_entry(&moved)));
            assert!(!session.read_with(window, |session, _| current_proposal.is_current(session)));
            assert!(item.read_with(window, |item, _| item.prepared_png().is_none()));
            edit(&item, window, 9, "9 / 2");
            commit(window);
            assert_eq!(columns(&session, window)[4], 4.5_f64.to_bits());
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                2
            );
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            assert!(matches!(
                session.update(window, |session, cx| session
                    .request_preview(PIXEL_BYTES, cx)),
                Err(ArtboardSessionError::ResourceRevoked)
            ));
            assert!(matches!(
                session.read_with(window, |session, _| PreparedArtboardPngProposal::new(
                    session,
                    current_proposal.prepared().clone(),
                    current_proposal.proposal().clone()
                )),
                Err(ArtboardProposalError::SourceNotCurrent)
            ));
            session.read_with(window, |session, _| {
                assert!(session.resource_is_revoked());
                assert!(session.current_output().is_none());
                assert!(Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("revoked but editable source")
                        .source(),
                    &source
                ));
            });
            assert_eq!(
                current_proposal
                    .prepared()
                    .source_output()
                    .encoded_png_input()
                    .expect("escaped historical output")
                    .placement()
                    .to_cols_array()
                    .map(f64::to_bits),
                updated.to_cols_array().map(f64::to_bits)
            );
            assert_eq!(
                current_proposal
                    .prepared()
                    .source_output()
                    .output()
                    .metadata
                    .backgrounds
                    .len(),
                1
            );
            verify_input(&fixture, &png);
        }
    }
    mod artboard_gesture_controls {
        use super::*;
        use crate::interop::GraphiteAssetBinding;
        use crate::resources::tests::{HostReaderFixture, IndexedRequestPng};
        use crate::session::ArtboardSessionError;
        use graphics_core::evaluate::{CpuSvgDocumentError, CpuSvgEncodedPngInput};
        use graphite_glam::DAffine2;

        async fn gesture_workspace(
            cx: &mut TestAppContext,
        ) -> (
            Entity<GraphiteArtboardItem>,
            Entity<workspace::Workspace>,
            Entity<GraphiteAssetBinding>,
            HostReaderFixture,
            IndexedRequestPng,
            &mut VisualTestContext,
        ) {
            let (_fixture_item, workspace, window) =
                native_artboard(cx, parameters(), PIXEL_BYTES).await;
            let fixture =
                crate::resources::tests::host_reader_fixture().expect("canonical real service");
            let png = crate::resources::tests::indexed_request_png(&fixture, 3)
                .expect("real indexed producer PNG");
            let binding = window
                .update(|_, cx| {
                    crate::register_asset_binding(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        cx,
                    )
                })
                .expect("genuine host binding");
            let mut request = crate::interop::artboard_binding_controls::request(&png);
            request.placement = DAffine2::from_cols_array(&[12., 0., 0., 8., 2., 2.]);
            let item = workspace
                .update_in(window, |_, window, cx| {
                    crate::open_asset_artboard(request, PIXEL_BYTES, window, cx)
                })
                .expect("existing explicit root route")
                .await
                .expect("verified checkpoint");
            item.update(window, |item, cx| {
                item.configure_png_limits(PIXEL_BYTES, 64 * 1024, cx)
            });
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            (item, workspace, binding, fixture, png, window)
        }

        #[gpui::test]
        async fn at_r28_1_committed_output_artboard_workspace_edits_refreshes_and_bakes_with_provenance(
            cx: &mut TestAppContext,
        ) {
            let (_fixture_item, workspace, window) =
                native_artboard(cx, parameters(), PIXEL_BYTES).await;
            let fixture = crate::resources::tests::host_reader_fixture().expect("actual service");
            let png = crate::resources::tests::indexed_request_png(&fixture, 3)
                .expect("actual PNG producer");
            let (receipt, output, _writer, reader) =
                crate::interop::tests::output_reader_tests::published(
                    &fixture,
                    &png,
                    "40000000-0000-0000-0000-000000000001",
                    "graphite-artboard-workspace",
                );
            let receipt_before = receipt.clone();
            let output_identity = output.asset_identity.clone();
            let expected_source = output.identity.clone();
            let binding = window
                .update(|_, cx| {
                    crate::register_asset_binding_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(reader.clone()),
                        cx,
                    )
                })
                .expect("separate genuine Output read grant");
            let mut request = crate::interop::artboard_binding_controls::request(&png);
            request.asset_identity = output.asset_identity;
            request.identity = output.identity;
            request.placement = DAffine2::from_cols_array(&[12., 0., 0., 8., 2., 2.]);
            let item = workspace
                .update_in(window, |_, window, cx| {
                    crate::open_committed_asset_artboard(request, &receipt, PIXEL_BYTES, window, cx)
                })
                .expect("public receipt-aware workspace route")
                .await
                .expect("real authenticated Output source");
            let session = item.read_with(window, |item, _| item.session().clone());
            item.update(window, |item, cx| {
                item.configure_png_limits(PIXEL_BYTES, 64 * 1024, cx)
            });
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
            let first = item.read_with(window, |item, _| {
                item.prepared_png().expect("first real PNG").clone()
            });
            assert_eq!(
                first
                    .source_output()
                    .encoded_png_input()
                    .expect("source")
                    .source()
                    .identity(),
                &expected_source
            );
            edit(&item, window, 9, "7");
            commit(window);
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            assert!(!session.read_with(window, |session, _| session.png_is_current(&first)));
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            let second = item.read_with(window, |item, _| {
                item.prepared_png().expect("refreshed real PNG").clone()
            });
            assert_ne!(first.png().encoded_bytes(), second.png().encoded_bytes());
            assert_eq!(second.source_output().revision(), 1);
            assert_eq!(
                second
                    .source_output()
                    .encoded_png_input()
                    .expect("retained source")
                    .source()
                    .identity(),
                &expected_source
            );
            assert_eq!(
                first
                    .source_output()
                    .encoded_png_input()
                    .expect("old placement")
                    .placement()
                    .translation
                    .x,
                2.
            );
            assert_eq!(
                second
                    .source_output()
                    .encoded_png_input()
                    .expect("edited placement")
                    .placement()
                    .translation
                    .x,
                7.
            );
            assert_eq!(receipt, receipt_before);
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("same service")
                    .read_verified(
                        &output_identity,
                        &reader,
                        &sim_tensor::CancellationToken::default(),
                        16 * 1024,
                    )
                    .expect("published bytes unchanged"),
                png.encoded
            );
            unchanged_input(&fixture, &png);
            binding
                .update(window, |binding, cx| binding.invalidate(cx))
                .expect("terminal revoke");
            window.run_until_parked();
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            assert!(!session.read_with(window, |session, _| session.png_is_current(&second)));
            session.read_with(window, |session, _| {
                assert!(session.resource_is_revoked());
                assert_eq!(session.revision(), 1);
                assert_eq!(
                    session
                        .image_placement()
                        .expect("editable work retained")
                        .translation
                        .x,
                    7.
                );
                assert_eq!(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("source retained")
                        .source()
                        .identity(),
                    &expected_source
                );
            });
        }

        fn unchanged_input(fixture: &HostReaderFixture, png: &IndexedRequestPng) {
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("actual indexed owner")
                    .read_verified(
                        &png.asset_identity,
                        &fixture.input_authorization,
                        &sim_tensor::CancellationToken::default(),
                        16 * 1024
                    )
                    .expect("original authenticated bytes"),
                png.encoded
            );
        }

        #[gpui::test]
        async fn at_r05_1_native_batch_stage_invalid_update_and_discard_preserve_committed_work(
            cx: &mut TestAppContext,
        ) {
            let (item, _workspace, _binding, fixture, png, window) = gesture_workspace(cx).await;
            let session = item.read_with(window, |item, _| item.session().clone());
            let image = presented(&item, window);
            let old_png = item.read_with(window, |item, _| {
                item.prepared_png().expect("actual committed PNG").clone()
            });
            let old_preview = session.read_with(window, |session, _| {
                session.preview().expect("genuine old output").clone()
            });
            let (network, parameters_before, placement, configuration_before, source) = session
                .read_with(window, |session, _| {
                    (
                        session.document().network().clone(),
                        session.parameters(),
                        session.image_placement().expect("placement"),
                        session.configuration(),
                        session
                            .document()
                            .encoded_png_input()
                            .expect("seal")
                            .source()
                            .clone(),
                    )
                });
            let detached =
                session.read_with(window, |session, _| session.document().detach_for_edit());
            assert!(!detached.is_pending());
            assert!(detached.accepted().is_none());
            assert_eq!(detached.network(), &network);
            assert!(Arc::ptr_eq(
                detached
                    .encoded_png_input()
                    .expect("same immutable source")
                    .source(),
                &source
            ));
            assert!(session.read_with(window, |session, _| {
                session.document().owns_current_output(old_preview.output())
            }));
            click(window, "GRAPHITE-ARTBOARD-BEGIN-EDITS");
            let token = session.read_with(window, |session, _| {
                session.gesture().expect("opaque owned gesture")
            });
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-APPLY-EDITS")
                    .is_some()
            );
            edit(&item, window, 2, "28");
            commit(window);
            edit(&item, window, 9, "18");
            commit(window);
            click(window, "GRAPHITE-ARTBOARD-CLIP");
            let candidate = session.read_with(window, |session, _| {
                (
                    session.authoring_parameters(),
                    session
                        .authoring_image_placement()
                        .expect("projected placement"),
                )
            });
            assert_eq!(candidate.0.dimensions.x, 28.);
            assert!(!candidate.0.clip);
            assert_eq!(candidate.1.translation.x, 18.);
            let mut invalid = candidate.0;
            invalid.dimensions.x = f64::NAN;
            assert!(matches!(
                session.update(window, |session, cx| session.stage_gesture(
                    &token,
                    invalid,
                    Some(candidate.1),
                    cx
                )),
                Err(ArtboardSessionError::Document(
                    CpuSvgDocumentError::InvalidGeometry
                ))
            ));
            assert!(matches!(
                session.update(window, |session, cx| session.stage_gesture(
                    &token,
                    candidate.0,
                    None,
                    cx
                )),
                Err(ArtboardSessionError::MissingEncodedPngInput)
            ));
            assert!(
                !session
                    .update(window, |session, cx| session.stage_gesture(
                        &token,
                        candidate.0,
                        Some(candidate.1),
                        cx
                    ))
                    .expect("exact candidate no-op")
            );
            session.read_with(window, |session, _| {
                assert_eq!(session.authoring_parameters(), candidate.0);
                assert_eq!(
                    session
                        .authoring_image_placement()
                        .expect("same valid draft")
                        .to_cols_array()
                        .map(f64::to_bits),
                    candidate.1.to_cols_array().map(f64::to_bits)
                );
                assert_eq!(session.parameters(), parameters_before);
                assert_eq!(session.configuration(), configuration_before);
                assert_eq!(session.revision(), 0);
                assert_eq!(
                    session
                        .image_placement()
                        .expect("committed placement")
                        .to_cols_array()
                        .map(f64::to_bits),
                    placement.to_cols_array().map(f64::to_bits)
                );
                assert_eq!(session.document().network(), &network);
                assert!(session.preview_is_current(&old_preview));
                assert!(session.png_is_current(&old_png));
            });
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                    .is_some()
            );
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-PNG-LAST-VALID")
                    .is_some()
            );
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            assert!(session.read_with(
                window,
                |session, _| session.preview_is_current(&old_preview)
                    && session.png_is_current(&old_png)
                    && !session.is_pending()
            ));
            assert!(Arc::ptr_eq(&presented(&item, window), &image));
            click(window, "GRAPHITE-ARTBOARD-DISCARD-EDITS");
            assert!(session.read_with(window, |session, _| session.gesture().is_none()));
            assert_eq!(
                item.read_with(window, |item, cx| item.artboard_editors[2]
                    .read(cx)
                    .text(cx)),
                "31"
            );
            assert_eq!(
                item.read_with(window, |item, cx| item.artboard_editors[9]
                    .read(cx)
                    .text(cx)),
                "2"
            );
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                    .is_some()
            );
            assert!(session.read_with(window, |session, _| session.png_is_current(&old_png)));
            assert!(matches!(
                session.update(window, |session, cx| session.cancel_gesture(&token, cx)),
                Err(ArtboardSessionError::InvalidGesture)
            ));
            let pending = session
                .update(window, |session, cx| {
                    session.request_preview(PIXEL_BYTES, cx)
                })
                .expect("actual unchanged pending work");
            let next = session
                .update(window, |session, cx| session.begin_gesture(0, cx))
                .expect("new opaque gesture");
            assert!(
                session
                    .update(window, |session, cx| session.stage_gesture(
                        &next,
                        candidate.0,
                        Some(candidate.1),
                        cx
                    ))
                    .expect("stage without cancelling pending work")
            );
            session
                .update(window, |session, cx| {
                    session.cancel_gesture(&next.clone(), cx)
                })
                .expect("clone is same opaque identity");
            let completed = pending
                .await
                .expect("discard preserves actual committed request");
            assert!(session.read_with(window, |session, _| session.preview_is_current(&completed)));
            session.read_with(window, |session, _| {
                assert_eq!(session.document().network(), &network);
                assert_eq!(session.revision(), 0);
                assert_eq!(session.authoring_parameters(), parameters_before);
                assert_eq!(
                    session
                        .authoring_image_placement()
                        .expect("discarded projection")
                        .to_cols_array()
                        .map(f64::to_bits),
                    placement.to_cols_array().map(f64::to_bits)
                );
                assert!(Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("retained encoded input")
                        .source(),
                    &source
                ));
            });
            assert_eq!(old_preview.output().parameters(), parameters_before);
            unchanged_input(&fixture, &png);
        }

        #[gpui::test]
        async fn at_r05_1_native_apply_staged_properties_commits_once_and_updates_shared_views(
            cx: &mut TestAppContext,
        ) {
            let (item, workspace, _binding, fixture, png, window) = gesture_workspace(cx).await;
            let session = item.read_with(window, |item, _| item.session().clone());
            let old_image = presented(&item, window);
            let old_png = item.read_with(window, |item, _| {
                item.prepared_png().expect("old actual PNG").clone()
            });
            let (network, source, configuration_before) =
                session.read_with(window, |session, _| {
                    (
                        session.document().network().clone(),
                        session
                            .document()
                            .encoded_png_input()
                            .expect("input")
                            .source()
                            .clone(),
                        session.configuration(),
                    )
                });
            let split = item
                .update_in(window, |item, window, cx| {
                    item.clone_on_split(None, window, cx)
                })
                .await
                .expect("native split");
            workspace.update_in(window, |workspace, window, cx| {
                workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx)
            });
            paint(window);
            click(window, "GRAPHITE-ARTBOARD-BEGIN-EDITS");
            edit(&split, window, 2, "28");
            commit(window);
            edit(&split, window, 4, "0,0,1,1");
            commit(window);
            edit(&split, window, 9, "18");
            commit(window);
            click(window, "GRAPHITE-ARTBOARD-CLIP");
            let (candidate, placement) = session.read_with(window, |session, _| {
                (
                    session.authoring_parameters(),
                    session.authoring_image_placement().expect("staged affine"),
                )
            });
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                0
            );
            assert_eq!(
                item.read_with(window, |item, cx| item.artboard_editors[2]
                    .read(cx)
                    .text(cx)),
                "28"
            );
            assert_eq!(
                item.read_with(window, |item, cx| item.artboard_editors[9]
                    .read(cx)
                    .text(cx)),
                "18"
            );
            let pending = session
                .update(window, |session, cx| {
                    session.request_preview(PIXEL_BYTES, cx)
                })
                .expect("genuine old-revision request");
            begin_click(window, "GRAPHITE-ARTBOARD-APPLY-EDITS");
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            assert!(pending.await.is_err());
            window.run_until_parked();
            paint(window);
            session.read_with(window, |session, _| {
                assert!(session.gesture().is_none());
                assert_eq!(session.parameters(), candidate);
                assert_eq!(session.authoring_parameters(), candidate);
                assert_eq!(
                    session
                        .image_placement()
                        .expect("single committed affine")
                        .to_cols_array()
                        .map(f64::to_bits),
                    placement.to_cols_array().map(f64::to_bits)
                );
                assert_eq!(session.configuration(), configuration_before);
                assert_ne!(session.document().network(), &network);
                assert!(!session.png_is_current(&old_png));
                assert!(Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("same immutable bytes")
                        .source(),
                    &source
                ));
            });
            assert!(Arc::ptr_eq(&presented(&split, window), &old_image));
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                    .is_some()
            );
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            let changed = presented(&split, window);
            assert_ne!(changed.as_bytes(0), old_image.as_bytes(0));
            assert_eq!(pixel(&changed, 0, 0), [255, 0, 0, 255]);
            assert_eq!(pixel(&changed, 30, 16), [0, 0, 0, 0]);
            assert!(Arc::ptr_eq(&presented(&item, window), &changed));
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            let new_png = split.read_with(window, |item, _| {
                item.prepared_png()
                    .expect("explicit new derived PNG")
                    .clone()
            });
            assert_eq!(new_png.source_output().revision(), 1);
            assert_eq!(new_png.source_output().parameters(), candidate);
            assert_eq!(
                new_png
                    .source_output()
                    .encoded_png_input()
                    .expect("captured source")
                    .placement()
                    .to_cols_array()
                    .map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
            assert_ne!(new_png.png().encoded_bytes(), old_png.png().encoded_bytes());
            assert!(session.read_with(window, |session, _| session.png_is_current(&new_png)));
            assert!(Arc::ptr_eq(
                new_png
                    .source_output()
                    .encoded_png_input()
                    .expect("same seal after atomic commit")
                    .source(),
                &source
            ));
            assert_eq!(old_png.source_output().revision(), 0);
            assert_eq!(old_png.source_output().parameters().dimensions.x, 31.);
            assert_eq!(
                old_png
                    .source_output()
                    .encoded_png_input()
                    .expect("old placement retained")
                    .placement()
                    .translation
                    .x,
                2.
            );
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            unchanged_input(&fixture, &png);
        }

        #[gpui::test]
        async fn at_r05_1_native_noop_batch_and_exact_signed_zero_preserve_or_invalidate_authority(
            cx: &mut TestAppContext,
        ) {
            let (item, _workspace, _binding, fixture, png, window) = gesture_workspace(cx).await;
            let session = item.read_with(window, |item, _| item.session().clone());
            let old_png = item.read_with(window, |item, _| {
                item.prepared_png().expect("old committed PNG").clone()
            });
            let old_preview = session.read_with(window, |session, _| {
                session.preview().expect("accepted output").clone()
            });
            let old_image = presented(&item, window);
            let (parameters_before, placement, network) =
                session.read_with(window, |session, _| {
                    (
                        session.parameters(),
                        session.image_placement().expect("source placement"),
                        session.document().network().clone(),
                    )
                });
            click(window, "GRAPHITE-ARTBOARD-BEGIN-EDITS");
            let consumed = session.read_with(window, |session, _| {
                session.gesture().expect("opaque identity")
            });
            edit(&item, window, 2, "28");
            commit(window);
            edit(&item, window, 9, "18");
            commit(window);
            edit(&item, window, 2, "31");
            commit(window);
            edit(&item, window, 9, "2");
            commit(window);
            click(window, "GRAPHITE-ARTBOARD-APPLY-EDITS");
            session.read_with(window, |session, _| {
                assert!(session.gesture().is_none());
                assert_eq!(session.revision(), 0);
                assert_eq!(session.document().network(), &network);
                assert!(session.preview_is_current(&old_preview));
                assert!(session.png_is_current(&old_png));
            });
            assert!(Arc::ptr_eq(&presented(&item, window), &old_image));
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-PNG-RESULT")
                    .is_some()
            );
            assert!(matches!(
                session.update(window, |session, cx| session.commit_gesture(&consumed, cx)),
                Err(ArtboardSessionError::InvalidGesture)
            ));
            let pending = session
                .update(window, |session, cx| {
                    session.request_preview(PIXEL_BYTES, cx)
                })
                .expect("actual pending unchanged request");
            let token = session
                .update(window, |session, cx| session.begin_gesture(0, cx))
                .expect("new exact no-op gesture");
            assert!(
                !session
                    .update(window, |session, cx| session.stage_gesture(
                        &token,
                        parameters_before,
                        Some(placement),
                        cx
                    ))
                    .expect("same exact values")
            );
            assert!(
                !session
                    .update(window, |session, cx| session.commit_gesture(&token, cx))
                    .expect("no-op commit does not cancel pending request")
            );
            let preview = pending.await.expect("pending request survived no-op");
            assert!(session.read_with(window, |session, _| session.preview_is_current(&preview)));
            let token = session
                .update(window, |session, cx| session.begin_gesture(0, cx))
                .expect("bitwise edit");
            let mut signed = placement.to_cols_array();
            signed[1] = -0.;
            assert!(
                session
                    .update(window, |session, cx| session.stage_gesture(
                        &token,
                        parameters_before,
                        Some(DAffine2::from_cols_array(&signed)),
                        cx
                    ))
                    .expect("negative-zero candidate differs")
            );
            assert_eq!(
                session.read_with(window, |session, _| session
                    .image_placement()
                    .expect("live not staged")
                    .to_cols_array()[1]
                    .to_bits()),
                0_f64.to_bits()
            );
            assert_eq!(
                session.read_with(window, |session, _| session
                    .authoring_image_placement()
                    .expect("staged sign")
                    .to_cols_array()[1]
                    .to_bits()),
                (-0_f64).to_bits()
            );
            assert!(
                session
                    .update(window, |session, cx| session.commit_gesture(&token, cx))
                    .expect("single bit-sensitive commit")
            );
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            assert!(!session.read_with(window, |session, _| session.preview_is_current(&preview)));
            assert_eq!(
                session.read_with(window, |session, _| session
                    .image_placement()
                    .expect("committed sign")
                    .to_cols_array()
                    .map(f64::to_bits)),
                signed.map(f64::to_bits)
            );
            click(window, "GRAPHITE-ARTBOARD-PREPARE");
            click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
            let signed_png = item.read_with(window, |item, _| {
                item.prepared_png()
                    .expect("actual negative-zero checkpoint PNG")
                    .clone()
            });
            let repeat = session
                .update(window, |session, cx| session.begin_gesture(1, cx))
                .expect("repeat exact bits");
            assert!(
                !session
                    .update(window, |session, cx| session.stage_gesture(
                        &repeat,
                        parameters_before,
                        Some(DAffine2::from_cols_array(&signed)),
                        cx
                    ))
                    .expect("signed repeat no-op")
            );
            assert!(
                !session
                    .update(window, |session, cx| session.commit_gesture(&repeat, cx))
                    .expect("exact repeat no-op")
            );
            assert!(session.read_with(window, |session, _| session.png_is_current(&signed_png)));
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            assert_eq!(
                old_png
                    .source_output()
                    .encoded_png_input()
                    .expect("escaped immutable positive-zero checkpoint")
                    .placement()
                    .to_cols_array()[1]
                    .to_bits(),
                0_f64.to_bits()
            );
            unchanged_input(&fixture, &png);
        }

        #[gpui::test]
        async fn at_r05_1_gesture_owner_stale_and_invalid_local_apply_rejections_keep_live_state_and_revocation_terminal(
            cx: &mut TestAppContext,
        ) {
            let (item, _workspace, binding, fixture, png, window) = gesture_workspace(cx).await;
            let session = item.read_with(window, |item, _| item.session().clone());
            let (original, placement, source, configuration_before, network) =
                session.read_with(window, |session, _| {
                    (
                        session.parameters(),
                        session.image_placement().expect("placement"),
                        session
                            .document()
                            .encoded_png_input()
                            .expect("source")
                            .source()
                            .clone(),
                        session.configuration(),
                        session.document().network().clone(),
                    )
                });
            let foreign_document = session
                .read_with(window, |session, _| {
                    CpuSvgArtboardDocument::new_with_encoded_png(
                        session.document().document_id_encoding(),
                        session.document().node_id(),
                        original,
                        configuration_before,
                        CpuSvgEvaluationBudget {
                            max_viewport_pixels: 31 * 17,
                            max_output_bytes: 64 * 1024,
                        },
                        CpuSvgEncodedPngInput::new(source.clone(), placement)
                            .expect("same public source labels"),
                    )
                })
                .expect("distinct genuine owner with colliding IDs");
            let foreign = window.new(|_| GraphiteArtboardSession::new(foreign_document));
            let foreign_token = foreign
                .update(window, |session, cx| session.begin_gesture(0, cx))
                .expect("genuine foreign opaque identity");
            let token = session
                .update(window, |session, cx| session.begin_gesture(0, cx))
                .expect("owned identity");
            assert!(matches!(
                session.update(window, |session, cx| session.begin_gesture(0, cx)),
                Err(ArtboardSessionError::GestureAlreadyActive)
            ));
            assert!(matches!(
                session.update(window, |session, cx| session.begin_gesture(1, cx)),
                Err(ArtboardSessionError::Document(
                    CpuSvgDocumentError::StaleRevision
                ))
            ));
            for result in [
                session.update(window, |session, cx| {
                    session.stage_gesture(&foreign_token, original, Some(placement), cx)
                }),
                session.update(window, |session, cx| {
                    session.commit_gesture(&foreign_token, cx)
                }),
            ] {
                assert!(matches!(result, Err(ArtboardSessionError::InvalidGesture)));
            }
            assert!(matches!(
                session.update(window, |session, cx| session
                    .cancel_gesture(&foreign_token, cx)),
                Err(ArtboardSessionError::InvalidGesture)
            ));
            let mut candidate = original;
            candidate.dimensions.x = 28.;
            assert!(
                session
                    .update(window, |session, cx| session.stage_gesture(
                        &token,
                        candidate,
                        Some(placement),
                        cx
                    ))
                    .expect("valid owned draft")
            );
            let mut successor_context = configuration_before;
            successor_context.pointer.x = 7.;
            assert!(
                session
                    .update(window, |session, cx| session.apply(
                        0,
                        original,
                        successor_context,
                        cx
                    ))
                    .expect("live context successor; no context staging")
            );
            let live_network =
                session.read_with(window, |session, _| session.document().network().clone());
            assert_eq!(live_network, network);
            assert!(matches!(
                session.update(window, |session, cx| session.stage_gesture(
                    &token,
                    candidate,
                    Some(placement),
                    cx
                )),
                Err(ArtboardSessionError::Document(
                    CpuSvgDocumentError::StaleRevision
                ))
            ));
            assert!(matches!(
                session.update(window, |session, cx| session.commit_gesture(&token, cx)),
                Err(ArtboardSessionError::Document(
                    CpuSvgDocumentError::StaleRevision
                ))
            ));
            session.read_with(window, |session, _| {
                assert_eq!(session.parameters(), original);
                assert_eq!(session.authoring_parameters(), candidate);
                assert_eq!(session.revision(), 1);
                assert_eq!(session.configuration(), successor_context);
                assert_eq!(session.document().network(), &live_network);
            });
            session
                .update(window, |session, cx| session.cancel_gesture(&token, cx))
                .expect("stale draft remains cancellable");
            assert_eq!(
                session.read_with(window, |session, _| session.authoring_parameters()),
                original
            );
            assert!(matches!(
                session.update(window, |session, cx| session.stage_gesture(
                    &token,
                    candidate,
                    Some(placement),
                    cx
                )),
                Err(ArtboardSessionError::InvalidGesture)
            ));
            paint(window);
            click(window, "GRAPHITE-ARTBOARD-BEGIN-EDITS");
            edit(&item, window, 2, "29");
            commit(window);
            let invalid_editor = edit(&item, window, 9, "invalid staged input");
            commit(window);
            click(window, "GRAPHITE-ARTBOARD-APPLY-EDITS");
            assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
            assert!(session.read_with(window, |session, _| session.gesture().is_some()));
            assert_eq!(
                session.read_with(window, |session, _| session
                    .authoring_parameters()
                    .dimensions
                    .x),
                29.
            );
            assert_eq!(
                session.read_with(window, |session, _| session.parameters().dimensions.x),
                31.
            );
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            invalid_editor.update_in(window, |editor, window, cx| {
                editor.focus_handle(cx).focus(window, cx)
            });
            window.simulate_keystrokes("escape");
            window.run_until_parked();
            paint(window);
            assert_eq!(
                invalid_editor.read_with(window, |editor, cx| editor.text(cx)),
                "2"
            );
            click(window, "GRAPHITE-ARTBOARD-DISCARD-EDITS");
            assert_eq!(
                session.read_with(window, |session, _| session.authoring_parameters()),
                original
            );
            click(window, "GRAPHITE-ARTBOARD-BEGIN-EDITS");
            edit(&item, window, 2, "26");
            commit(window);
            binding
                .update(window, |binding, cx| binding.invalidate(cx))
                .expect("real lease loss while draft retained");
            window.run_until_parked();
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            click(window, "GRAPHITE-ARTBOARD-APPLY-EDITS");
            session.read_with(window, |session, _| {
                assert!(session.resource_is_revoked());
                assert_eq!(session.revision(), 2);
                assert_eq!(session.parameters().dimensions.x, 26.);
                assert!(session.gesture().is_none());
                assert!(session.current_output().is_none());
                assert!(Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("editable original source survives")
                        .source(),
                    &source
                ));
            });
            assert!(matches!(
                session.update(window, |session, cx| session
                    .request_preview(PIXEL_BYTES, cx)),
                Err(ArtboardSessionError::ResourceRevoked)
            ));
            assert!(item.read_with(window, |item, _| item.prepared_png().is_none()));
            assert_eq!(
                session.read_with(window, |session, _| session.configuration()),
                successor_context
            );
            unchanged_input(&fixture, &png);
        }
    }
    mod provided_numeric_controls {
        use super::*;
        use graphics_core::document::{
            NumericAddCommand, NumericAddDocument, NumericAddInput, NumericDocumentError,
        };

        fn document(document_id: u64, augend: f64, addend: f64) -> NumericAddDocument {
            NumericAddDocument::new(document_id, NodeId(701), augend, addend)
                .expect("caller-supplied original numeric network")
        }

        fn command(
            document_id: u64,
            revision: u64,
            input: NumericAddInput,
            value: f64,
        ) -> NumericAddCommand {
            NumericAddCommand {
                document_id_encoding: document_id,
                network_path: Vec::new(),
                target_node_id: NodeId(701),
                expected_revision: revision,
                input,
                value,
            }
        }

        async fn numeric_workspace(
            cx: &mut TestAppContext,
            document: NumericAddDocument,
        ) -> (
            Entity<GraphicsWorkspaceItem>,
            Entity<workspace::Workspace>,
            &mut VisualTestContext,
        ) {
            let (fixture, workspace, window) = native_artboard(cx, parameters(), PIXEL_BYTES).await;
            let fixture_id = fixture.entity_id();
            workspace
                .update_in(window, |workspace, window, cx| {
                    workspace.active_pane().clone().update(cx, |pane, cx| {
                        pane.close_item_by_id(fixture_id, workspace::SaveIntent::Close, window, cx)
                    })
                })
                .await
                .expect("native close of existing fixture Item");
            window.update(|_, _| drop(fixture));
            paint(window);
            window.run_until_parked();
            let item = workspace.update_in(window, |workspace, window, cx| {
                let item = crate::graph::open_numeric_document(document, window, cx)
                    .expect("supplied owner opens original evaluation");
                workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
                item.focus_handle(cx).focus(window, cx);
                item
            });
            (item, workspace, window)
        }

        fn edit_numeric(
            item: &Entity<GraphicsWorkspaceItem>,
            window: &mut VisualTestContext,
            input: NumericAddInput,
            text: &str,
        ) -> Entity<editor::Editor> {
            let editor = item.read_with(window, |item, _| match input {
                NumericAddInput::Augend => item.augend_editor.clone(),
                NumericAddInput::Addend => item.addend_editor.clone(),
            });
            editor.update_in(window, |editor, window, cx| {
                editor.focus_handle(cx).focus(window, cx)
            });
            window.simulate_keystrokes("ctrl-a");
            window.simulate_input(text);
            window.run_until_parked();
            paint(window);
            editor
        }

        #[gpui::test]
        async fn at_r04_r18_supplied_numeric_workspace_executes_original_network_and_addressed_native_expressions(
            cx: &mut TestAppContext,
        ) {
            let supplied = document(811, 2., 3.);
            let snapshot = supplied.snapshot();
            let network = snapshot.network().clone();
            let (item, workspace, window) = numeric_workspace(cx, supplied).await;
            let session = item.read_with(window, |item, _| item.session.clone());
            session.read_with(window, |session, _| {
                assert!(session.numeric_pending());
                assert!(session.current_numeric_result().is_none());
                let document = session.numeric_document().expect("supplied document owner");
                assert_eq!(document.document_id_encoding(), 811);
                assert_eq!(document.node_id(), NodeId(701));
                assert_eq!(document.revision(), 0);
                assert_eq!(document.network(), &network);
            });
            window.run_until_parked();
            paint(window);
            assert_eq!(
                session.read_with(window, |session, _| session.current_numeric_result()),
                Some(5.)
            );
            assert!(window.debug_bounds("GRAPHITE-RESULT-5").is_some());
            assert_eq!(
                workspace.read_with(window, |workspace, cx| workspace
                    .active_item_as::<GraphicsWorkspaceItem>(cx)
                    .expect("active supplied Item")
                    .entity_id()),
                item.entity_id()
            );
            edit_numeric(&item, window, NumericAddInput::Augend, "2 * (3 + 4)");
            session.read_with(window, |session, _| {
                assert_eq!(
                    session
                        .numeric_document()
                        .expect("owner")
                        .inputs()
                        .expect("original inputs"),
                    (14., 3.)
                );
                assert_eq!(session.revision(), 1);
                assert_eq!(session.current_numeric_result(), Some(17.));
            });
            let addend = item.read_with(window, |item, _| item.addend_editor.clone());
            addend.update_in(window, |editor, window, cx| {
                editor.focus_handle(cx).focus(window, cx)
            });
            window.simulate_keystrokes("ctrl-a");
            for (character, text, value, revision, result_id) in [
                ("1", "1", 1., 2, "GRAPHITE-RESULT-15"),
                (
                    "e",
                    "1e",
                    std::f64::consts::E,
                    3,
                    "GRAPHITE-RESULT-16.718281828459045",
                ),
                ("1", "1e1", 10., 4, "GRAPHITE-RESULT-24"),
                (" ", "1e1 ", 10., 4, "GRAPHITE-RESULT-24"),
            ] {
                window.simulate_input(character);
                window.run_until_parked();
                paint(window);
                let expected = graphics_core::evaluate_add(14., value);
                session.read_with(window, |session, _| {
                    let (augend, addend) = session
                        .numeric_document()
                        .expect("supplied source")
                        .inputs()
                        .expect("original prefix inputs");
                    assert_eq!(augend.to_bits(), 14_f64.to_bits());
                    assert_eq!(addend.to_bits(), value.to_bits());
                    assert_eq!(session.revision(), revision);
                    assert_eq!(session.current_numeric_result(), Some(expected));
                    assert!(session.diagnostic().is_none());
                });
                assert_eq!(addend.read_with(window, |editor, cx| editor.text(cx)), text);
                assert!(window.debug_bounds(result_id).is_some());
            }
            for (character, text) in [("/", "1e1 /"), (" ", "1e1 / ")] {
                window.simulate_input(character);
                window.run_until_parked();
                paint(window);
                session.read_with(window, |session, _| {
                    assert_eq!(session.revision(), 4);
                    assert_eq!(
                        session
                            .numeric_document()
                            .expect("source")
                            .inputs()
                            .expect("retained inputs"),
                        (14., 10.)
                    );
                    assert!(session.current_numeric_result().is_none());
                    assert!(!session.numeric_pending());
                    assert!(session.diagnostic().is_some());
                    assert!(session.expression_text().contains("Last valid result: 24"));
                });
                assert_eq!(addend.read_with(window, |editor, cx| editor.text(cx)), text);
                assert!(window.debug_bounds("GRAPHITE-RESULT-24").is_some());
                assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
            }
            window.simulate_input("2");
            window.run_until_parked();
            paint(window);
            assert_eq!(
                addend.read_with(window, |editor, cx| editor.text(cx)),
                "1e1 / 2"
            );
            session.read_with(window, |session, _| {
                assert_eq!(
                    session
                        .numeric_document()
                        .expect("owner")
                        .inputs()
                        .expect("original inputs"),
                    (14., 5.)
                );
                assert_eq!(session.revision(), 5);
                assert_eq!(session.current_numeric_result(), Some(19.));
            });
            assert!(window.debug_bounds("GRAPHITE-RESULT-19").is_some());
            edit_numeric(&item, window, NumericAddInput::Addend, "5");
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                5
            );
            assert_eq!(snapshot.inputs().expect("detached original pair"), (2., 3.));
            assert_eq!(snapshot.network(), &network);
            assert_eq!(
                snapshot
                    .evaluate_network()
                    .await
                    .expect("detached original interpreter"),
                5.
            );
            click(window, "GRAPHITE-NUMERIC-CANCEL");
            session.read_with(window, |session, _| {
                assert!(session.current_numeric_result().is_none());
                assert!(!session.numeric_pending());
                assert_eq!(session.revision(), 5);
                assert!(session.expression_text().contains("Last valid result: 19"));
            });
            assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
            click(window, "GRAPHITE-NUMERIC-REFRESH");
            assert_eq!(
                session.read_with(window, |session, _| session.current_numeric_result()),
                Some(19.)
            );
            assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_none());
        }

        #[gpui::test]
        async fn at_r04_r18_numeric_split_and_colliding_nodes_preserve_invalid_draft_until_native_discard(
            cx: &mut TestAppContext,
        ) {
            let (item, workspace, window) = numeric_workspace(cx, document(812, 2., 3.)).await;
            window.run_until_parked();
            paint(window);
            let session = item.read_with(window, |item, _| item.session.clone());
            let split_task = item.update_in(window, |item, window, cx| {
                item.clone_on_split(None, window, cx)
            });
            let split = split_task.await.expect("actual numeric native split");
            workspace.update_in(window, |workspace, window, cx| {
                workspace.add_item_to_active_pane(Box::new(split.clone()), None, true, window, cx)
            });
            split.read_with(window, |split, cx| {
                assert_eq!(split.session, session);
                assert_ne!(
                    split.augend_editor.entity_id(),
                    item.read(cx).augend_editor.entity_id()
                );
                assert_ne!(split.focus_handle(cx), item.read(cx).focus_handle(cx));
                assert_eq!(session.read(cx).current_numeric_result(), Some(5.));
                assert!(!session.read(cx).numeric_pending());
            });
            let other = workspace.update_in(window, |workspace, window, cx| {
                let other =
                    crate::graph::open_numeric_document(document(813, 20., 30.), window, cx)
                        .expect("second supplied document with colliding NodeId");
                workspace.add_item_to_active_pane(Box::new(other.clone()), None, true, window, cx);
                other
            });
            window.run_until_parked();
            let other_session = other.read_with(window, |item, _| item.session.clone());
            workspace.update_in(window, |workspace, window, cx| {
                assert!(workspace.activate_item(&item, true, true, window, cx));
            });
            paint(window);
            let draft = edit_numeric(
                &item,
                window,
                NumericAddInput::Augend,
                "not a number expression",
            );
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                0
            );
            assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
            assert!(window.update(|window, cx| draft.read(cx).is_focused(window)));
            let split_addend = split.read_with(window, |item, _| item.addend_editor.clone());
            split_addend.update_in(window, |editor, window, cx| {
                editor.set_text("8", window, cx)
            });
            window.run_until_parked();
            paint(window);
            assert!(window.update(|window, cx| draft.read(cx).is_focused(window)));
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            assert_eq!(
                session.read_with(window, |session, _| session.current_numeric_result()),
                Some(10.)
            );
            assert_eq!(
                draft.read_with(window, |editor, cx| editor.text(cx)),
                "not a number expression"
            );
            workspace.update_in(window, |workspace, window, cx| {
                assert!(workspace.activate_item(&split, true, true, window, cx));
            });
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-RESULT-10").is_some());
            assert_eq!(
                split.read_with(window, |item, cx| item.augend_editor.read(cx).text(cx)),
                "2"
            );
            workspace.update_in(window, |workspace, window, cx| {
                assert!(workspace.activate_item(&item, true, true, window, cx));
            });
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
            edit_numeric(&item, window, NumericAddInput::Augend, "7");
            session.read_with(window, |session, _| {
                assert_eq!(session.revision(), 1);
                assert_eq!(
                    session
                        .numeric_document()
                        .expect("owner")
                        .inputs()
                        .expect("pair"),
                    (2., 8.)
                );
                assert!(
                    session
                        .diagnostic()
                        .expect("stale addressed draft")
                        .contains("revision 0")
                );
            });
            assert_eq!(draft.read_with(window, |editor, cx| editor.text(cx)), "7");
            click(window, "GRAPHITE-NUMERIC-DISCARD-DRAFTS");
            assert_eq!(draft.read_with(window, |editor, cx| editor.text(cx)), "2");
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            assert_eq!(
                session.read_with(window, |session, _| session.current_numeric_result()),
                Some(10.)
            );
            assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_none());
            edit_numeric(&item, window, NumericAddInput::Augend, "7");
            session.read_with(window, |session, _| {
                assert_eq!(session.revision(), 2);
                assert_eq!(session.current_numeric_result(), Some(15.));
                assert_eq!(
                    session
                        .numeric_document()
                        .expect("owner")
                        .inputs()
                        .expect("pair"),
                    (7., 8.)
                );
            });
            other_session.read_with(window, |session, _| {
                assert_eq!(
                    session.numeric_document().expect("second owner").node_id(),
                    NodeId(701)
                );
                assert_eq!(
                    session
                        .numeric_document()
                        .expect("second owner")
                        .document_id_encoding(),
                    813
                );
                assert_eq!(session.revision(), 0);
                assert_eq!(session.current_numeric_result(), Some(50.));
            });
            workspace.update_in(window, |workspace, window, cx| {
                assert!(workspace.activate_item(&other, true, true, window, cx));
            });
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-RESULT-50").is_some());
        }

        #[gpui::test]
        async fn at_r04_r18_numeric_opaque_completions_reject_retries_foreign_successors_cancel_and_released_owner(
            cx: &mut TestAppContext,
        ) {
            let (item, workspace, window) = numeric_workspace(cx, document(814, 4., 5.)).await;
            window.run_until_parked();
            paint(window);
            let session = item.read_with(window, |item, _| item.session.clone());
            let original = session.read_with(window, |session, _| {
                session.numeric_document().expect("owner").snapshot()
            });
            let original_network = original.network().clone();
            let old = session
                .update(window, |session, cx| session.begin_numeric_evaluation(cx))
                .expect("genuine pending retry");
            let current = session
                .update(window, |session, cx| session.begin_numeric_evaluation(cx))
                .expect("same revision successor");
            let old_completion = old.evaluate().await;
            assert!(!session.update(window, |session, cx| {
                session.accept_numeric_evaluation(old_completion, cx)
            }));
            assert!(session.read_with(window, |session, _| session.numeric_pending()));
            let foreign = window.update(|_, cx| {
                cx.new(|_| {
                    GraphicsSession::from_numeric_document(document(814, 4., 5.))
                        .expect("distinct owner, same public labels")
                })
            });
            let foreign_request = foreign
                .update(window, |session, cx| session.begin_numeric_evaluation(cx))
                .expect("foreign genuine request");
            let foreign_completion = foreign_request.evaluate().await;
            assert!(!session.update(window, |session, cx| {
                session.accept_numeric_evaluation(foreign_completion, cx)
            }));
            let current_completion = current.evaluate().await;
            assert!(session.update(window, |session, cx| {
                session.accept_numeric_evaluation(current_completion, cx)
            }));
            assert_eq!(
                session.read_with(window, |session, _| session.current_numeric_result()),
                Some(9.)
            );
            let stale = session
                .update(window, |session, cx| session.begin_numeric_evaluation(cx))
                .expect("real old snapshot");
            assert!(
                session
                    .update(window, |session, cx| session.apply_numeric_command(
                        command(814, 0, NumericAddInput::Addend, 6.),
                        cx
                    ))
                    .expect("source successor")
            );
            let stale_completion = stale.evaluate().await;
            assert!(!session.update(window, |session, cx| {
                session.accept_numeric_evaluation(stale_completion, cx)
            }));
            window.run_until_parked();
            assert_eq!(
                session.read_with(window, |session, _| session.current_numeric_result()),
                Some(10.)
            );
            let cancelled = session
                .update(window, |session, cx| session.begin_numeric_evaluation(cx))
                .expect("pending genuine evaluation");
            session.update(window, |session, cx| session.cancel_numeric_evaluation(cx));
            let cancelled_completion = cancelled.evaluate().await;
            assert!(!session.update(window, |session, cx| {
                session.accept_numeric_evaluation(cancelled_completion, cx)
            }));
            assert!(!session.read_with(window, |session, _| session.numeric_pending()));
            assert!(session.read_with(window, |session, _| {
                session.expression_text().contains("Last valid result: 10")
            }));
            let invalidated = session
                .update(window, |session, cx| session.begin_numeric_evaluation(cx))
                .expect("request before invalid native expression");
            assert!(
                session
                    .update(window, |session, cx| session.submit_numeric_input(
                        1,
                        NumericAddInput::Augend,
                        "1 / 0",
                        cx
                    ))
                    .is_err()
            );
            let invalidated_completion = invalidated.evaluate().await;
            assert!(!session.update(window, |session, cx| {
                session.accept_numeric_evaluation(invalidated_completion, cx)
            }));
            assert_eq!(
                session.read_with(window, |session, _| session.revision()),
                1
            );
            session
                .update(window, |session, cx| session.request_numeric_evaluation(cx))
                .expect("recovery after invalidation");
            window.run_until_parked();
            assert_eq!(
                session.read_with(window, |session, _| session.current_numeric_result()),
                Some(10.)
            );
            assert_eq!(original.network(), &original_network);
            assert_eq!(original.inputs().expect("immutable snapshot"), (4., 5.));
            let weak_item = item.downgrade();
            let weak_session = session.downgrade();
            session
                .update(window, |session, cx| session.request_numeric_evaluation(cx))
                .expect("owned pending request before close");
            let item_id = item.entity_id();
            workspace
                .update_in(window, |workspace, window, cx| {
                    workspace.active_pane().clone().update(cx, |pane, cx| {
                        pane.close_item_by_id(item_id, workspace::SaveIntent::Close, window, cx)
                    })
                })
                .await
                .expect("real native close retires Toolbar owner");
            window.update(|_, _| {
                drop(item);
                drop(session);
            });
            paint(window);
            window.run_until_parked();
            assert!(weak_item.upgrade().is_none());
            assert!(weak_session.upgrade().is_none());
            assert_eq!(
                original
                    .evaluate_network()
                    .await
                    .expect("detached original remains executable"),
                9.
            );
        }

        #[gpui::test]
        async fn at_r04_r18_numeric_validated_bitwise_noop_signed_zero_and_finite_overflow_keep_source_contract(
            cx: &mut TestAppContext,
        ) {
            init(cx);
            let session = cx.new(|_| {
                GraphicsSession::from_numeric_document(document(815, 0., 0.))
                    .expect("supplied finite owner")
            });
            let request = session
                .update(cx, |session, cx| session.begin_numeric_evaluation(cx))
                .expect("original evaluation");
            let completion = request.evaluate().await;
            assert!(session.update(cx, |session, cx| {
                session.accept_numeric_evaluation(completion, cx)
            }));
            let pending = session
                .update(cx, |session, cx| session.begin_numeric_evaluation(cx))
                .expect("same revision genuine pending request");
            assert!(
                !session
                    .update(cx, |session, cx| session.apply_numeric_command(
                        command(815, 0, NumericAddInput::Augend, 0.),
                        cx
                    ))
                    .expect("exact bit no-op")
            );
            let unchanged = session.read_with(cx, |session, _| {
                session.numeric_document().expect("owner").network().clone()
            });
            for (invalid, expected) in [
                (
                    NumericAddCommand {
                        document_id_encoding: 816,
                        ..command(815, 0, NumericAddInput::Augend, 0.)
                    },
                    NumericDocumentError::ForeignDocument,
                ),
                (
                    NumericAddCommand {
                        network_path: vec![NodeId(701)],
                        ..command(815, 0, NumericAddInput::Augend, 0.)
                    },
                    NumericDocumentError::UnsupportedNetworkPath,
                ),
                (
                    NumericAddCommand {
                        target_node_id: NodeId(702),
                        ..command(815, 0, NumericAddInput::Augend, 0.)
                    },
                    NumericDocumentError::MissingNode,
                ),
                (
                    command(815, 1, NumericAddInput::Augend, 0.),
                    NumericDocumentError::StaleRevision {
                        expected: 1,
                        actual: 0,
                    },
                ),
                (
                    command(815, 0, NumericAddInput::Augend, f64::NAN),
                    NumericDocumentError::NonFiniteValue,
                ),
                (
                    command(815, 0, NumericAddInput::Augend, f64::INFINITY),
                    NumericDocumentError::NonFiniteValue,
                ),
            ] {
                assert_eq!(
                    session.update(cx, |session, cx| session.apply_numeric_command(invalid, cx)),
                    Err(expected)
                );
                session.read_with(cx, |session, _| {
                    assert_eq!(session.revision(), 0);
                    assert!(session.numeric_pending());
                    assert_eq!(
                        session.numeric_document().expect("owner").network(),
                        &unchanged
                    );
                });
            }
            let pending_completion = pending.evaluate().await;
            assert!(session.update(cx, |session, cx| {
                session.accept_numeric_evaluation(pending_completion, cx)
            }));
            assert_eq!(
                session.read_with(cx, |session, _| session
                    .current_numeric_result()
                    .expect("original zero")
                    .to_bits()),
                0_f64.to_bits()
            );
            let old = session
                .update(cx, |session, cx| session.begin_numeric_evaluation(cx))
                .expect("old positive-zero snapshot");
            assert!(
                session
                    .update(cx, |session, cx| session.apply_numeric_command(
                        command(815, 0, NumericAddInput::Augend, -0.),
                        cx
                    ))
                    .expect("bit-distinct source edit")
            );
            assert_eq!(
                session.read_with(cx, |session, _| session
                    .numeric_document()
                    .expect("owner")
                    .inputs()
                    .expect("pair")
                    .0
                    .to_bits()),
                (-0_f64).to_bits()
            );
            assert_eq!(session.read_with(cx, |session, _| session.revision()), 1);
            let old_completion = old.evaluate().await;
            assert!(!session.update(cx, |session, cx| {
                session.accept_numeric_evaluation(old_completion, cx)
            }));
            cx.run_until_parked();
            assert!(
                !session
                    .update(cx, |session, cx| session.apply_numeric_command(
                        command(815, 1, NumericAddInput::Augend, -0.),
                        cx
                    ))
                    .expect("exact negative-zero repeat")
            );
            assert_eq!(session.read_with(cx, |session, _| session.revision()), 1);
            assert!(
                session
                    .update(cx, |session, cx| session.apply_numeric_command(
                        command(815, 1, NumericAddInput::Augend, f64::MAX),
                        cx
                    ))
                    .expect("finite first input")
            );
            assert!(
                session
                    .update(cx, |session, cx| session.apply_numeric_command(
                        command(815, 2, NumericAddInput::Addend, f64::MAX),
                        cx
                    ))
                    .expect("finite second input")
            );
            cx.run_until_parked();
            assert_eq!(
                session.read_with(cx, |session, _| session.current_numeric_result()),
                Some(f64::INFINITY)
            );
            assert_eq!(session.read_with(cx, |session, _| session.revision()), 3);
            assert!(
                session
                    .update(cx, |session, cx| session.submit_numeric_input(
                        3,
                        NumericAddInput::Augend,
                        "1e309",
                        cx
                    ))
                    .is_err()
            );
            assert_eq!(session.read_with(cx, |session, _| session.revision()), 3);
            assert!(session.read_with(cx, |session, _| session.current_numeric_result().is_none()));
            assert!(
                session
                    .update(cx, |session, cx| session.submit_numeric_input(
                        3,
                        NumericAddInput::Augend,
                        "2 * 3",
                        cx
                    ))
                    .expect("admitted expression recovery")
            );
            assert!(
                session
                    .update(cx, |session, cx| session.submit_numeric_input(
                        4,
                        NumericAddInput::Addend,
                        "4",
                        cx
                    ))
                    .expect("second field recovery")
            );
            cx.run_until_parked();
            assert_eq!(
                session.read_with(cx, |session, _| session.current_numeric_result()),
                Some(10.)
            );
            assert_eq!(session.read_with(cx, |session, _| session.revision()), 5);
            assert!(session.read_with(cx, |session, _| session.diagnostic().is_none()));
            assert_eq!(unchanged.nodes.len(), 1);
        }
        mod synchronization_corrections {
            use super::*;

            #[gpui::test]
            async fn at_r04_r18_numeric_canonical_seeding_and_clean_sync_preserve_signed_zero_and_pending_authority(
                cx: &mut TestAppContext,
            ) {
                let supplied = document(816, -0., 3.);
                let initial = supplied.snapshot();
                let network = initial.network().clone();
                let (item, _, window) = numeric_workspace(cx, supplied).await;
                let session = item.read_with(window, |item, _| item.session.clone());
                window.run_until_parked();
                paint(window);
                window.run_until_parked();
                session.read_with(window, |session, _| {
                    let document = session
                        .numeric_document()
                        .expect("supplied signed-zero owner");
                    assert_eq!(document.revision(), 0);
                    assert_eq!(
                        document.inputs().expect("pair").0.to_bits(),
                        (-0_f64).to_bits()
                    );
                    assert_eq!(document.network(), &network);
                    assert_eq!(session.current_numeric_result(), Some(3.));
                });
                let augend = item.read_with(window, |item, _| item.augend_editor.clone());
                let addend = item.read_with(window, |item, _| item.addend_editor.clone());
                assert_eq!(augend.read_with(window, |editor, cx| editor.text(cx)), "-0");
                assert!(!window.update(|window, cx| addend.read(cx).is_focused(window)));
                let pending = session
                    .update(window, |session, cx| {
                        assert!(
                            session
                                .apply_numeric_command(
                                    command(816, 0, NumericAddInput::Addend, -0.),
                                    cx
                                )
                                .expect("external addressed signed-zero source change")
                        );
                        session.begin_numeric_evaluation(cx)
                    })
                    .expect("genuine completion authority before canonical synchronization");
                paint(window);
                window.run_until_parked();
                paint(window);
                session.read_with(window, |session, _| {
                    assert_eq!(session.revision(), 1);
                    assert!(session.numeric_pending());
                    assert!(session.current_numeric_result().is_none());
                    let (augend, addend) = session
                        .numeric_document()
                        .expect("owner")
                        .inputs()
                        .expect("exact pair");
                    assert_eq!(augend.to_bits(), (-0_f64).to_bits());
                    assert_eq!(addend.to_bits(), (-0_f64).to_bits());
                });
                assert_eq!(addend.read_with(window, |editor, cx| editor.text(cx)), "-0");
                let completion = pending.evaluate().await;
                assert!(session.update(window, |session, cx| {
                    session.accept_numeric_evaluation(completion, cx)
                }));
                window.run_until_parked();
                paint(window);
                session.read_with(window, |session, _| {
                    assert_eq!(session.revision(), 1);
                    assert!(!session.numeric_pending());
                    assert!(session.diagnostic().is_none());
                    assert_eq!(
                        session
                            .current_numeric_result()
                            .expect("genuine original result")
                            .to_bits(),
                        (-0_f64).to_bits()
                    );
                    let (augend, addend) = session
                        .numeric_document()
                        .expect("owner")
                        .inputs()
                        .expect("pair");
                    assert_eq!(
                        [augend.to_bits(), addend.to_bits()],
                        [(-0_f64).to_bits(); 2]
                    );
                });
                assert!(window.debug_bounds("GRAPHITE-RESULT--0").is_some());
                assert_eq!(initial.network(), &network);
                assert_eq!(initial.revision(), 0);
                assert_eq!(
                    initial.inputs().expect("immutable source").0.to_bits(),
                    (-0_f64).to_bits()
                );
                assert_eq!(
                    initial
                        .evaluate_network()
                        .await
                        .expect("retained original source evaluation"),
                    3.
                );
            }

            #[gpui::test]
            async fn at_r04_r18_numeric_same_item_peer_commit_does_not_rebase_invalid_draft_without_explicit_discard(
                cx: &mut TestAppContext,
            ) {
                let (item, _, window) = numeric_workspace(cx, document(817, 2., 3.)).await;
                window.run_until_parked();
                paint(window);
                let session = item.read_with(window, |item, _| item.session.clone());
                let initial = session.read_with(window, |session, _| {
                    session.numeric_document().expect("owner").snapshot()
                });
                let network = initial.network().clone();
                let draft = edit_numeric(
                    &item,
                    window,
                    NumericAddInput::Augend,
                    "invalid local expression",
                );
                assert!(window.update(|window, cx| draft.read(cx).is_focused(window)));
                assert_eq!(
                    session.read_with(window, |session, _| session.revision()),
                    0
                );
                assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
                edit_numeric(&item, window, NumericAddInput::Addend, "8");
                session.read_with(window, |session, _| {
                    assert_eq!(session.revision(), 1);
                    assert_eq!(
                        session
                            .numeric_document()
                            .expect("owner")
                            .inputs()
                            .expect("pair"),
                        (2., 8.)
                    );
                    assert_eq!(session.current_numeric_result(), Some(10.));
                });
                assert_eq!(
                    draft.read_with(window, |editor, cx| editor.text(cx)),
                    "invalid local expression"
                );
                assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
                let committed = session.read_with(window, |session, _| {
                    session.numeric_document().expect("owner").network().clone()
                });
                edit_numeric(&item, window, NumericAddInput::Augend, "7");
                session.read_with(window, |session, _| {
                    assert_eq!(session.revision(), 1);
                    assert_eq!(
                        session
                            .numeric_document()
                            .expect("owner")
                            .inputs()
                            .expect("pair"),
                        (2., 8.)
                    );
                    assert_eq!(
                        session.numeric_document().expect("owner").network(),
                        &committed
                    );
                    assert!(session.current_numeric_result().is_none());
                    assert!(!session.numeric_pending());
                    assert!(
                        session
                            .diagnostic()
                            .expect("captured revision rejected")
                            .contains("revision 0")
                    );
                    assert!(session.expression_text().contains("Last valid result: 10"));
                });
                assert_eq!(draft.read_with(window, |editor, cx| editor.text(cx)), "7");
                assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_some());
                click(window, "GRAPHITE-NUMERIC-DISCARD-DRAFTS");
                assert_eq!(draft.read_with(window, |editor, cx| editor.text(cx)), "2");
                session.read_with(window, |session, _| {
                    assert_eq!(session.revision(), 1);
                    assert_eq!(
                        session.numeric_document().expect("owner").network(),
                        &committed
                    );
                    assert_eq!(session.current_numeric_result(), Some(10.));
                });
                assert!(window.debug_bounds("GRAPHITE-INPUT-ERROR").is_none());
                edit_numeric(&item, window, NumericAddInput::Augend, "7");
                session.read_with(window, |session, _| {
                    let document = session.numeric_document().expect("owner");
                    assert_eq!(document.revision(), 2);
                    assert_eq!(document.document_id_encoding(), 817);
                    assert_eq!(document.node_id(), NodeId(701));
                    assert_eq!(document.inputs().expect("pair"), (7., 8.));
                    assert_eq!(session.current_numeric_result(), Some(15.));
                    assert!(session.diagnostic().is_none());
                });
                assert!(window.debug_bounds("GRAPHITE-RESULT-15").is_some());
                assert_eq!(initial.inputs().expect("detached initial pair"), (2., 3.));
                assert_eq!(initial.network(), &network);
                assert_eq!(initial.revision(), 0);
            }
        }
    }
}
