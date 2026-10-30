use std::sync::mpsc;

use gpui::{AppContext as _, TestAppContext};
use search::collaboration_search::{
    CollaborationResultGroup, CollaborationSearchEvent, CollaborationSearchFreshness,
    CollaborationSearchPresentation, CollaborationSearchView, NativeResultGroup,
    SearchPresentationItem, SearchResultIdentity,
};
use settings::SettingsStore;
use zed_actions::search::{SelectNextMatch, SelectPreviousMatch};

fn init_test(cx: &mut TestAppContext) {
    cx.update(|cx| {
        let settings = SettingsStore::test(cx);
        cx.set_global(settings);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        search::init(cx);
    });
}

fn native_file(identity: &str, label: &str) -> SearchPresentationItem {
    SearchPresentationItem::native(identity, NativeResultGroup::File, label, None)
}

fn collaboration_item(
    identity: &str,
    group: CollaborationResultGroup,
    label: &str,
) -> SearchPresentationItem {
    SearchPresentationItem::collaboration(identity, group, label, None)
}

#[gpui::test]
fn collaboration_search_keyboard_flow(cx: &mut TestAppContext) {
    init_test(cx);
    let window = cx.add_window(|_, cx| {
        CollaborationSearchView::new(
            vec![native_file("file:one", "main.rs")],
            CollaborationSearchPresentation::authorized(
                "Community · Acme",
                CollaborationSearchFreshness::Current,
                vec![collaboration_item(
                    "channel:one",
                    CollaborationResultGroup::Channel,
                    "general",
                )],
            ),
            cx,
        )
    });
    let view = window.root(cx).expect("search view");
    let (sender, receiver) = mpsc::channel();
    let _subscription = cx.update(|cx| {
        cx.subscribe(&view, move |_, event: &CollaborationSearchEvent, _| {
            sender.send(event.clone()).expect("confirmation receiver");
        })
    });

    window
        .update(cx, |_, window, cx| cx.focus_self(window))
        .expect("focus search view");
    cx.run_until_parked();
    assert_eq!(
        cx.read(|cx| {
            view.read(cx)
                .selected_item()
                .map(|item| item.identity.clone())
        }),
        Some(SearchResultIdentity::Native("file:one".into()))
    );

    cx.dispatch_action(window.into(), SelectNextMatch);
    assert_eq!(
        cx.read(|cx| {
            view.read(cx)
                .selected_item()
                .map(|item| item.identity.clone())
        }),
        Some(SearchResultIdentity::Collaboration("channel:one".into()))
    );
    cx.dispatch_action(window.into(), SelectPreviousMatch);
    cx.dispatch_action(window.into(), menu::Confirm);
    assert_eq!(
        receiver.try_recv().expect("confirmed event"),
        CollaborationSearchEvent::Confirmed(SearchResultIdentity::Native("file:one".into()))
    );
}

#[gpui::test]
fn collaboration_search_empty_and_stale_states(cx: &mut TestAppContext) {
    init_test(cx);
    let empty = cx.new(|cx| {
        CollaborationSearchView::new(
            Vec::new(),
            CollaborationSearchPresentation::authorized(
                "Community · Acme",
                CollaborationSearchFreshness::Current,
                Vec::new(),
            ),
            cx,
        )
    });
    assert!(cx.read(|cx| empty.read(cx).ordered_items().is_empty()));
    assert_eq!(
        cx.read(|cx| empty.read(cx).collaboration_status_label()),
        "No collaboration results in Community · Acme"
    );

    empty.update(cx, |view, cx| {
        view.update_results(
            Vec::new(),
            CollaborationSearchPresentation::authorized(
                "Community · Acme",
                CollaborationSearchFreshness::Lagging {
                    affected_checkpoints: 2,
                },
                vec![collaboration_item(
                    "member:one",
                    CollaborationResultGroup::Member,
                    "Ada",
                )],
            ),
            cx,
        );
    });
    assert_eq!(
        cx.read(|cx| empty.read(cx).collaboration_status_label()),
        "Community · Acme · Results may be stale · 2 sources behind"
    );
}

#[gpui::test]
fn collaboration_search_unauthorized_state_exposes_no_rows_or_scope(cx: &mut TestAppContext) {
    init_test(cx);
    let view = cx.new(|cx| {
        CollaborationSearchView::new(
            vec![
                native_file("file:one", "main.rs"),
                collaboration_item(
                    "channel:untrusted",
                    CollaborationResultGroup::Channel,
                    "must not render",
                ),
            ],
            CollaborationSearchPresentation::unauthorized(),
            cx,
        )
    });

    assert_eq!(cx.read(|cx| view.read(cx).ordered_items().len()), 1);
    assert!(cx.read(|cx| matches!(
        view.read(cx).ordered_items()[0].identity,
        SearchResultIdentity::Native(_)
    )));
    assert_eq!(
        cx.read(|cx| view.read(cx).collaboration_status_label()),
        "Collaboration results unavailable"
    );
}

#[gpui::test]
fn collaboration_search_preserves_stable_selection_across_refresh(cx: &mut TestAppContext) {
    init_test(cx);
    let window = cx.add_window(|_, cx| {
        CollaborationSearchView::new(
            Vec::new(),
            CollaborationSearchPresentation::authorized(
                "Community · Acme",
                CollaborationSearchFreshness::Current,
                vec![
                    collaboration_item("channel:one", CollaborationResultGroup::Channel, "general"),
                    collaboration_item("member:one", CollaborationResultGroup::Member, "Ada"),
                ],
            ),
            cx,
        )
    });
    let view = window.root(cx).expect("search view");
    window
        .update(cx, |_, window, cx| cx.focus_self(window))
        .expect("focus search view");
    cx.run_until_parked();
    cx.dispatch_action(window.into(), SelectNextMatch);

    view.update(cx, |view, cx| {
        view.update_results(
            Vec::new(),
            CollaborationSearchPresentation::authorized(
                "Community · Acme",
                CollaborationSearchFreshness::Current,
                vec![
                    collaboration_item(
                        "community:one",
                        CollaborationResultGroup::Community,
                        "Acme",
                    ),
                    collaboration_item("channel:one", CollaborationResultGroup::Channel, "general"),
                    collaboration_item(
                        "member:one",
                        CollaborationResultGroup::Member,
                        "Ada Lovelace",
                    ),
                ],
            ),
            cx,
        );
    });

    assert_eq!(
        cx.read(|cx| {
            view.read(cx)
                .selected_item()
                .map(|item| item.identity.clone())
        }),
        Some(SearchResultIdentity::Collaboration("member:one".into()))
    );
}
