use std::collections::BTreeSet;

use editor::Editor;
use gpui::{AppContext as _, Context, Entity, Focusable, Render, Subscription, Window};
use graphics_core::registry::{GraphiteCatalogCategory, GraphiteNodeCatalog};
use ui::{Button, Label, prelude::*};

pub(crate) struct NodeCatalog {
    search: Entity<Editor>,
    source: GraphiteNodeCatalog,
    query: String,
    groups: Vec<GraphiteCatalogCategory>,
    toggled_categories: BTreeSet<&'static str>,
    _search_subscription: Subscription,
}

impl NodeCatalog {
    pub(crate) fn new(
        source: GraphiteNodeCatalog,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let search = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("Search Nodes", window, cx);
            editor
        });
        search.focus_handle(cx).focus(window, cx);
        let search_subscription = cx.observe(&search, |catalog, editor, cx| {
            let query = editor.read(cx).text(cx);
            if query != catalog.query {
                catalog.groups = catalog.source.search(&query);
                catalog.query = query;
                catalog.toggled_categories.clear();
                cx.notify();
            }
        });
        Self {
            search,
            query: String::new(),
            groups: source.search(""),
            source,
            toggled_categories: BTreeSet::new(),
            _search_subscription: search_subscription,
        }
    }
}

impl Render for NodeCatalog {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let searching = !self.query.is_empty();
        v_flex()
            .gap_2()
            .child(
                div()
                    .id("graphite-node-search")
                    .aria_label("Search Nodes")
                    .debug_selector(|| "GRAPHITE-NODE-CATALOG-SEARCH".into())
                    .child(self.search.clone()),
            )
            .child(
                v_flex()
                    .id("graphite-node-catalog-results")
                    .max_h(rems(20.))
                    .overflow_y_scroll()
                    .gap_1()
                    .when(self.groups.is_empty(), |results| {
                        results.child(
                            div()
                                .debug_selector(|| "GRAPHITE-NODE-CATALOG-EMPTY".into())
                                .child(Label::new("No search results")),
                        )
                    })
                    .children(self.groups.iter().enumerate().map(|(index, group)| {
                        let name = group.name;
                        let open = searching ^ self.toggled_categories.contains(name);
                        v_flex()
                            .child(
                                div()
                                    .debug_selector(move || {
                                        format!("GRAPHITE-NODE-CATALOG-CATEGORY-{name}")
                                    })
                                    .child(
                                        Button::new(("graphite-catalog-category", index), name)
                                            .on_click(cx.listener(move |catalog, _, _, cx| {
                                                if !catalog.toggled_categories.remove(name) {
                                                    catalog.toggled_categories.insert(name);
                                                }
                                                cx.notify();
                                            })),
                                    ),
                            )
                            .when(open, |category| {
                                category.children(group.nodes.iter().map(|node| {
                                    let name = node.name.as_str();
                                    div()
                                        .pl_2()
                                        .debug_selector(move || {
                                            format!("GRAPHITE-NODE-CATALOG-NODE-{name}")
                                        })
                                        .child(Label::new(name))
                                }))
                            })
                    })),
            )
    }
}
