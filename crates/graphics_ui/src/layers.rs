use gpui::{App, Entity, Window};
use graphics_core::formats::gdd::{
    GddLayerGrouping, GddLayerOrder, GddRootLayer, GddRootLayerChange,
};
use std::collections::BTreeSet;
use ui::{
    Button, ButtonStyle, Color, ContextMenu, Label, LabelSize, PopoverMenu, TintColor, prelude::*,
};

pub(crate) enum SavedLayerAction {
    Select { extend: bool, range: bool },
    Edit(GddRootLayerChange),
    ToggleExpansion { recursive: bool },
}

pub(crate) fn render_saved_layer_delete(
    enabled: bool,
    dispatch: impl Fn(&mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .debug_selector(|| "GRAPHITE-GDD-LAYER-DELETE".into())
        .child(
            Button::new("graphite-saved-layer-delete", "Delete")
                .disabled(!enabled)
                .on_click(move |_, window, cx| dispatch(window, cx)),
        )
}

pub(crate) fn render_saved_layer_move(
    enabled: bool,
    menu: impl Fn(&mut Window, &mut App) -> Option<Entity<ContextMenu>> + 'static,
) -> impl IntoElement {
    div()
        .debug_selector(|| "GRAPHITE-GDD-LAYER-MOVE".into())
        .child(
            PopoverMenu::new("graphite-saved-layer-move-menu")
                .trigger(Button::new("graphite-saved-layer-move", "Move To").disabled(!enabled))
                .menu(menu),
        )
}

pub(crate) fn render_saved_layer_grouping(
    editing: bool,
    selected: bool,
    dispatch: impl Fn(GddLayerGrouping, &mut Window, &mut App) + Clone + 'static,
) -> impl IntoElement {
    h_flex().gap_2().flex_wrap().children(
        [
            ("NEW", "New Group", GddLayerGrouping::Empty),
            ("GROUP", "Group", GddLayerGrouping::Group),
            ("UNGROUP", "Ungroup", GddLayerGrouping::Ungroup),
        ]
        .into_iter()
        .map(move |(id, label, operation)| {
            let dispatch = dispatch.clone();
            div()
                .debug_selector(move || format!("GRAPHITE-GDD-LAYER-GROUPING-{id}"))
                .child(
                    Button::new(format!("graphite-saved-layer-grouping-{id}"), label)
                        .disabled(
                            !(editing
                                && (selected || matches!(operation, GddLayerGrouping::Empty))),
                        )
                        .on_click(move |_, window, cx| dispatch(operation, window, cx)),
                )
        }),
    )
}

pub(crate) fn render_saved_layer_order(
    enabled: bool,
    dispatch: impl Fn(GddLayerOrder, &mut Window, &mut App) + Clone + 'static,
) -> impl IntoElement {
    h_flex().gap_2().flex_wrap().children(
        [
            ("RAISE", "Raise", GddLayerOrder::Raise),
            ("LOWER", "Lower", GddLayerOrder::Lower),
            ("FRONT", "To Front", GddLayerOrder::Front),
            ("BACK", "To Back", GddLayerOrder::Back),
            ("REVERSE", "Reverse Order", GddLayerOrder::Reverse),
        ]
        .into_iter()
        .map(move |(id, label, order)| {
            let dispatch = dispatch.clone();
            div()
                .debug_selector(move || format!("GRAPHITE-GDD-LAYER-ORDER-{id}"))
                .child(
                    Button::new(format!("graphite-saved-layer-order-{id}"), label)
                        .disabled(!enabled)
                        .on_click(move |_, window, cx| dispatch(order, window, cx)),
                )
        }),
    )
}

pub(crate) fn update_layer_selection(
    order: &[usize],
    index: usize,
    anchor: &mut Option<usize>,
    selection: &mut BTreeSet<usize>,
    extend: bool,
    range: bool,
) {
    if let Some(origin) = anchor.filter(|origin| range && order.contains(origin)) {
        if !extend {
            selection.clear();
        }
        selection.extend([origin, index]);
        selection.extend(
            order
                .iter()
                .copied()
                .skip_while(|&row| row != origin && row != index)
                .skip(1)
                .take_while(|&row| row != origin && row != index),
        );
    } else {
        *anchor = Some(index);
        if !extend {
            selection.clear();
        }
        if !extend || !selection.remove(&index) {
            selection.insert(index);
        }
    }
}

pub(crate) struct SavedLayerRow<'a> {
    pub layer: &'a GddRootLayer,
    pub index: usize,
    pub name: String,
    pub selected: bool,
    pub can_select: bool,
    pub edit_enabled: Option<bool>,
    pub expanded: Option<bool>,
}

pub(crate) fn render_saved_layer(
    row: SavedLayerRow<'_>,
    dispatch: impl Fn(SavedLayerAction, &mut Window, &mut App) + Clone + 'static,
) -> impl IntoElement {
    let SavedLayerRow {
        layer,
        index,
        name,
        selected,
        can_select,
        edit_enabled,
        expanded,
    } = row;
    let select = dispatch.clone();
    let toggle_expansion = dispatch.clone();
    h_flex()
        .gap_2()
        .flex_wrap()
        .pl(px(layer.depth.min(16) as f32 * 16.))
        .child(
            div()
                .debug_selector(move || format!("GRAPHITE-GDD-SAVED-LAYER-{index}"))
                .child(
                    Button::new(("graphite-gdd-saved-layer", index), name.clone())
                        .toggle_state(selected)
                        .selected_style(ButtonStyle::Tinted(TintColor::Accent))
                        .disabled(!can_select)
                        .on_click(move |event, window, cx| {
                            let modifiers = event.modifiers();
                            let opposite_accelerator = if cfg!(target_os = "macos") {
                                modifiers.control
                            } else {
                                modifiers.platform
                            };
                            if !modifiers.alt && !opposite_accelerator {
                                select(
                                    SavedLayerAction::Select {
                                        extend: modifiers.secondary(),
                                        range: modifiers.shift,
                                    },
                                    window,
                                    cx,
                                );
                            }
                        }),
                ),
        )
        .when_some(expanded, |row, expanded| {
            let label = if expanded { "Collapse" } else { "Expand" };
            row.child(
                div()
                    .debug_selector(move || format!("GRAPHITE-GDD-LAYER-EXPANSION-{index}"))
                    .child(
                        Button::new(("graphite-gdd-layer-expansion", index), label)
                            .aria_label(format!("{label} {name}"))
                            .on_click(move |event, window, cx| {
                                let modifiers = event.modifiers();
                                toggle_expansion(
                                    SavedLayerAction::ToggleExpansion {
                                        recursive: modifiers.alt || modifiers.secondary(),
                                    },
                                    window,
                                    cx,
                                );
                            }),
                    ),
            )
        })
        .when(edit_enabled.is_none() && !layer.visible, |row| {
            row.child(
                Label::new("Hidden")
                    .size(LabelSize::XSmall)
                    .color(Color::Muted),
            )
        })
        .when(edit_enabled.is_none() && layer.locked, |row| {
            row.child(
                Label::new("Locked")
                    .size(LabelSize::XSmall)
                    .color(Color::Muted),
            )
        })
        .when_some(edit_enabled, |row, enabled| {
            let visibility = if layer.visible { "Hide" } else { "Show" };
            let locking = if layer.locked { "Unlock" } else { "Lock" };
            let visible = !layer.visible;
            let locked = !layer.locked;
            let set_visibility = dispatch.clone();
            row.child(
                div()
                    .debug_selector(move || format!("GRAPHITE-GDD-LAYER-VISIBILITY-{index}"))
                    .child(
                        Button::new(("graphite-gdd-layer-visibility", index), visibility)
                            .aria_label(format!("{visibility} {name}"))
                            .disabled(!enabled)
                            .on_click(move |_, window, cx| {
                                set_visibility(
                                    SavedLayerAction::Edit(GddRootLayerChange::Visibility(visible)),
                                    window,
                                    cx,
                                );
                            }),
                    ),
            )
            .child(
                div()
                    .debug_selector(move || format!("GRAPHITE-GDD-LAYER-LOCK-{index}"))
                    .child(
                        Button::new(("graphite-gdd-layer-lock", index), locking)
                            .aria_label(format!("{locking} {name}"))
                            .disabled(!enabled)
                            .on_click(move |_, window, cx| {
                                dispatch(
                                    SavedLayerAction::Edit(GddRootLayerChange::Locked(locked)),
                                    window,
                                    cx,
                                );
                            }),
                    ),
            )
        })
}
