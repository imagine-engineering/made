use editor::Editor;
use gpui::{App, Entity, Window};
use graphics_core::formats::gdd::{
    GddLayerGrouping, GddLayerOpacityKind, GddLayerOrder, GddRootLayer, GddRootLayerChange,
};
use graphite_core_types::BlendMode;
use std::collections::BTreeSet;
use ui::{
    Button, ButtonStyle, Color, ContextMenu, Label, LabelSize, PopoverMenu, TintColor, prelude::*,
};

pub(crate) enum SavedLayerAction {
    Select { extend: bool, range: bool },
    Edit(GddRootLayerChange),
    ToggleExpansion { recursive: bool },
    ToggleClipping,
}

pub(crate) fn render_saved_layer_blend_mode(
    mode: Option<BlendMode>,
    menu: impl Fn(&mut Window, &mut App) -> Option<Entity<ContextMenu>> + 'static,
) -> impl IntoElement {
    h_flex()
        .gap_1()
        .child(Label::new("Blend Mode").size(LabelSize::Small))
        .child(
            div()
                .debug_selector(|| "GRAPHITE-GDD-LAYER-BLEND-MODE".into())
                .child(
                    PopoverMenu::new("graphite-saved-layer-blend-mode-menu")
                        .trigger(Button::new(
                            "graphite-saved-layer-blend-mode",
                            mode.map_or_else(|| "Mixed".into(), |mode| mode.to_string()),
                        ))
                        .menu(menu),
                ),
        )
}

pub(crate) fn render_saved_layer_opacity(
    kind: GddLayerOpacityKind,
    input: &Entity<Editor>,
    cx: &App,
    dispatch: impl Fn(&mut Window, &mut App) + Clone + 'static,
) -> impl IntoElement {
    let (id, label) = match kind {
        GddLayerOpacityKind::Overall => ("OPACITY", "Opacity"),
        GddLayerOpacityKind::Fill => ("FILL-OPACITY", "Fill Opacity"),
    };
    let enter = dispatch.clone();
    h_flex()
        .gap_1()
        .child(Label::new(label).size(LabelSize::Small))
        .child(
            div()
                .id(format!("graphite-saved-layer-{id}-percent"))
                .debug_selector(move || format!("GRAPHITE-GDD-LAYER-{id}-PERCENT"))
                .aria_label(format!("{label} percent"))
                .w(gpui::px(56.))
                .px_1()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border)
                .bg(cx.theme().colors().editor_background)
                .on_key_down(move |event, window, cx| {
                    if event.keystroke.key == "enter"
                        && event.keystroke.modifiers == gpui::Modifiers::default()
                    {
                        cx.stop_propagation();
                        enter(window, cx);
                    }
                })
                .child(input.clone()),
        )
        .child(Label::new("%").size(LabelSize::Small))
        .child(
            div()
                .debug_selector(move || format!("GRAPHITE-GDD-LAYER-{id}-APPLY"))
                .child(
                    Button::new(format!("graphite-saved-layer-{id}-apply"), "Apply")
                        .on_click(move |_, window, cx| dispatch(window, cx)),
                ),
        )
}

pub(crate) fn render_saved_layer_duplicate(
    enabled: bool,
    dispatch: impl Fn(&mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .debug_selector(|| "GRAPHITE-GDD-LAYER-DUPLICATE".into())
        .child(
            Button::new("graphite-saved-layer-duplicate", "Duplicate")
                .disabled(!enabled)
                .on_click(move |_, window, cx| dispatch(window, cx)),
        )
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SavedLayerPlacement {
    Move,
    Duplicate,
}

impl SavedLayerPlacement {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Move => "Move To",
            Self::Duplicate => "Duplicate To",
        }
    }
}

pub(crate) fn render_saved_layer_placement(
    placement: SavedLayerPlacement,
    enabled: bool,
    menu: impl Fn(&mut Window, &mut App) -> Option<Entity<ContextMenu>> + 'static,
) -> impl IntoElement {
    let (menu_id, id, selector) = match placement {
        SavedLayerPlacement::Move => (
            "graphite-saved-layer-move-menu",
            "graphite-saved-layer-move",
            "GRAPHITE-GDD-LAYER-MOVE",
        ),
        SavedLayerPlacement::Duplicate => (
            "graphite-saved-layer-duplicate-to-menu",
            "graphite-saved-layer-duplicate-to",
            "GRAPHITE-GDD-LAYER-DUPLICATE-TO",
        ),
    };
    div().debug_selector(move || selector.into()).child(
        PopoverMenu::new(menu_id)
            .trigger(Button::new(id, placement.label()).disabled(!enabled))
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
    let toggle_clipping = dispatch.clone();
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
        .when(
            edit_enabled.is_none() && layer.clipping == Some(true),
            |row| {
                row.child(
                    Label::new("Clipped")
                        .size(LabelSize::XSmall)
                        .color(Color::Muted),
                )
            },
        )
        .when_some(
            edit_enabled.zip(layer.clipping),
            |row, (enabled, clipped)| {
                let label = if clipped { "Unclip" } else { "Clip" };
                row.child(
                    div()
                        .debug_selector(move || format!("GRAPHITE-GDD-LAYER-CLIP-{index}"))
                        .child(
                            Button::new(("graphite-gdd-layer-clip", index), label)
                                .aria_label(format!("{label} {name}"))
                                .disabled(!enabled || layer.effectively_locked)
                                .on_click(move |_, window, cx| {
                                    toggle_clipping(SavedLayerAction::ToggleClipping, window, cx)
                                }),
                        ),
                )
            },
        )
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
