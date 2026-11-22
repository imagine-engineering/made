use gpui::{App, Window};
use graphics_core::formats::gdd::{GddRootLayer, GddRootLayerChange};
use ui::{Button, ButtonStyle, Color, Label, LabelSize, TintColor, prelude::*};

pub(crate) enum SavedLayerAction {
    Select,
    Edit(GddRootLayerChange),
}

pub(crate) fn render_saved_layer(
    layer: &GddRootLayer,
    index: usize,
    name: String,
    selected: bool,
    can_select: bool,
    edit_enabled: Option<bool>,
    dispatch: impl Fn(SavedLayerAction, &mut Window, &mut App) + Clone + 'static,
) -> impl IntoElement {
    let select = dispatch.clone();
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
                        .on_click(move |_, window, cx| {
                            select(SavedLayerAction::Select, window, cx)
                        }),
                ),
        )
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
