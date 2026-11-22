use gpui::{App, Window};
use graphics_core::formats::gdd::GddRootLayer;
use ui::{Button, ButtonStyle, Color, Label, LabelSize, TintColor, prelude::*};

pub(crate) fn render_saved_layer(
    layer: &GddRootLayer,
    index: usize,
    name: String,
    selected: bool,
    enabled: bool,
    select: impl Fn(&mut Window, &mut App) + 'static,
) -> impl IntoElement {
    h_flex()
        .gap_2()
        .pl(px(layer.depth.min(16) as f32 * 16.))
        .child(
            div()
                .debug_selector(move || format!("GRAPHITE-GDD-SAVED-LAYER-{index}"))
                .child(
                    Button::new(("graphite-gdd-saved-layer", index), name)
                        .toggle_state(selected)
                        .selected_style(ButtonStyle::Tinted(TintColor::Accent))
                        .disabled(!enabled)
                        .on_click(move |_, window, cx| select(window, cx)),
                ),
        )
        .when(!layer.visible, |row| {
            row.child(
                Label::new("Hidden")
                    .size(LabelSize::XSmall)
                    .color(Color::Muted),
            )
        })
        .when(layer.locked, |row| {
            row.child(
                Label::new("Locked")
                    .size(LabelSize::XSmall)
                    .color(Color::Muted),
            )
        })
}
