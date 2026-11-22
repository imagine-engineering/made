use gpui::{AnyElement, PathBuilder, Pixels, Point, Rgba, Styled, canvas};

use crate::prelude::IntoElement;

pub struct GraphWireStroke {
    pub points: Vec<Point<Pixels>>,
    pub width: Pixels,
    pub color: Rgba,
}

pub fn graph_wire_layer(wires: Vec<GraphWireStroke>) -> AnyElement {
    canvas(
        move |_, _, _| {},
        move |bounds, _, window, _| {
            for wire in &wires {
                let Some((first, remaining)) = wire.points.split_first() else {
                    continue;
                };
                let mut builder = PathBuilder::stroke(wire.width);
                builder.move_to(bounds.origin + *first);
                for point in remaining {
                    builder.line_to(bounds.origin + *point);
                }
                if let Ok(path) = builder.build() {
                    window.paint_path(path, wire.color);
                }
            }
        },
    )
    .absolute()
    .size_full()
    .into_any_element()
}
