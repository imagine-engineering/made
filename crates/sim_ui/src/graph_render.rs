use crate::{
    GeneratedGraphContextInfrastructure, GraphActionInput, GraphContextTarget, GraphModelAction,
    GraphWorkspaceItem, QueueOverlayTab, build_graph_context_menu,
    require_graph_context_infrastructure,
};
use gpui::{
    Action as _, AnyElement, Context, FocusHandle, InteractiveElement as _, KeyDownEvent,
    MouseButton, MouseDownEvent, MouseMoveEvent, ParentElement as _, PathBuilder, Rgba, Role,
    ScrollWheelEvent, StatefulInteractiveElement as _, Styled as _, Toggled, Window, canvas, point,
    px, rgb,
};
use serde_json::Value;
use sim_runtime::{
    GraphCommand, GraphDocument, GraphIdentifier, GraphLevel, GraphLink, GraphNode, GraphNodeMode,
    GraphPoint, GraphRect, GraphSize, GraphSlotDirection, GraphWidget, GraphWidgetKind,
    LayoutOperation, SelectionMode, SubgraphDefinition, SubgraphPort, WidgetValidation,
};
use std::collections::BTreeSet;
use ui::prelude::*;

pub(crate) fn render_graph_item(
    item: &mut GraphWorkspaceItem,
    window: &mut Window,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    item.begin_control_focus_handle_render();
    let window_viewport = window.viewport_size();
    let available_viewport = GraphSize {
        width: window_viewport.width.into(),
        height: window_viewport.height.into(),
    };
    let active_execution = item.active_execution_presentation(cx);
    let execution_snapshot = item.execution_snapshot(cx);
    let execute_focus_handle = item.control_focus_handle("execution:execute-button", cx);
    let execute_output_feedback_active = item.qpov2_enabled
        && (item.execute_output_feedback_hovered() || execute_focus_handle.is_focused(window));
    let queue_unavailable_reason = item.execution_queue_unavailable_reason(cx);
    let focus_handle = item.focus_handle.clone();
    let mut root = div()
        .id("sim-native-graph")
        .debug_selector(|| "SIM-GRAPH".into())
        .role(Role::Application)
        .aria_label("Native Sim workflow graph")
        .key_context(crate::SIM_GRAPH_KEY_CONTEXT)
        .track_focus(&focus_handle)
        .on_action(cx.listener(GraphWorkspaceItem::graph_undo))
        .on_action(cx.listener(GraphWorkspaceItem::graph_redo))
        .on_action(cx.listener(GraphWorkspaceItem::graph_copy))
        .on_action(cx.listener(GraphWorkspaceItem::graph_cut))
        .on_action(cx.listener(GraphWorkspaceItem::graph_paste))
        .on_action(cx.listener(GraphWorkspaceItem::graph_delete))
        .on_action(cx.listener(GraphWorkspaceItem::graph_select_all))
        .on_action(cx.listener(GraphWorkspaceItem::graph_zoom_in))
        .on_action(cx.listener(GraphWorkspaceItem::graph_zoom_out))
        .on_action(cx.listener(GraphWorkspaceItem::graph_fit_view))
        .on_action(cx.listener(GraphWorkspaceItem::graph_cancel_gesture))
        .on_action(cx.listener(GraphWorkspaceItem::shell_queue_prompt))
        .on_action(cx.listener(GraphWorkspaceItem::shell_queue_prompt_front))
        .on_action(cx.listener(GraphWorkspaceItem::shell_queue_selected_output_nodes))
        .on_action(cx.listener(GraphWorkspaceItem::shell_interrupt))
        .on_action(cx.listener(GraphWorkspaceItem::shell_clear_pending_tasks))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_queue_overlay))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_qpov2))
        .on_action(cx.listener(GraphWorkspaceItem::execution_run_manual))
        .on_action(cx.listener(GraphWorkspaceItem::execution_run_on_change))
        .on_action(cx.listener(GraphWorkspaceItem::execution_run_instant_idle))
        .on_action(cx.listener(GraphWorkspaceItem::restore_execution_navigation_action))
        .on_action(cx.listener(GraphWorkspaceItem::shell_refresh_node_definitions))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_workflows_sidebar))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_node_library_sidebar))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_model_library_sidebar))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_assets_sidebar))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_linear))
        .on_action(cx.listener(GraphWorkspaceItem::shell_save_workflow))
        .on_action(cx.listener(GraphWorkspaceItem::shell_open_workflow))
        .on_action(cx.listener(GraphWorkspaceItem::shell_group_selected_nodes))
        .on_action(cx.listener(GraphWorkspaceItem::shell_show_settings))
        .on_action(cx.listener(GraphWorkspaceItem::shell_show_keybindings))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_selected_items_pin))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_selected_collapse))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_selected_bypass))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_selected_mute))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_logs_panel))
        .on_action(cx.listener(GraphWorkspaceItem::shell_convert_to_subgraph))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_minimap))
        .on_action(cx.listener(GraphWorkspaceItem::shell_unlock_canvas))
        .on_action(cx.listener(GraphWorkspaceItem::shell_lock_canvas))
        .on_action(cx.listener(GraphWorkspaceItem::shell_exit_subgraph))
        .on_action(cx.listener(GraphWorkspaceItem::shell_paste_with_connect))
        .on_action(cx.listener(GraphWorkspaceItem::shell_move_selected_down))
        .on_action(cx.listener(GraphWorkspaceItem::shell_move_selected_left))
        .on_action(cx.listener(GraphWorkspaceItem::shell_move_selected_right))
        .on_action(cx.listener(GraphWorkspaceItem::shell_move_selected_up))
        .on_action(cx.listener(GraphWorkspaceItem::shell_reset_view))
        .on_action(cx.listener(GraphWorkspaceItem::shell_resize_selected_nodes))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_link_visibility))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_canvas_lock))
        .on_action(cx.listener(GraphWorkspaceItem::shell_toggle_selected_nodes_pin))
        .on_action(cx.listener(GraphWorkspaceItem::shell_fit_group_to_contents))
        .on_action(cx.listener(GraphWorkspaceItem::shell_unpack_subgraph))
        .on_action(cx.listener(GraphWorkspaceItem::shell_publish_subgraph))
        .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
            if is_context_menu_keystroke(event) {
                cx.stop_propagation();
                open_keyboard_graph_context_menu(this, window, cx);
            } else if event.keystroke.key == "escape"
                && (this.drag_anchor.is_some()
                    || this.box_anchor.is_some()
                    || this.pending_link.is_some()
                    || this.canvas_pan_anchor.is_some()
                    || this.subgraph_publish_task.is_some())
            {
                cx.stop_propagation();
                this.cancel_gesture(cx);
            } else if this.focus_handle.is_focused(window)
                && event.keystroke.modifiers.alt
                && event.keystroke.key == "l"
            {
                cx.stop_propagation();
                this.execute_model_action(
                    GraphModelAction::LayoutSelection,
                    GraphActionInput::Layout {
                        operation: LayoutOperation::AlignTop,
                        spacing: 24.0,
                    },
                    cx,
                );
            }
        }))
        .relative()
        .size_full()
        .overflow_hidden()
        .bg(rgb(0x17191d))
        .text_color(rgb(0xe7e9ed));

    let active_subgraph_definition = item
        .model
        .document()
        .and_then(|document| document.active_subgraph_definition().ok())
        .cloned();
    if let Some(graph) = item
        .model
        .document()
        .and_then(|document| document.active_graph().ok())
        .cloned()
    {
        let hidden_nodes = collapsed_group_nodes(&graph);
        let breadcrumbs = item
            .model
            .document()
            .map(subgraph_breadcrumbs)
            .unwrap_or_default();
        let active_execution_for_nodes = active_execution.as_ref();
        let subgraph_boundary_slots = active_subgraph_definition
            .as_ref()
            .map(|definition| {
                render_subgraph_boundary_slots(
                    definition,
                    f32::from(window_viewport.width),
                    item,
                    window,
                    cx,
                )
            })
            .unwrap_or_default();
        let pointer_link_routes = if graph.viewport.links_visible {
            graph
                .links
                .values()
                .filter(|link| {
                    !hidden_nodes.contains(&link.origin_node)
                        && !hidden_nodes.contains(&link.target_node)
                })
                .filter_map(|link| {
                    link_route_points(&graph, link, item.drag_delta)
                        .map(|points| (link.identifier.clone(), points))
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        root = root
            .child(render_grid(&graph))
            .children(graph.groups.values().cloned().map(|group| {
                let viewport = graph.viewport.clone();
                let selected = graph.selection.groups.contains(&group.identifier);
                let mut origin = viewport.graph_to_screen(group.bounds.origin);
                if selected {
                    origin = origin.translated(item.drag_delta);
                }
                let identifier = group.identifier.clone();
                let keyboard_identifier = identifier.clone();
                let keyboard_context_identifier = identifier.clone();
                let mouse_context_identifier = identifier.clone();
                let context_position = GraphPoint {
                    x: origin.x + 16.0,
                    y: origin.y + 16.0,
                };
                let focus_handle =
                    item.control_focus_handle(format!("group:{}", identifier.text()), cx);
                let focused = focus_handle.is_focused(window);
                let mouse_focus_handle = focus_handle.clone();
                let context_focus_handle = focus_handle.clone();
                let group_font_size = group
                    .source_fields
                    .get("font_size")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .unwrap_or(20.0) as f32;
                let group_background = group
                    .color
                    .as_deref()
                    .and_then(parse_hex_color)
                    .unwrap_or_else(|| rgb(0x202530))
                    .opacity(0.28);
                div()
                    .id(SharedString::from(format!(
                        "sim-group-{}",
                        group.identifier.text()
                    )))
                    .debug_selector({
                        let identifier = group.identifier.text();
                        move || format!("SIM-GROUP-{identifier}")
                    })
                    .track_focus(&focus_handle)
                    .tab_stop(true)
                    .role(Role::Group)
                    .aria_label(format!(
                        "Group {}, {} nodes{}{}",
                        group.title,
                        group.node_ids.len(),
                        if group.collapsed { ", collapsed" } else { "" },
                        if group.pinned { ", pinned" } else { "" }
                    ))
                    .aria_selected(selected)
                    .absolute()
                    .left(px(origin.x))
                    .top(px(origin.y))
                    .w(px(group.bounds.size.width * viewport.scale))
                    .h(px(group.bounds.size.height * viewport.scale))
                    .rounded_lg()
                    .border_2()
                    .border_color(if focused {
                        rgb(0xffffff)
                    } else if selected {
                        rgb(0x72a7ff)
                    } else {
                        rgb(0x4d5668)
                    })
                    .bg(group_background)
                    .p_2()
                    .cursor_pointer()
                    .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
                        if is_context_menu_keystroke(event) {
                            cx.stop_propagation();
                            this.open_graph_context_menu(
                                GraphContextTarget::Group(keyboard_context_identifier.clone()),
                                context_position,
                                window,
                                cx,
                            );
                            return;
                        }
                        match event.keystroke.key.as_str() {
                            "enter" | "space" => {
                                cx.stop_propagation();
                                this.select_group(
                                    keyboard_identifier.clone(),
                                    if event.keystroke.modifiers.shift {
                                        SelectionMode::Toggle
                                    } else {
                                        SelectionMode::Replace
                                    },
                                    cx,
                                );
                            }
                            "f" => {
                                cx.stop_propagation();
                                this.execute_catalog_action(
                                    sim_runtime::CatalogGraphAction::FitGroupToContents,
                                    GraphActionInput::GroupIdentifier(keyboard_identifier.clone()),
                                    cx,
                                );
                            }
                            "u" => {
                                cx.stop_propagation();
                                this.execute_model_action(
                                    GraphModelAction::Ungroup,
                                    GraphActionInput::GroupIdentifier(keyboard_identifier.clone()),
                                    cx,
                                );
                            }
                            _ => {}
                        }
                    }))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                            cx.stop_propagation();
                            window.focus(&mouse_focus_handle, cx);
                            let already_selected = this
                                .model
                                .selection()
                                .is_some_and(|selection| selection.groups.contains(&identifier));
                            if event.modifiers.shift || !already_selected {
                                this.select_group(
                                    identifier.clone(),
                                    if event.modifiers.shift {
                                        SelectionMode::Toggle
                                    } else {
                                        SelectionMode::Replace
                                    },
                                    cx,
                                );
                            }
                            this.begin_selection_drag(GraphPoint {
                                x: event.position.x.into(),
                                y: event.position.y.into(),
                            });
                        }),
                    )
                    .capture_any_mouse_down(cx.listener(
                        move |this, event: &MouseDownEvent, window, cx| {
                            if event.button != MouseButton::Right {
                                return;
                            }
                            window.focus(&context_focus_handle, cx);
                            this.stage_pointer_graph_context_target(GraphContextTarget::Group(
                                mouse_context_identifier.clone(),
                            ));
                        },
                    ))
                    .child(
                        div()
                            .text_size(px((group_font_size * viewport.scale).max(6.0)))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child(group.title),
                    )
            }))
            .when(graph.viewport.links_visible, |root| {
                root.child(render_links(&graph, &hidden_nodes, item.drag_delta))
                    .children(render_link_hit_targets(
                        &graph,
                        &hidden_nodes,
                        item.drag_delta,
                        item,
                        cx,
                    ))
            })
            .children(graph.nodes.values().filter_map(|node| {
                (!hidden_nodes.contains(&node.identifier)).then(|| {
                    render_node(
                        node.clone(),
                        &graph,
                        item.drag_delta,
                        item.native_node_renderer_enabled(cx),
                        active_execution_for_nodes,
                        execute_output_feedback_active,
                        item,
                        window,
                        cx,
                    )
                })
            }))
            .children(graph.reroutes.values().map(|reroute| {
                let viewport = graph.viewport.clone();
                let mut position = viewport.graph_to_screen(reroute.position);
                let selected = graph.selection.reroutes.contains(&reroute.identifier);
                if selected {
                    position = position.translated(item.drag_delta);
                }
                let identifier = reroute.identifier.clone();
                let keyboard_identifier = identifier.clone();
                let keyboard_context_identifier = identifier.clone();
                let mouse_context_identifier = identifier.clone();
                let keyboard_position = reroute.position;
                let keyboard_parent = reroute.parent.clone();
                let reparent_target = graph
                    .selection
                    .reroutes
                    .iter()
                    .chain(graph.reroutes.keys())
                    .find(|candidate| *candidate != &identifier)
                    .cloned();
                let focus_handle =
                    item.control_focus_handle(format!("reroute:{}", identifier.text()), cx);
                let focused = focus_handle.is_focused(window);
                let mouse_focus_handle = focus_handle.clone();
                let context_focus_handle = focus_handle.clone();
                let show_type = item.context_reroute_type_visible(&identifier, cx);
                let type_label = graph
                    .resolve_reroute_port_type(&identifier)
                    .map(|port_type| port_type.display_name())
                    .unwrap_or_else(|_| "*".to_owned());
                let type_aria_label = graph
                    .resolve_reroute_port_type(&identifier)
                    .map(|port_type| format!(", type {}", port_type.display_name()))
                    .unwrap_or_default();
                let hit_width = if show_type {
                    28.0 + type_label.chars().count() as f32 * 7.0
                } else {
                    16.0
                };
                div()
                    .id(SharedString::from(format!(
                        "sim-reroute-{}",
                        reroute.identifier.text()
                    )))
                    .debug_selector({
                        let identifier = reroute.identifier.text();
                        move || format!("SIM-REROUTE-{identifier}")
                    })
                    .track_focus(&focus_handle)
                    .tab_stop(true)
                    .role(Role::Button)
                    .aria_label(format!(
                        "Reroute {}{}",
                        reroute.identifier.text(),
                        type_aria_label
                    ))
                    .aria_selected(selected)
                    .absolute()
                    .left(px(position.x - 8.0))
                    .top(px(position.y - 8.0))
                    .w(px(hit_width))
                    .h(px(16.0))
                    .rounded_full()
                    .border_2()
                    .border_color(if focused {
                        rgb(0xffffff)
                    } else if selected {
                        rgb(0xffffff)
                    } else {
                        rgb(0x7ca6ff)
                    })
                    .bg(rgb(0x355287))
                    .cursor_pointer()
                    .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
                        if is_context_menu_keystroke(event) {
                            cx.stop_propagation();
                            this.open_graph_context_menu(
                                GraphContextTarget::Reroute(keyboard_context_identifier.clone()),
                                position,
                                window,
                                cx,
                            );
                            return;
                        }
                        match event.keystroke.key.as_str() {
                            "enter" | "space" => {
                                cx.stop_propagation();
                                this.select_reroute(
                                    keyboard_identifier.clone(),
                                    if event.keystroke.modifiers.shift {
                                        SelectionMode::Toggle
                                    } else {
                                        SelectionMode::Replace
                                    },
                                    cx,
                                );
                            }
                            "delete" | "backspace" => {
                                cx.stop_propagation();
                                this.execute_model_action(
                                    GraphModelAction::RemoveReroute,
                                    GraphActionInput::RerouteIdentifier(
                                        keyboard_identifier.clone(),
                                    ),
                                    cx,
                                );
                            }
                            "left" | "right" | "up" | "down" => {
                                cx.stop_propagation();
                                let delta = match event.keystroke.key.as_str() {
                                    "left" => GraphPoint { x: -10.0, y: 0.0 },
                                    "right" => GraphPoint { x: 10.0, y: 0.0 },
                                    "up" => GraphPoint { x: 0.0, y: -10.0 },
                                    _ => GraphPoint { x: 0.0, y: 10.0 },
                                };
                                this.execute_model_action(
                                    GraphModelAction::MoveReroute,
                                    GraphActionInput::RerouteMove {
                                        identifier: keyboard_identifier.clone(),
                                        position: keyboard_position.translated(delta),
                                    },
                                    cx,
                                );
                            }
                            "p" if keyboard_parent.is_some() || reparent_target.is_some() => {
                                cx.stop_propagation();
                                this.execute_model_action(
                                    GraphModelAction::ReparentReroute,
                                    GraphActionInput::RerouteParent {
                                        identifier: keyboard_identifier.clone(),
                                        parent: if keyboard_parent.is_some() {
                                            None
                                        } else {
                                            reparent_target.clone()
                                        },
                                    },
                                    cx,
                                );
                            }
                            _ => {}
                        }
                    }))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                            cx.stop_propagation();
                            window.focus(&mouse_focus_handle, cx);
                            let already_selected = this
                                .model
                                .selection()
                                .is_some_and(|selection| selection.reroutes.contains(&identifier));
                            if event.modifiers.shift || !already_selected {
                                this.select_reroute(
                                    identifier.clone(),
                                    if event.modifiers.shift {
                                        SelectionMode::Toggle
                                    } else {
                                        SelectionMode::Replace
                                    },
                                    cx,
                                );
                            }
                            this.begin_selection_drag(GraphPoint {
                                x: event.position.x.into(),
                                y: event.position.y.into(),
                            });
                        }),
                    )
                    .capture_any_mouse_down(cx.listener(
                        move |this, event: &MouseDownEvent, window, cx| {
                            if event.button != MouseButton::Right {
                                return;
                            }
                            window.focus(&context_focus_handle, cx);
                            this.stage_pointer_graph_context_target(GraphContextTarget::Reroute(
                                mouse_context_identifier.clone(),
                            ));
                        },
                    ))
                    .when(show_type, |this| {
                        this.child(
                            div()
                                .absolute()
                                .left_4()
                                .top(px(-4.0))
                                .rounded_sm()
                                .bg(rgb(0x252932))
                                .px_1()
                                .text_xs()
                                .child(type_label),
                        )
                    })
            }))
            .children(subgraph_boundary_slots)
            .when(graph.viewport.minimap_visible, |root| {
                root.child(render_minimap(
                    &graph,
                    &hidden_nodes,
                    available_viewport,
                    cx,
                ))
            })
            .when(!breadcrumbs.is_empty(), |root| {
                root.child(render_breadcrumbs(breadcrumbs, cx))
            })
            .when_some(pending_link_points(item, &graph), |root, points| {
                root.child(render_pending_link(points))
            })
            .when(item.model.canvas_info_visible, |root| {
                root.child(
                    div()
                        .absolute()
                        .left_3()
                        .bottom_3()
                        .rounded_md()
                        .bg(rgb(0x252932))
                        .border_1()
                        .border_color(rgb(0x424957))
                        .px_3()
                        .py_2()
                        .text_xs()
                        .child(format!(
                            "{} nodes · {} links · {:.0}%",
                            graph.nodes.len(),
                            graph.links.len(),
                            graph.viewport.scale * 100.0
                        )),
                )
            })
            .when_some(selection_box(item), |root, bounds| {
                root.child(
                    div()
                        .id("sim-selection-box")
                        .debug_selector(|| "SIM-SELECTION-BOX".into())
                        .absolute()
                        .left(px(bounds.origin.x))
                        .top(px(bounds.origin.y))
                        .w(px(bounds.size.width))
                        .h(px(bounds.size.height))
                        .border_1()
                        .border_color(rgb(0x72a7ff))
                        .bg(rgb(0x477dcc).opacity(0.16)),
                )
            })
            .on_scroll_wheel(cx.listener(handle_scroll_wheel))
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, window, cx| {
                let position = GraphPoint {
                    x: event.position.x.into(),
                    y: event.position.y.into(),
                };
                this.update_pending_link_position(position, cx);
                if this.pending_link.is_some()
                    && let Some(delta) = edge_pan_delta(position, window)
                {
                    this.pan_viewport(delta, cx);
                }
                if event.pressed_button == Some(MouseButton::Left) {
                    if this.canvas_is_locked() && this.canvas_pan_anchor.is_some() {
                        this.update_canvas_pan(position, cx);
                    } else if this.drag_anchor.is_some() {
                        this.update_selection_drag(position, cx);
                    } else {
                        this.update_box_selection(position, cx);
                    }
                }
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    if this.canvas_is_locked() && this.canvas_pan_anchor.is_some() {
                        this.finish_canvas_pan(cx);
                    } else if this.pending_link.is_some() {
                        this.reject_pending_link("link drop did not target a compatible input", cx);
                    } else if this.drag_anchor.is_some() {
                        this.finish_selection_drag(cx);
                    } else {
                        this.finish_box_selection(cx);
                    }
                }),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                    let position = GraphPoint {
                        x: event.position.x.into(),
                        y: event.position.y.into(),
                    };
                    if this.canvas_is_locked() {
                        this.begin_canvas_pan(position, cx);
                        return;
                    }
                    if let Some(identifier) = link_at_screen_point(&pointer_link_routes, position) {
                        cx.stop_propagation();
                        this.select_link(
                            identifier,
                            if event.modifiers.shift {
                                SelectionMode::Toggle
                            } else {
                                SelectionMode::Replace
                            },
                            cx,
                        );
                        return;
                    }
                    this.begin_box_selection(
                        position,
                        if event.modifiers.shift {
                            SelectionMode::Toggle
                        } else {
                            SelectionMode::Replace
                        },
                        cx,
                    );
                }),
            )
            .capture_any_mouse_down(cx.listener(|this, event: &MouseDownEvent, window, cx| {
                if event.button != MouseButton::Right {
                    return;
                }
                let screen_position = GraphPoint {
                    x: event.position.x.into(),
                    y: event.position.y.into(),
                };
                let Some(target) = this
                    .model
                    .document()
                    .and_then(|document| document.active_graph().ok())
                    .map(|graph| GraphContextTarget::Canvas {
                        graph_position: graph.viewport.screen_to_graph(screen_position),
                    })
                else {
                    this.model.report_error("active graph is unavailable");
                    cx.notify();
                    return;
                };
                if this.pending_pointer_context_target.is_none() {
                    window.focus(&this.focus_handle, cx);
                    this.stage_pointer_graph_context_target(target);
                }
            }));
    } else {
        let diagnostic = item
            .model
            .read_only_diagnostic()
            .unwrap_or("workflow cannot be edited")
            .to_owned();
        root = root.child(
            v_flex()
                .id("sim-read-only")
                .size_full()
                .items_center()
                .justify_center()
                .gap_3()
                .role(Role::Document)
                .aria_label(format!("Read-only workflow: {diagnostic}"))
                .child(
                    div()
                        .text_lg()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child("Workflow opened read-only"),
                )
                .child(
                    div()
                        .max_w_128()
                        .text_sm()
                        .text_color(rgb(0xc8a971))
                        .child(diagnostic),
                )
                .child(div().text_xs().text_color(rgb(0x9ca3af)).child(format!(
                    "Original data preserved ({} bytes)",
                    item.model.original_bytes().len()
                ))),
        );
    }

    root = root
        .when(item.qpov2_enabled, |root| {
            root.child(render_execution_controls(
                item.execution_run_mode(),
                item.execution_mode_menu_open(),
                queue_unavailable_reason.clone(),
                execute_focus_handle.clone(),
                execute_output_feedback_active,
                item.can_restore_execution_navigation(),
                active_execution.clone(),
                item.show_execution_progress,
                cx,
            ))
        })
        .when(item.qpov2_enabled && item.queue_overlay_visible, |root| {
            root.child(render_queue_overlay(
                execution_snapshot.as_ref(),
                item.queue_overlay_tab,
                item.queue_details_attempt,
                cx,
            ))
        })
        .when_some(
            active_execution.filter(|attempt| {
                attempt.failure.is_some() && !item.execution_error_is_dismissed(attempt.attempt_id)
            }),
            |root, attempt| root.child(render_execution_error_overlay(attempt, cx)),
        )
        .when_some(item.model.last_error.clone(), |root, error| {
            root.child(
                ui::diagnostic_alert("sim-graph-error", format!("Graph error: {error}"), error)
                    .debug_selector(|| "SIM-GRAPH-ERROR".into())
                    .absolute()
                    .top_3()
                    .left_3()
                    .right_3()
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0xd06b70))
                    .bg(rgb(0x4a2428))
                    .px_3()
                    .py_2()
                    .text_sm(),
            )
        })
        .when_some(item.model.announcement.clone(), |root, announcement| {
            root.child(
                div()
                    .id("sim-graph-announcement")
                    .debug_selector(|| "SIM-GRAPH-ANNOUNCEMENT".into())
                    .role(Role::Status)
                    .aria_label(announcement.clone())
                    .absolute()
                    .left_3()
                    .bottom_3()
                    .max_w_96()
                    .rounded_md()
                    .bg(rgb(0x252932).opacity(0.94))
                    .px_3()
                    .py_2()
                    .text_xs()
                    .child(announcement),
            )
        });
    item.finish_control_focus_handle_render();
    let menu_handle = item.context_menu_handle.clone();
    let weak_item = cx.weak_entity();
    ui::right_click_menu("sim-native-graph-context-menu")
        .full_width(true)
        .full_height(true)
        .with_handle(menu_handle)
        .maybe_menu(move |window, cx| {
            if let Err(error) = require_graph_context_infrastructure(
                GeneratedGraphContextInfrastructure::NativeContextRenderer,
            ) {
                log::error!("native graph context event renderer is unavailable: {error}");
                return None;
            }
            let mouse_position = window.mouse_position();
            let captured = weak_item
                .update(cx, |item, cx| {
                    item.capture_pointer_graph_context_menu(
                        GraphPoint {
                            x: mouse_position.x.into(),
                            y: mouse_position.y.into(),
                        },
                        window,
                        cx,
                    )
                })
                .ok()?;
            captured.then(|| build_graph_context_menu(weak_item.clone(), window, cx))?
        })
        .trigger(move |_, _, _| root)
        .into_any_element()
}

fn is_context_menu_keystroke(event: &KeyDownEvent) -> bool {
    event.keystroke.key == "contextmenu"
        || (event.keystroke.key == "f10" && event.keystroke.modifiers.shift)
}

fn open_keyboard_graph_context_menu(
    item: &mut GraphWorkspaceItem,
    window: &mut Window,
    cx: &mut Context<GraphWorkspaceItem>,
) {
    let viewport_size = window.viewport_size();
    let screen_position = GraphPoint {
        x: f32::from(viewport_size.width) / 2.0,
        y: f32::from(viewport_size.height) / 2.0,
    };
    let Some((target, screen_position)) = item
        .model
        .document()
        .and_then(|document| document.active_graph().ok())
        .map(|graph| {
            (
                GraphContextTarget::Canvas {
                    graph_position: graph.viewport.screen_to_graph(screen_position),
                },
                screen_position,
            )
        })
    else {
        item.model.report_error("active graph is unavailable");
        cx.notify();
        return;
    };
    item.open_graph_context_menu(target, screen_position, window, cx);
}

fn render_subgraph_boundary_slots(
    definition: &SubgraphDefinition,
    viewport_width: f32,
    item: &mut GraphWorkspaceItem,
    window: &Window,
    cx: &mut Context<GraphWorkspaceItem>,
) -> Vec<AnyElement> {
    let mut elements = Vec::with_capacity(definition.inputs.len() + definition.outputs.len());
    for (slot, port) in definition.inputs.iter().cloned().enumerate() {
        elements.push(render_subgraph_boundary_slot(
            port,
            GraphSlotDirection::Input,
            slot,
            viewport_width,
            item,
            window,
            cx,
        ));
    }
    for (slot, port) in definition.outputs.iter().cloned().enumerate() {
        elements.push(render_subgraph_boundary_slot(
            port,
            GraphSlotDirection::Output,
            slot,
            viewport_width,
            item,
            window,
            cx,
        ));
    }
    elements
}

fn render_subgraph_boundary_slot(
    port: SubgraphPort,
    direction: GraphSlotDirection,
    slot: usize,
    viewport_width: f32,
    item: &mut GraphWorkspaceItem,
    window: &Window,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    let direction_name = match direction {
        GraphSlotDirection::Input => "input",
        GraphSlotDirection::Output => "output",
    };
    let focus_handle = item.control_focus_handle(format!("subgraph-{direction_name}:{slot}"), cx);
    let focused = focus_handle.is_focused(window);
    let mouse_focus_handle = focus_handle.clone();
    let keyboard_position = GraphPoint {
        x: if direction == GraphSlotDirection::Input {
            16.0
        } else {
            viewport_width - 16.0
        },
        y: 104.0 + slot as f32 * 38.0,
    };
    let connected = port.internal_node.is_some();
    h_flex()
        .id(SharedString::from(format!(
            "sim-subgraph-{direction_name}-{slot}"
        )))
        .debug_selector(move || format!("SIM-SUBGRAPH-{}-{slot}", direction_name.to_uppercase()))
        .track_focus(&focus_handle)
        .tab_stop(true)
        .role(Role::Button)
        .aria_label(format!(
            "Subgraph {direction_name} slot {}, type {}{}",
            port.name,
            port.port_type.display_name(),
            if connected {
                ", connected"
            } else {
                ", disconnected"
            }
        ))
        .absolute()
        .top(px(88.0 + slot as f32 * 38.0))
        .when(direction == GraphSlotDirection::Input, |this| this.left_2())
        .when(direction == GraphSlotDirection::Output, |this| {
            this.right_2()
        })
        .max_w_64()
        .gap_2()
        .rounded_md()
        .border_1()
        .border_color(if focused {
            rgb(0xffffff)
        } else if connected {
            rgb(0x6d94d8)
        } else {
            rgb(0x596273)
        })
        .bg(rgb(0x252932).opacity(0.96))
        .px_2()
        .py_1()
        .text_xs()
        .cursor_pointer()
        .child(if direction == GraphSlotDirection::Input {
            format!("● {}", port.name)
        } else {
            format!("{} ●", port.name)
        })
        .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
            if is_context_menu_keystroke(event) {
                cx.stop_propagation();
                this.open_graph_context_menu(
                    GraphContextTarget::Slot { direction, slot },
                    keyboard_position,
                    window,
                    cx,
                );
            }
        }))
        .capture_any_mouse_down(
            cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                if event.button != MouseButton::Right {
                    return;
                }
                window.focus(&mouse_focus_handle, cx);
                this.stage_pointer_graph_context_target(GraphContextTarget::Slot {
                    direction,
                    slot,
                });
            }),
        )
        .into_any_element()
}

fn render_execution_controls(
    run_mode: crate::ExecutionRunMode,
    mode_menu_open: bool,
    queue_unavailable_reason: Option<String>,
    execute_focus_handle: FocusHandle,
    execute_output_feedback_active: bool,
    can_restore_navigation: bool,
    attempt: Option<sim_runtime::AttemptPresentation>,
    show_progress: bool,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    let queue_available = queue_unavailable_reason.is_none();
    let run_mode_label = execution_run_mode_label(run_mode);
    let execute_label = match run_mode {
        crate::ExecutionRunMode::Manual => "Execute".to_owned(),
        crate::ExecutionRunMode::OnChange => "Execute · On change".to_owned(),
        crate::ExecutionRunMode::InstantIdle => "Execute · Instant".to_owned(),
    };
    let execute_label_selector = match run_mode {
        crate::ExecutionRunMode::Manual => "SIM-EXECUTE-BUTTON-LABEL-MANUAL",
        crate::ExecutionRunMode::OnChange => "SIM-EXECUTE-BUTTON-LABEL-ON-CHANGE",
        crate::ExecutionRunMode::InstantIdle => "SIM-EXECUTE-BUTTON-LABEL-INSTANT",
    };
    let state_label = attempt
        .as_ref()
        .map(|attempt| format!("{:?}", attempt.state))
        .unwrap_or_else(|| "Idle".to_owned());
    let progress = attempt
        .as_ref()
        .and_then(|attempt| attempt.progress.as_ref())
        .cloned();
    v_flex()
        .id("sim-execution-actionbar")
        .debug_selector(|| "SIM-EXECUTION-ACTIONBAR".into())
        .role(Role::Toolbar)
        .aria_label("Native execution controls")
        .absolute()
        .top_3()
        .right_3()
        .gap_1()
        .rounded_md()
        .border_1()
        .border_color(rgb(0x596273))
        .bg(rgb(0x242832).opacity(0.96))
        .p_2()
        .child(
            h_flex()
                .gap_2()
                .child(
                    div()
                        .id("sim-execute-button")
                        .debug_selector(|| "SIM-EXECUTE-BUTTON".into())
                        .role(Role::Button)
                        .track_focus(&execute_focus_handle)
                        .tab_stop(true)
                        .tab_index(0)
                        .aria_label(queue_unavailable_reason.as_ref().map_or_else(
                            || "Queue selected native workflow output nodes".to_owned(),
                            |reason| {
                                format!(
                                    "Queue selected native workflow output nodes unavailable: {reason}"
                                )
                            },
                        ))
                        .rounded_sm()
                        .bg(rgb(0x365b8c))
                        .px_3()
                        .py_1()
                        .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                            this.set_execute_output_feedback_hovered(*hovered, cx);
                        }))
                        .when(queue_available, |this| {
                            this.cursor_pointer()
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.dispatch_shell_command(
                                        "Sim.QueueSelectedOutputNodes",
                                        cx,
                                    );
                                }))
                                .on_key_down(cx.listener(
                                    |this, event: &KeyDownEvent, _, cx| {
                                        if event.keystroke.key == "enter"
                                            || event.keystroke.key == "space"
                                        {
                                            cx.stop_propagation();
                                            this.dispatch_shell_command(
                                                "Sim.QueueSelectedOutputNodes",
                                                cx,
                                            );
                                        }
                                    },
                                ))
                        })
                        .when(!queue_available, |this| {
                            this.cursor_not_allowed().opacity(0.5)
                        })
                        .child(
                            div()
                                .debug_selector(move || execute_label_selector.into())
                                .child(execute_label),
                        ),
                )
                .child(
                    div()
                        .id("sim-execution-state-badge")
                        .role(Role::Status)
                        .aria_label(format!("Native execution state {state_label}"))
                        .rounded_sm()
                        .bg(rgb(0x303641))
                        .px_2()
                        .py_1()
                        .text_xs()
                        .child(state_label),
                ),
        )
        .when(execute_output_feedback_active, |this| {
            this.child(
                div()
                    .id("sim-execute-output-feedback")
                    .debug_selector(|| "SIM-EXECUTE-OUTPUT-FEEDBACK".into())
                    .role(Role::Status)
                    .aria_label("Selected native output target nodes are highlighted")
                    // Hover feedback must not resize the toolbar and move its own trigger.
                    .absolute()
                    .top_full()
                    .right_0()
                    .w_full()
                    .mt_1()
                    .text_xs()
                    .text_color(rgb(0x9be6b0))
                    .child("Selected output targets highlighted"),
            )
        })
        .when_some(queue_unavailable_reason, |this, reason| {
            this.child(
                div()
                    .id("sim-execute-unavailable-reason")
                    .debug_selector(|| "SIM-EXECUTE-UNAVAILABLE-REASON".into())
                    .role(Role::Status)
                    .aria_label(format!("Native execution unavailable: {reason}"))
                    .max_w_96()
                    .text_xs()
                    .text_color(rgb(0xffc1c5))
                    .child(reason),
            )
        })
        .child(
            div()
                .id("sim-execution-run-mode-trigger")
                .debug_selector(|| "SIM-EXECUTION-RUN-MODE-TRIGGER".into())
                .role(Role::Button)
                .tab_stop(true)
                .aria_label(format!("Native execution run mode {run_mode_label}"))
                .aria_expanded(mode_menu_open)
                .cursor_pointer()
                .rounded_sm()
                .bg(rgb(0x303641))
                .px_2()
                .py_1()
                .text_xs()
                .text_color(rgb(0xaeb6c2))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|_, _, _, cx| cx.stop_propagation()),
                )
                .on_click(cx.listener(|this, _, _, cx| {
                    this.toggle_execution_mode_menu(cx);
                }))
                .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        cx.stop_propagation();
                        this.toggle_execution_mode_menu(cx);
                    }
                }))
                .child(format!("Run mode: {run_mode_label}")),
        )
        .when(mode_menu_open, |this| {
            this.child(
                v_flex()
                    .id("sim-execution-run-mode-menu")
                    .debug_selector(|| "SIM-EXECUTION-RUN-MODE-MENU".into())
                    .role(Role::Menu)
                    .aria_label("Native execution run modes")
                    .gap_1()
                    .rounded_sm()
                    .border_1()
                    .border_color(rgb(0x596273))
                    .bg(rgb(0x20242c))
                    .p_1()
                    .child(render_execution_mode_option(
                        "Manual",
                        crate::ExecutionRunMode::Manual,
                        run_mode,
                        cx,
                    ))
                    .child(render_execution_mode_option(
                        "On change",
                        crate::ExecutionRunMode::OnChange,
                        run_mode,
                        cx,
                    ))
                    .child(render_execution_mode_option(
                        "Instant idle",
                        crate::ExecutionRunMode::InstantIdle,
                        run_mode,
                        cx,
                    )),
            )
        })
        .when_some(progress.filter(|_| show_progress), |this, progress| {
            let total = progress.total.max(1);
            let completed = progress.completed.min(total);
            this.child(
                div()
                    .id("sim-execution-progress")
                    .role(Role::ProgressIndicator)
                    .aria_label(progress.node_id.as_ref().map_or_else(
                        || "Native execution progress".to_owned(),
                        |node_id| format!("Native execution progress for node {}", node_id.0),
                    ))
                    .aria_value(format!("{completed} of {total}"))
                    .aria_numeric_value(completed as f64)
                    .aria_min_numeric_value(0.0)
                    .aria_max_numeric_value(total as f64)
                    .text_xs()
                    .child(format!("Progress {completed}/{total}")),
            )
        })
        .when(can_restore_navigation, |this| {
            this.child(
                div()
                    .id("sim-restore-execution-navigation")
                    .role(Role::Button)
                    .tab_stop(true)
                    .aria_label("Restore graph view from before execution navigation")
                    .cursor_pointer()
                    .text_xs()
                    .on_click(cx.listener(|this, _, _, cx| {
                        if let Err(error) = this.restore_execution_navigation(cx) {
                            this.model.report_error(error);
                            cx.notify();
                        }
                    }))
                    .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                        if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                            cx.stop_propagation();
                            if let Err(error) = this.restore_execution_navigation(cx) {
                                this.model.report_error(error);
                                cx.notify();
                            }
                        }
                    }))
                    .child("Restore graph view"),
            )
        })
        .into_any_element()
}

fn execution_run_mode_label(mode: crate::ExecutionRunMode) -> &'static str {
    match mode {
        crate::ExecutionRunMode::Manual => "Manual",
        crate::ExecutionRunMode::OnChange => "On change",
        crate::ExecutionRunMode::InstantIdle => "Instant idle",
    }
}

fn render_execution_mode_option(
    label: &'static str,
    mode: crate::ExecutionRunMode,
    selected_mode: crate::ExecutionRunMode,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    div()
        .id(SharedString::from(format!(
            "sim-execution-run-mode-option-{}",
            label.to_ascii_lowercase().replace(' ', "-")
        )))
        .debug_selector(move || {
            format!(
                "SIM-EXECUTION-RUN-MODE-{}",
                label.to_ascii_uppercase().replace(' ', "-")
            )
        })
        .role(Role::MenuItem)
        .tab_stop(true)
        .aria_label(format!(
            "{label} native execution mode{}",
            if mode == selected_mode {
                ", selected"
            } else {
                ""
            }
        ))
        .cursor_pointer()
        .rounded_sm()
        .px_2()
        .py_1()
        .when(mode == selected_mode, |this| this.bg(rgb(0x365b8c)))
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(|_, _, _, cx| cx.stop_propagation()),
        )
        .on_click(cx.listener(move |this, _, _, cx| {
            this.choose_execution_run_mode(mode, cx);
        }))
        .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                cx.stop_propagation();
                this.choose_execution_run_mode(mode, cx);
            }
        }))
        .child(label)
        .into_any_element()
}

fn render_queue_overlay(
    snapshot: Option<&sim_runtime::ExecutionSnapshot>,
    selected_tab: QueueOverlayTab,
    details_attempt: Option<sim_runtime::AttemptId>,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    let (queued, active) = snapshot.map_or((0, 0), |snapshot| {
        (
            snapshot.queue.len(),
            snapshot
                .attempts
                .iter()
                .filter(|attempt| !attempt.state.is_terminal())
                .count(),
        )
    });
    let has_failed = snapshot.is_some_and(|snapshot| {
        snapshot
            .attempts
            .iter()
            .any(|attempt| attempt.state == sim_runtime::AttemptState::Failed)
    });
    let attempts = snapshot
        .map(|snapshot| filtered_queue_overlay_attempts(snapshot, selected_tab))
        .unwrap_or_default();
    v_flex()
        .id("sim-queue-overlay-expanded")
        .debug_selector(|| "SIM-QUEUE-OVERLAY".into())
        .role(Role::Group)
        .aria_label(format!(
            "Native queue overlay, {queued} queued and {active} active"
        ))
        .absolute()
        .right_3()
        .top_32()
        .rounded_md()
        .border_1()
        .border_color(rgb(0x596273))
        .bg(rgb(0x242832).opacity(0.96))
        .w_96()
        .max_h_96()
        .overflow_y_scroll()
        .gap_2()
        .px_3()
        .py_2()
        .text_xs()
        .child(
            h_flex()
                .id("sim-queue-overlay-header")
                .debug_selector(|| "SIM-QUEUE-OVERLAY-HEADER".into())
                .justify_between()
                .child(format!("{queued} queued · {active} active"))
                .child(
                    div()
                        .id("sim-queue-overlay-docked-history")
                        .debug_selector(|| "SIM-QUEUE-OVERLAY-DOCKED-HISTORY".into())
                        .role(Role::Button)
                        .tab_stop(true)
                        .aria_label("Open docked native execution history")
                        .cursor_pointer()
                        .rounded_sm()
                        .px_2()
                        .py_1()
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.queue_overlay_visible = false;
                            cx.notify();
                            window.dispatch_action(
                                crate::ToggleDockedExecutionHistory.boxed_clone(),
                                cx,
                            );
                        }))
                        .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                cx.stop_propagation();
                                this.queue_overlay_visible = false;
                                cx.notify();
                                window.dispatch_action(
                                    crate::ToggleDockedExecutionHistory.boxed_clone(),
                                    cx,
                                );
                            }
                        }))
                        .child("History"),
                ),
        )
        .child(
            h_flex()
                .id("sim-queue-overlay-filter-tabs")
                .debug_selector(|| "SIM-QUEUE-OVERLAY-FILTER-TABS".into())
                .role(Role::TabList)
                .aria_label("Native queue overlay filters")
                .gap_1()
                .child(render_queue_overlay_tab(
                    "All",
                    QueueOverlayTab::All,
                    selected_tab,
                    cx,
                ))
                .child(render_queue_overlay_tab(
                    "Completed",
                    QueueOverlayTab::Completed,
                    selected_tab,
                    cx,
                ))
                .when(has_failed, |this| {
                    this.child(render_queue_overlay_tab(
                        "Failed",
                        QueueOverlayTab::Failed,
                        selected_tab,
                        cx,
                    ))
                }),
        )
        .children(
            attempts
                .into_iter()
                .map(|attempt| render_queue_overlay_attempt(attempt, details_attempt, cx)),
        )
        .when(snapshot.is_none(), |this| {
            this.child(
                div()
                    .id("sim-queue-overlay-unavailable")
                    .role(Role::Status)
                    .aria_label("Native queue state is unavailable")
                    .child("Queue unavailable"),
            )
        })
        .into_any_element()
}

pub(crate) fn filtered_queue_overlay_attempts(
    snapshot: &sim_runtime::ExecutionSnapshot,
    selected_tab: QueueOverlayTab,
) -> Vec<sim_runtime::AttemptPresentation> {
    snapshot
        .attempts
        .iter()
        .filter(|attempt| match selected_tab {
            QueueOverlayTab::All => true,
            QueueOverlayTab::Completed => attempt.state.is_terminal(),
            QueueOverlayTab::Failed => attempt.state == sim_runtime::AttemptState::Failed,
        })
        .cloned()
        .collect()
}

fn render_queue_overlay_tab(
    label: &'static str,
    tab: QueueOverlayTab,
    selected_tab: QueueOverlayTab,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    let selected = tab == selected_tab;
    div()
        .id(SharedString::from(format!(
            "sim-queue-overlay-tab-{}",
            label.to_ascii_lowercase()
        )))
        .debug_selector(move || format!("SIM-QUEUE-OVERLAY-TAB-{}", label.to_ascii_uppercase()))
        .role(Role::Tab)
        .tab_stop(true)
        .aria_label(format!("{label} native queue jobs"))
        .aria_selected(selected)
        .cursor_pointer()
        .rounded_sm()
        .px_2()
        .py_1()
        .when(selected, |this| this.bg(rgb(0x365b8c)))
        .on_click(cx.listener(move |this, _, _, cx| {
            this.select_queue_overlay_tab(tab, cx);
        }))
        .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                cx.stop_propagation();
                this.select_queue_overlay_tab(tab, cx);
            }
        }))
        .child(label)
        .into_any_element()
}

fn render_queue_overlay_attempt(
    attempt: sim_runtime::AttemptPresentation,
    details_attempt: Option<sim_runtime::AttemptId>,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    let attempt_id = attempt.attempt_id;
    let expanded = details_attempt == Some(attempt_id);
    let state = attempt.state;
    let failure = attempt.failure.clone();
    v_flex()
        .id(SharedString::from(format!(
            "sim-queue-overlay-attempt-{}",
            attempt_id.0
        )))
        .debug_selector(|| "SIM-QUEUE-OVERLAY-ATTEMPT".into())
        .role(Role::Group)
        .aria_label(format!(
            "Native execution job {} state {state:?}",
            attempt_id.0
        ))
        .rounded_sm()
        .border_1()
        .border_color(rgb(0x454d5a))
        .p_2()
        .child(
            h_flex()
                .justify_between()
                .child(
                    div()
                        .id(SharedString::from(format!(
                            "sim-queue-overlay-details-trigger-{}",
                            attempt_id.0
                        )))
                        .debug_selector(|| "SIM-QUEUE-OVERLAY-DETAILS-TRIGGER".into())
                        .role(Role::Button)
                        .tab_stop(true)
                        .aria_label(format!(
                            "{} details for native execution job {}",
                            if expanded { "Hide" } else { "Show" },
                            attempt_id.0
                        ))
                        .aria_expanded(expanded)
                        .cursor_pointer()
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.toggle_queue_attempt_details(attempt_id, cx);
                        }))
                        .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                cx.stop_propagation();
                                this.toggle_queue_attempt_details(attempt_id, cx);
                            }
                        }))
                        .child(format!("{state:?}")),
                )
                .child(
                    div()
                        .id(SharedString::from(format!(
                            "sim-queue-overlay-copy-job-id-{}",
                            attempt_id.0
                        )))
                        .role(Role::Button)
                        .tab_stop(true)
                        .aria_label(format!("Copy native execution job ID {}", attempt_id.0))
                        .cursor_pointer()
                        .on_click(cx.listener(move |this, _, _, cx| {
                            cx.stop_propagation();
                            this.copy_queue_attempt_id(attempt_id, cx);
                        }))
                        .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                cx.stop_propagation();
                                this.copy_queue_attempt_id(attempt_id, cx);
                            }
                        }))
                        .child("Copy ID"),
                ),
        )
        .when(expanded, |this| {
            this.child(
                v_flex()
                    .id(SharedString::from(format!(
                        "sim-job-details-popover-{}",
                        attempt_id.0
                    )))
                    .debug_selector(|| "SIM-QUEUE-OVERLAY-DETAILS-CONTENT".into())
                    .role(Role::Document)
                    .aria_label(format!("Details for native execution job {}", attempt_id.0))
                    .max_h_40()
                    .overflow_y_scroll()
                    .gap_1()
                    .rounded_sm()
                    .bg(rgb(0x1d2128))
                    .p_2()
                    .child(format!("Job ID: {}", attempt_id.0))
                    .child(format!("Prompt ID: {}", attempt.prompt_id.0))
                    .child(format!("State: {state:?}"))
                    .child(format!("Created: {}", attempt.created_at))
                    .when_some(attempt.finished_at, |details, finished_at| {
                        details.child(format!("Finished: {finished_at}"))
                    })
                    .when_some(failure, |details, failure| {
                        details.child(format!("{}: {}", failure.code, failure.message))
                    }),
            )
        })
        .into_any_element()
}

fn render_execution_error_overlay(
    attempt: sim_runtime::AttemptPresentation,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    let Some(failure) = attempt.failure else {
        return div().into_any_element();
    };
    let node_label = failure
        .node_id
        .as_ref()
        .map(|node_id| format!(" · node {}", node_id.0))
        .unwrap_or_default();
    v_flex()
        .id("sim-execution-error-overlay")
        .debug_selector(|| "SIM-EXECUTION-ERROR-OVERLAY".into())
        .role(Role::Alert)
        .aria_label(format!(
            "Native execution error {}: {}{}",
            failure.code, failure.message, node_label
        ))
        .absolute()
        .left_3()
        .bottom_16()
        .max_w_128()
        .gap_1()
        .rounded_md()
        .border_1()
        .border_color(rgb(0xd06b70))
        .bg(rgb(0x4a2428).opacity(0.98))
        .p_3()
        .child(
            div()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(format!("{}: {}", failure.code, failure.message)),
        )
        .child(
            div()
                .text_xs()
                .child(format!("Attempt {}{}", attempt.attempt_id.0, node_label)),
        )
        .child(
            h_flex()
                .gap_2()
                .child(
                    div()
                        .id("sim-copy-execution-error")
                        .role(Role::Button)
                        .tab_stop(true)
                        .aria_label("Copy structured native execution error")
                        .cursor_pointer()
                        .text_xs()
                        .on_click(cx.listener(|this, _, _, cx| this.copy_execution_error(cx)))
                        .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                cx.stop_propagation();
                                this.copy_execution_error(cx);
                            }
                        }))
                        .child("Copy"),
                )
                .child(
                    div()
                        .id("sim-locate-execution-error")
                        .role(Role::Button)
                        .tab_stop(true)
                        .aria_label("Locate native execution error node")
                        .cursor_pointer()
                        .text_xs()
                        .on_click(
                            cx.listener(|this, _, _, cx| this.locate_active_execution_error(cx)),
                        )
                        .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                cx.stop_propagation();
                                this.locate_active_execution_error(cx);
                            }
                        }))
                        .child("Locate"),
                )
                .child(
                    div()
                        .id("sim-dismiss-execution-error")
                        .role(Role::Button)
                        .tab_stop(true)
                        .aria_label("Dismiss native execution error overlay")
                        .cursor_pointer()
                        .text_xs()
                        .on_click(
                            cx.listener(|this, _, _, cx| this.dismiss_active_execution_error(cx)),
                        )
                        .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                cx.stop_propagation();
                                this.dismiss_active_execution_error(cx);
                            }
                        }))
                        .child("Dismiss"),
                ),
        )
        .into_any_element()
}

fn render_grid(graph: &GraphLevel) -> AnyElement {
    let offset = graph.viewport.offset;
    let scale = graph.viewport.scale;
    canvas(
        move |_, _, _| {},
        move |bounds, _, window, _| {
            let spacing = (32.0 * scale).max(8.0);
            let start_x = offset.x.rem_euclid(spacing);
            let start_y = offset.y.rem_euclid(spacing);
            let width: f32 = bounds.size.width.into();
            let height: f32 = bounds.size.height.into();
            let mut x = start_x;
            while x < width {
                let mut builder = PathBuilder::stroke(px(1.0));
                builder.move_to(point(bounds.origin.x + px(x), bounds.origin.y));
                builder.line_to(point(
                    bounds.origin.x + px(x),
                    bounds.origin.y + bounds.size.height,
                ));
                if let Ok(path) = builder.build() {
                    window.paint_path(path, rgb(0x242832));
                }
                x += spacing;
            }
            let mut y = start_y;
            while y < height {
                let mut builder = PathBuilder::stroke(px(1.0));
                builder.move_to(point(bounds.origin.x, bounds.origin.y + px(y)));
                builder.line_to(point(
                    bounds.origin.x + bounds.size.width,
                    bounds.origin.y + px(y),
                ));
                if let Ok(path) = builder.build() {
                    window.paint_path(path, rgb(0x242832));
                }
                y += spacing;
            }
        },
    )
    .absolute()
    .size_full()
    .into_any_element()
}

fn render_links(
    graph: &GraphLevel,
    hidden_nodes: &BTreeSet<GraphIdentifier>,
    drag_delta: GraphPoint,
) -> AnyElement {
    let links = graph
        .links
        .values()
        .filter_map(|link| {
            if hidden_nodes.contains(&link.origin_node) || hidden_nodes.contains(&link.target_node)
            {
                return None;
            }
            Some((
                link_route_points(graph, link, drag_delta)?,
                graph.selection.links.contains(&link.identifier),
            ))
        })
        .collect::<Vec<_>>();
    canvas(
        move |_, _, _| {},
        move |bounds, _, window, _| {
            for (points, selected) in &links {
                let mut builder = PathBuilder::stroke(px(if *selected { 3.0 } else { 2.0 }));
                let Some(first) = points.first() else {
                    continue;
                };
                builder.move_to(point(
                    bounds.origin.x + px(first.x),
                    bounds.origin.y + px(first.y),
                ));
                for point_value in points.iter().skip(1) {
                    builder.line_to(point(
                        bounds.origin.x + px(point_value.x),
                        bounds.origin.y + px(point_value.y),
                    ));
                }
                if let Ok(path) = builder.build() {
                    window.paint_path(
                        path,
                        if *selected {
                            rgb(0xf2c14e)
                        } else {
                            rgb(0x77a7ff)
                        },
                    );
                }
            }
        },
    )
    .absolute()
    .size_full()
    .into_any_element()
}

fn render_link_hit_targets(
    graph: &GraphLevel,
    hidden_nodes: &BTreeSet<GraphIdentifier>,
    drag_delta: GraphPoint,
    item: &mut GraphWorkspaceItem,
    cx: &mut Context<GraphWorkspaceItem>,
) -> Vec<AnyElement> {
    let mut targets = Vec::new();
    for link in graph.links.values() {
        if hidden_nodes.contains(&link.origin_node) || hidden_nodes.contains(&link.target_node) {
            continue;
        }
        let Some(points) = link_route_points(graph, link, drag_delta) else {
            continue;
        };
        let selected = graph.selection.links.contains(&link.identifier);
        let mut sample_index = 0usize;
        for segment in points.windows(2) {
            let delta_x = segment[1].x - segment[0].x;
            let delta_y = segment[1].y - segment[0].y;
            let distance = delta_x.hypot(delta_y);
            let sample_count = ((distance / 32.0).ceil() as usize).clamp(1, 32);
            for offset in 0..sample_count {
                let ratio = (offset as f32 + 0.5) / sample_count as f32;
                let position = GraphPoint {
                    x: segment[0].x + delta_x * ratio,
                    y: segment[0].y + delta_y * ratio,
                };
                let identifier = link.identifier.clone();
                let mouse_identifier = identifier.clone();
                let key_identifier = identifier.clone();
                let debug_identifier = identifier.text();
                let insert_position = position;
                let focus_handle = item
                    .control_focus_handle(format!("link:{}:{sample_index}", identifier.text()), cx);
                let mouse_focus_handle = focus_handle.clone();
                let target = div()
                    .id(SharedString::from(format!(
                        "sim-link-{}-{sample_index}",
                        identifier.text()
                    )))
                    .debug_selector(move || format!("SIM-LINK-{debug_identifier}-{sample_index}"))
                    .track_focus(&focus_handle)
                    .tab_stop(sample_index == 0)
                    .role(Role::Button)
                    .aria_label(format!(
                        "Link {} from {} output {} to {} input {}{}",
                        link.identifier.text(),
                        link.origin_node.text(),
                        link.origin_slot,
                        link.target_node.text(),
                        link.target_slot,
                        if selected { ", selected" } else { "" }
                    ))
                    .aria_selected(selected)
                    .absolute()
                    .left(px(position.x - 7.0))
                    .top(px(position.y - 7.0))
                    .size(px(14.0))
                    .rounded_full()
                    .border_1()
                    .border_color(if selected {
                        rgb(0xffffff)
                    } else {
                        rgb(0x77a7ff).opacity(0.18)
                    })
                    .bg(rgb(0x17191d).opacity(if selected { 0.72 } else { 0.06 }))
                    .cursor_pointer()
                    .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                        if event.keystroke.key == "i" {
                            cx.stop_propagation();
                            this.insert_reroute_on_link(
                                key_identifier.clone(),
                                insert_position,
                                cx,
                            );
                        } else if event.keystroke.key == "r" {
                            cx.stop_propagation();
                            this.start_link_reconnect(key_identifier.clone(), cx);
                        } else if event.keystroke.key == "enter" || event.keystroke.key == "space" {
                            cx.stop_propagation();
                            this.select_link(key_identifier.clone(), SelectionMode::Replace, cx);
                        }
                    }))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                            cx.stop_propagation();
                            window.focus(&mouse_focus_handle, cx);
                            if event.modifiers.alt {
                                this.insert_reroute_on_link(
                                    mouse_identifier.clone(),
                                    insert_position,
                                    cx,
                                );
                            } else if event.click_count >= 2 {
                                this.start_link_reconnect(mouse_identifier.clone(), cx);
                            } else {
                                this.select_link(
                                    mouse_identifier.clone(),
                                    if event.modifiers.shift {
                                        SelectionMode::Toggle
                                    } else {
                                        SelectionMode::Replace
                                    },
                                    cx,
                                );
                            }
                        }),
                    )
                    .into_any_element();
                targets.push(target);
                sample_index = sample_index.saturating_add(1);
            }
        }
    }
    targets
}

pub(crate) fn link_at_screen_point(
    routes: &[(GraphIdentifier, Vec<GraphPoint>)],
    point: GraphPoint,
) -> Option<GraphIdentifier> {
    const HIT_RADIUS: f32 = 8.0;
    routes
        .iter()
        .filter_map(|(identifier, points)| {
            let distance = points
                .windows(2)
                .map(|segment| point_segment_distance(point, segment[0], segment[1]))
                .fold(f32::INFINITY, f32::min);
            (distance <= HIT_RADIUS).then_some((identifier, distance))
        })
        .min_by(
            |(first_identifier, first_distance), (second_identifier, second_distance)| {
                first_distance
                    .total_cmp(second_distance)
                    .then_with(|| first_identifier.cmp(second_identifier))
            },
        )
        .map(|(identifier, _)| identifier.clone())
}

fn point_segment_distance(point: GraphPoint, start: GraphPoint, end: GraphPoint) -> f32 {
    let delta_x = end.x - start.x;
    let delta_y = end.y - start.y;
    let length_squared = delta_x * delta_x + delta_y * delta_y;
    if length_squared <= f32::EPSILON {
        return (point.x - start.x).hypot(point.y - start.y);
    }
    let projection = (((point.x - start.x) * delta_x + (point.y - start.y) * delta_y)
        / length_squared)
        .clamp(0.0, 1.0);
    let closest_x = start.x + projection * delta_x;
    let closest_y = start.y + projection * delta_y;
    (point.x - closest_x).hypot(point.y - closest_y)
}

fn link_route_points(
    graph: &GraphLevel,
    link: &GraphLink,
    drag_delta: GraphPoint,
) -> Option<Vec<GraphPoint>> {
    let origin = graph.nodes.get(&link.origin_node)?;
    let target = graph.nodes.get(&link.target_node)?;
    let mut from = graph.viewport.graph_to_screen(GraphPoint {
        x: origin.position.x + origin.size.width,
        y: origin.position.y + 42.0 + link.origin_slot as f32 * 22.0,
    });
    let mut to = graph.viewport.graph_to_screen(GraphPoint {
        x: target.position.x,
        y: target.position.y + 42.0 + link.target_slot as f32 * 22.0,
    });
    if node_moves_with_selection(graph, &link.origin_node) {
        from = from.translated(drag_delta);
    }
    if node_moves_with_selection(graph, &link.target_node) {
        to = to.translated(drag_delta);
    }
    let mut reroute_points = Vec::new();
    let mut current = link.parent_reroute.as_ref();
    let mut visited = BTreeSet::new();
    while let Some(identifier) = current {
        if !visited.insert(identifier.clone()) {
            return None;
        }
        let reroute = graph.reroutes.get(identifier)?;
        let mut position = graph.viewport.graph_to_screen(reroute.position);
        if graph.selection.reroutes.contains(identifier) {
            position = position.translated(drag_delta);
        }
        reroute_points.push(position);
        current = reroute.parent.as_ref();
    }
    reroute_points.reverse();
    let mut points = Vec::with_capacity(reroute_points.len() + 2);
    points.push(from);
    points.extend(reroute_points);
    points.push(to);
    Some(points)
}

fn node_moves_with_selection(graph: &GraphLevel, identifier: &GraphIdentifier) -> bool {
    graph.selection.nodes.contains(identifier)
        || graph.selection.groups.iter().any(|group_identifier| {
            graph
                .groups
                .get(group_identifier)
                .is_some_and(|group| group.node_ids.contains(identifier))
        })
}

fn collapsed_group_nodes(graph: &GraphLevel) -> BTreeSet<GraphIdentifier> {
    graph
        .groups
        .values()
        .filter(|group| group.collapsed)
        .flat_map(|group| group.node_ids.iter().cloned())
        .collect()
}

fn pending_link_points(item: &GraphWorkspaceItem, graph: &GraphLevel) -> Option<[GraphPoint; 2]> {
    let (origin_identifier, origin_slot) = item.pending_link.as_ref()?;
    let origin = graph.nodes.get(origin_identifier)?;
    let from = graph.viewport.graph_to_screen(GraphPoint {
        x: origin.position.x + origin.size.width,
        y: origin.position.y + 42.0 + *origin_slot as f32 * 22.0,
    });
    Some([from, item.pending_link_position?])
}

fn edge_pan_delta(position: GraphPoint, window: &Window) -> Option<GraphPoint> {
    const EDGE: f32 = 28.0;
    const STEP: f32 = 18.0;
    let viewport = window.viewport_size();
    let width: f32 = viewport.width.into();
    let height: f32 = viewport.height.into();
    let delta = GraphPoint {
        x: if position.x < EDGE {
            STEP
        } else if position.x > width - EDGE {
            -STEP
        } else {
            0.0
        },
        y: if position.y < EDGE {
            STEP
        } else if position.y > height - EDGE {
            -STEP
        } else {
            0.0
        },
    };
    (delta != GraphPoint::ZERO).then_some(delta)
}

fn render_pending_link(points: [GraphPoint; 2]) -> AnyElement {
    div()
        .id("sim-pending-link")
        .debug_selector(|| "SIM-PENDING-LINK".into())
        .absolute()
        .size_full()
        .child(canvas(
            move |_, _, _| {},
            move |bounds, _, window, _| {
                let mut builder = PathBuilder::stroke(px(2.0));
                builder.move_to(point(
                    bounds.origin.x + px(points[0].x),
                    bounds.origin.y + px(points[0].y),
                ));
                builder.line_to(point(
                    bounds.origin.x + px(points[1].x),
                    bounds.origin.y + px(points[1].y),
                ));
                if let Ok(path) = builder.build() {
                    window.paint_path(path, rgb(0xf2c14e));
                }
            },
        ))
        .into_any_element()
}

fn subgraph_breadcrumbs(document: &GraphDocument) -> Vec<(String, usize)> {
    if document.navigation.is_empty() {
        return Vec::new();
    }
    let mut breadcrumbs = vec![("Root".to_owned(), document.navigation.len())];
    let mut graph = &document.root;
    for (index, identifier) in document.navigation.iter().enumerate() {
        let Some(definition) = graph.definitions.get(identifier) else {
            break;
        };
        breadcrumbs.push((
            definition.name.clone(),
            document.navigation.len().saturating_sub(index + 1),
        ));
        graph = &definition.graph;
    }
    breadcrumbs
}

fn render_breadcrumbs(
    breadcrumbs: Vec<(String, usize)>,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    h_flex()
        .id("sim-subgraph-breadcrumbs")
        .debug_selector(|| "SIM-SUBGRAPH-BREADCRUMBS".into())
        .role(Role::Navigation)
        .aria_label("Subgraph navigation")
        .absolute()
        .top_3()
        .left_1_2()
        .gap_1()
        .rounded_md()
        .bg(rgb(0x252932).opacity(0.96))
        .px_2()
        .py_1()
        .children(
            breadcrumbs
                .into_iter()
                .enumerate()
                .map(|(index, (label, exit_count))| {
                    let key_exit_count = exit_count;
                    let mouse_exit_count = exit_count;
                    h_flex()
                        .id(SharedString::from(format!("sim-breadcrumb-{index}")))
                        .debug_selector(move || format!("SIM-BREADCRUMB-{index}"))
                        .focusable()
                        .tab_stop(exit_count > 0)
                        .role(if exit_count > 0 {
                            Role::Button
                        } else {
                            Role::Label
                        })
                        .aria_label(format!(
                            "Subgraph breadcrumb {label}{}",
                            if exit_count == 0 { ", current" } else { "" }
                        ))
                        .text_xs()
                        .cursor_pointer()
                        .when(index > 0, |element| element.child("›"))
                        .child(label)
                        .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                            if key_exit_count > 0
                                && (event.keystroke.key == "enter"
                                    || event.keystroke.key == "space")
                            {
                                cx.stop_propagation();
                                exit_subgraphs(this, key_exit_count, cx);
                            }
                        }))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(move |this, _, _, cx| {
                                cx.stop_propagation();
                                if mouse_exit_count > 0 {
                                    exit_subgraphs(this, mouse_exit_count, cx);
                                }
                            }),
                        )
                }),
        )
        .into_any_element()
}

fn exit_subgraphs(
    item: &mut GraphWorkspaceItem,
    count: usize,
    cx: &mut Context<GraphWorkspaceItem>,
) {
    item.apply_graph_command(
        GraphCommand::Batch {
            commands: (0..count).map(|_| GraphCommand::ExitSubgraph).collect(),
        },
        cx,
    );
}

fn selection_box(item: &GraphWorkspaceItem) -> Option<GraphRect> {
    let anchor = item.box_anchor?;
    let current = item.box_current?;
    Some(GraphRect {
        origin: GraphPoint {
            x: anchor.x.min(current.x),
            y: anchor.y.min(current.y),
        },
        size: sim_runtime::GraphSize {
            width: (anchor.x - current.x).abs().max(1.0),
            height: (anchor.y - current.y).abs().max(1.0),
        },
    })
}

fn render_node(
    node: GraphNode,
    graph: &GraphLevel,
    drag_delta: GraphPoint,
    detailed_native_renderer: bool,
    execution: Option<&sim_runtime::AttemptPresentation>,
    execute_output_feedback_active: bool,
    item: &mut GraphWorkspaceItem,
    window: &Window,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    let viewport = graph.viewport.clone();
    let node_scale = viewport.scale;
    let mut screen = viewport.graph_to_screen(node.position);
    let selected = graph.selection.nodes.contains(&node.identifier);
    let execute_target_highlighted = execute_output_feedback_active && selected;
    if node_moves_with_selection(graph, &node.identifier) {
        screen = screen.translated(drag_delta);
    }
    let identifier = node.identifier.clone();
    let title = node.title.clone();
    let rename_buffer = item
        .node_rename
        .as_ref()
        .filter(|(rename_identifier, _)| rename_identifier == &identifier)
        .map(|(_, buffer)| buffer.clone());
    let keyboard_identifier = identifier.clone();
    let keyboard_context_identifier = identifier.clone();
    let keyboard_title = title.clone();
    let keyboard_color = node.color.clone();
    let keyboard_subgraph = node.subgraph_definition.clone();
    let mouse_subgraph = node.subgraph_definition.clone();
    let mouse_identifier = identifier.clone();
    let mouse_context_identifier = identifier.clone();
    let mouse_title = title.clone();
    let focus_handle = item.control_focus_handle(format!("node:{}", identifier.text()), cx);
    let focused = focus_handle.is_focused(window);
    let mouse_focus_handle = focus_handle.clone();
    let context_focus_handle = focus_handle.clone();
    let node_execution = execution.and_then(|attempt| {
        let node_id = node.identifier.text();
        let progress = attempt
            .node_progress
            .iter()
            .find(|(progress_node_id, _)| progress_node_id.0 == node_id)
            .map(|(_, progress)| progress)
            .map(|progress| (progress.completed, progress.total));
        let failed = attempt
            .failure
            .as_ref()
            .is_some_and(|failure| node_has_execution_failure(&node, failure));
        let output_count = attempt
            .outputs
            .iter()
            .filter(|output| output.node_id.0 == node_id)
            .count();
        (progress.is_some() || failed || output_count > 0).then_some((
            attempt.attempt_id,
            attempt.state,
            progress,
            failed,
            output_count,
        ))
    });
    let mut aria_label = node_accessibility_label(&node, graph);
    if let Some((attempt_id, state, progress, failed, output_count)) = node_execution {
        aria_label.push_str(&format!(
            ", execution attempt {}, state {state:?}, {output_count} outputs{}{}",
            attempt_id.0,
            progress.map_or_else(String::new, |(completed, total)| {
                format!(", progress {completed} of {total}")
            }),
            if failed { ", failed" } else { "" }
        ));
    }
    if execute_target_highlighted {
        aria_label.push_str(", selected output target highlighted for execution");
    }
    let state = node.mode;
    let node_corner_radius = match node
        .source_fields
        .get("shape")
        .and_then(Value::as_str)
        .unwrap_or("default")
    {
        "box" => 0.0,
        "round" => 10.0,
        "card" => 14.0,
        _ => 6.0,
    };
    let show_advanced_widgets = node
        .source_fields
        .get("show_advanced")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let input_elements = node
        .inputs
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, port)| {
            let target_node = node.identifier.clone();
            let keyboard_target_node = target_node.clone();
            let focus_handle =
                item.control_focus_handle(format!("input:{}:{index}", target_node.text()), cx);
            let mouse_focus_handle = focus_handle.clone();
            h_flex()
                .id(SharedString::from(format!(
                    "sim-input-{}-{index}",
                    target_node.text()
                )))
                .debug_selector({
                    let identifier = target_node.text();
                    move || format!("SIM-INPUT-{identifier}-{index}")
                })
                .track_focus(&focus_handle)
                .tab_stop(true)
                .role(Role::Button)
                .aria_label(format!(
                    "Input {} of type {}",
                    port.name,
                    port.port_type.display_name()
                ))
                .absolute()
                .left(px(-9.0))
                .top(px((42.0 + index as f32 * 22.0) * node_scale - 9.0))
                .size(px(14.0))
                .items_center()
                .justify_center()
                .text_size(px((12.0 * node_scale).max(4.0)))
                .rounded_full()
                .bg(rgb(0x20242c))
                .child("●")
                .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _window, cx| {
                    if event.keystroke.key == "enter" || event.keystroke.key == "space" {
                        cx.stop_propagation();
                        if this.pending_link.is_some() {
                            this.complete_pending_link(keyboard_target_node.clone(), index, cx);
                        }
                    }
                }))
                .on_mouse_up(
                    MouseButton::Left,
                    cx.listener(move |this, _, window, cx| {
                        if this.pending_link.is_none() {
                            return;
                        }
                        cx.stop_propagation();
                        window.focus(&mouse_focus_handle, cx);
                        this.complete_pending_link(target_node.clone(), index, cx);
                    }),
                )
        })
        .collect::<Vec<_>>();
    let output_elements = node
        .outputs
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, port)| {
            let origin_node = node.identifier.clone();
            let keyboard_origin_node = origin_node.clone();
            let focus_handle =
                item.control_focus_handle(format!("output:{}:{index}", origin_node.text()), cx);
            let mouse_focus_handle = focus_handle.clone();
            h_flex()
                .id(SharedString::from(format!(
                    "sim-output-{}-{index}",
                    origin_node.text()
                )))
                .debug_selector({
                    let identifier = origin_node.text();
                    move || format!("SIM-OUTPUT-{identifier}-{index}")
                })
                .track_focus(&focus_handle)
                .tab_stop(true)
                .role(Role::Button)
                .aria_label(format!(
                    "Output {} of type {}",
                    port.name,
                    port.port_type.display_name()
                ))
                .absolute()
                .right(px(-9.0))
                .top(px((42.0 + index as f32 * 22.0) * node_scale - 9.0))
                .size(px(14.0))
                .items_center()
                .justify_center()
                .text_size(px((12.0 * node_scale).max(4.0)))
                .rounded_full()
                .bg(rgb(0x20242c))
                .child("●")
                .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _window, cx| {
                    if event.keystroke.key == "enter" || event.keystroke.key == "space" {
                        cx.stop_propagation();
                        this.start_link(keyboard_origin_node.clone(), index);
                        cx.notify();
                    }
                }))
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |this, _, window, cx| {
                        cx.stop_propagation();
                        window.focus(&mouse_focus_handle, cx);
                        this.start_link(origin_node.clone(), index);
                        cx.notify();
                    }),
                )
        })
        .collect::<Vec<_>>();
    let body = if node.collapsed {
        div().into_any_element()
    } else {
        v_flex()
            .gap(px(4.0 * node_scale))
            .px(px(8.0 * node_scale))
            .py(px(8.0 * node_scale))
            .text_size(px((12.0 * node_scale).max(4.0)))
            .child(
                h_flex()
                    .items_start()
                    .justify_between()
                    .child(
                        v_flex()
                            .gap(px(4.0 * node_scale))
                            .children(node.inputs.iter().map(|port| port.name.clone())),
                    )
                    .child(
                        v_flex()
                            .gap(px(4.0 * node_scale))
                            .children(node.outputs.iter().map(|port| port.name.clone())),
                    ),
            )
            .children(
                node.widgets
                    .iter()
                    .filter(|widget| {
                        widget.visible
                            && (show_advanced_widgets
                                || !widget
                                    .unknown
                                    .get("advanced")
                                    .and_then(Value::as_bool)
                                    .unwrap_or(false))
                    })
                    .cloned()
                    .map(|widget| {
                        render_widget(node.identifier.clone(), widget, node_scale, item, cx)
                    }),
            )
            .into_any_element()
    };
    div()
        .id(SharedString::from(format!(
            "sim-node-{}",
            identifier.text()
        )))
        .debug_selector({
            let identifier = identifier.text();
            move || format!("SIM-NODE-{identifier}")
        })
        .track_focus(&focus_handle)
        .tab_stop(true)
        .role(Role::Group)
        .aria_label(aria_label)
        .aria_selected(selected)
        .absolute()
        .left(px(screen.x))
        .top(px(screen.y))
        .w(px(node.size.width * viewport.scale))
        .h(px(if node.collapsed {
            34.0 * viewport.scale
        } else {
            node.size.height * viewport.scale
        }))
        .rounded(px(node_corner_radius))
        .border_2()
        .border_color(if focused {
            rgb(0xffffff)
        } else if execute_target_highlighted {
            rgb(0x68d391)
        } else if node_execution.is_some_and(|(_, _, _, failed, _)| failed) {
            rgb(0xd06b70)
        } else if node_execution.is_some() {
            rgb(0x72a7ff)
        } else if selected {
            rgb(0x72a7ff)
        } else {
            rgb(0x3f4653)
        })
        .bg(node_background_color(
            &node,
            state,
            detailed_native_renderer,
        ))
        .shadow_md()
        .cursor_pointer()
        .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
            if is_context_menu_keystroke(event) {
                cx.stop_propagation();
                this.open_graph_context_menu(
                    GraphContextTarget::Node(keyboard_context_identifier.clone()),
                    GraphPoint {
                        x: screen.x + 16.0,
                        y: screen.y + 16.0,
                    },
                    window,
                    cx,
                );
                return;
            }
            let editing = this
                .node_rename
                .as_ref()
                .is_some_and(|(identifier, _)| identifier == &keyboard_identifier);
            if editing {
                cx.stop_propagation();
                match event.keystroke.key.as_str() {
                    "escape" => {
                        this.node_rename = None;
                        cx.notify();
                    }
                    "enter" => {
                        let title = this
                            .node_rename
                            .take()
                            .map(|(_, title)| title)
                            .unwrap_or_default();
                        if title.trim().is_empty() {
                            this.model.report_error("node title cannot be empty");
                            cx.notify();
                        } else {
                            this.execute_model_action(
                                GraphModelAction::RenameNode,
                                GraphActionInput::RenameNode {
                                    identifier: keyboard_identifier.clone(),
                                    title,
                                },
                                cx,
                            );
                        }
                    }
                    "backspace" => {
                        if let Some((_, buffer)) = this.node_rename.as_mut() {
                            buffer.pop();
                        }
                        cx.notify();
                    }
                    _ if !event.keystroke.modifiers.control
                        && !event.keystroke.modifiers.alt
                        && !event.keystroke.modifiers.platform =>
                    {
                        if let Some(text) = event.keystroke.key_char.as_deref()
                            && !text.chars().any(char::is_control)
                            && let Some((_, buffer)) = this.node_rename.as_mut()
                        {
                            buffer.push_str(text);
                            cx.notify();
                        }
                    }
                    _ => {}
                }
                return;
            }
            match event.keystroke.key.as_str() {
                "f2" => {
                    cx.stop_propagation();
                    this.node_rename = Some((keyboard_identifier.clone(), keyboard_title.clone()));
                    cx.notify();
                }
                "d" => {
                    cx.stop_propagation();
                    this.execute_model_action(
                        GraphModelAction::ToggleNodeDisable,
                        GraphActionInput::NodeIdentifiers(BTreeSet::from([
                            keyboard_identifier.clone()
                        ])),
                        cx,
                    );
                }
                "c" => {
                    cx.stop_propagation();
                    let color = keyboard_color.is_none().then(|| "#355287".to_owned());
                    this.execute_model_action(
                        GraphModelAction::SetNodeColor,
                        GraphActionInput::NodeColor {
                            identifier: keyboard_identifier.clone(),
                            color,
                        },
                        cx,
                    );
                }
                "o" if keyboard_subgraph.is_some() => {
                    cx.stop_propagation();
                    if let Some(definition_identifier) = keyboard_subgraph.clone() {
                        this.apply_graph_command(
                            GraphCommand::OpenSubgraph {
                                definition_identifier,
                            },
                            cx,
                        );
                    }
                }
                "enter" | "space" => {
                    cx.stop_propagation();
                    this.select_node(
                        keyboard_identifier.clone(),
                        if event.keystroke.modifiers.shift {
                            SelectionMode::Toggle
                        } else {
                            SelectionMode::Replace
                        },
                        cx,
                    );
                }
                _ => {}
            }
        }))
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                cx.stop_propagation();
                window.focus(&mouse_focus_handle, cx);
                if event.click_count >= 2 {
                    if let Some(definition_identifier) = mouse_subgraph.clone() {
                        this.apply_graph_command(
                            GraphCommand::OpenSubgraph {
                                definition_identifier,
                            },
                            cx,
                        );
                    } else {
                        this.node_rename = Some((mouse_identifier.clone(), mouse_title.clone()));
                        cx.notify();
                    }
                    return;
                }
                let already_selected = this
                    .model
                    .selection()
                    .is_some_and(|selection| selection.nodes.contains(&mouse_identifier));
                if event.modifiers.shift || !already_selected {
                    this.select_node(
                        mouse_identifier.clone(),
                        if event.modifiers.shift {
                            SelectionMode::Toggle
                        } else {
                            SelectionMode::Replace
                        },
                        cx,
                    );
                }
                this.begin_selection_drag(GraphPoint {
                    x: event.position.x.into(),
                    y: event.position.y.into(),
                });
            }),
        )
        .capture_any_mouse_down(
            cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                if event.button != MouseButton::Right {
                    return;
                }
                window.focus(&context_focus_handle, cx);
                this.stage_pointer_graph_context_target(GraphContextTarget::Node(
                    mouse_context_identifier.clone(),
                ));
            }),
        )
        .child(
            h_flex()
                .justify_between()
                .rounded_t_md()
                .bg(if detailed_native_renderer {
                    rgb(0x343a46)
                } else {
                    rgb(0x292e37)
                })
                .px(px(8.0 * node_scale))
                .py(px(4.0 * node_scale))
                .text_size(px((14.0 * node_scale).max(4.0)))
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(rename_buffer.map_or(title, |buffer| format!("✎ {buffer}")))
                .child(if node.pinned { "📌" } else { "" })
                .when(execute_target_highlighted, |this| {
                    this.child(
                        div()
                            .id(format!(
                                "sim-execute-output-target-{}",
                                node.identifier.text()
                            ))
                            .debug_selector(|| "SIM-EXECUTE-OUTPUT-TARGET".into())
                            .role(Role::Status)
                            .aria_label(format!(
                                "Node {} is a selected execution output target",
                                node.identifier.text()
                            ))
                            .text_xs()
                            .text_color(rgb(0x9be6b0))
                            .child("Output target"),
                    )
                })
                .when_some(
                    node_execution,
                    |this, (attempt_id, state, progress, failed, output_count)| {
                        let status = progress.map_or_else(
                            || format!("{state:?} · {output_count} outputs"),
                            |(completed, total)| format!("{state:?} · {completed}/{total}"),
                        );
                        this.child(
                            div()
                                .id(format!("sim-node-execution-{}", node.identifier.text()))
                                .role(if failed { Role::Alert } else { Role::Status })
                                .aria_label(format!(
                                    "Node {} execution attempt {}: {status}{}",
                                    node.identifier.text(),
                                    attempt_id.0,
                                    if failed { ", failed" } else { "" }
                                ))
                                .text_xs()
                                .text_color(if failed { rgb(0xffa2a8) } else { rgb(0xa9c9ff) })
                                .child(status),
                        )
                    },
                ),
        )
        .children(input_elements)
        .children(output_elements)
        .child(body)
        .into_any_element()
}

pub(crate) fn node_has_execution_failure(
    node: &GraphNode,
    failure: &sim_runtime::ExecutionFailure,
) -> bool {
    let Some(failed_node_id) = failure.node_id.as_ref() else {
        return false;
    };
    let node_id = node.identifier.text();
    if failed_node_id.0 == node_id {
        return true;
    }
    node.subgraph_definition.is_some()
        && failed_node_id
            .0
            .strip_prefix(&node_id)
            .is_some_and(|suffix| suffix.starts_with("::"))
}

fn node_background_color(
    node: &GraphNode,
    state: GraphNodeMode,
    detailed_native_renderer: bool,
) -> Rgba {
    if state == GraphNodeMode::Always
        && let Some(color) = node
            .source_fields
            .get("bgcolor")
            .and_then(Value::as_str)
            .or(node.color.as_deref())
            .and_then(parse_hex_color)
    {
        return color;
    }
    match state {
        GraphNodeMode::Always if detailed_native_renderer => rgb(0x2a2f38),
        GraphNodeMode::Always => rgb(0x22262d),
        GraphNodeMode::OnEvent => rgb(0x343246),
        GraphNodeMode::Never => rgb(0x413c2b),
        GraphNodeMode::OnTrigger => rgb(0x314137),
        GraphNodeMode::Bypass => rgb(0x293e46),
    }
}

fn parse_hex_color(value: &str) -> Option<Rgba> {
    let value = value.strip_prefix('#').unwrap_or(value);
    match value.len() {
        3 => {
            let mut expanded = String::with_capacity(6);
            for component in value.chars() {
                expanded.push(component);
                expanded.push(component);
            }
            u32::from_str_radix(&expanded, 16).ok().map(rgb)
        }
        6 => u32::from_str_radix(value, 16).ok().map(rgb),
        _ => None,
    }
}

fn render_widget(
    node_identifier: GraphIdentifier,
    widget: GraphWidget,
    scale: f32,
    item: &mut GraphWorkspaceItem,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    let display_value = widget_display_value(&widget.value);
    let validation = match &widget.validation {
        WidgetValidation::Valid => String::new(),
        WidgetValidation::Invalid(reason) => format!(", invalid: {reason}"),
    };
    let role = match widget.kind {
        GraphWidgetKind::Boolean => Role::Switch,
        GraphWidgetKind::Integer { .. } | GraphWidgetKind::Float { .. } => Role::SpinButton,
        GraphWidgetKind::Text { .. } | GraphWidgetKind::Combo { .. } => Role::TextInput,
        GraphWidgetKind::Preserved { .. } => Role::Group,
    };
    let operable = !matches!(widget.kind, GraphWidgetKind::Preserved { .. });
    let editable = matches!(
        widget.kind,
        GraphWidgetKind::Text { .. } | GraphWidgetKind::Combo { dynamic: true, .. }
    );
    let mouse_node_identifier = node_identifier.clone();
    let mouse_widget_identifier = widget.identifier.clone();
    let mouse_widget = widget.clone();
    let key_widget_identifier = widget.identifier.clone();
    let key_widget = widget.clone();
    let focus_handle = item.control_focus_handle(
        format!("widget:{}:{}", node_identifier.text(), widget.identifier),
        cx,
    );
    let mouse_focus_handle = focus_handle.clone();
    let element = div()
        .id(SharedString::from(format!(
            "sim-widget-{}-{}",
            node_identifier.text(),
            widget.identifier
        )))
        .debug_selector({
            let node_identifier = node_identifier.text();
            let widget_identifier = widget.identifier.clone();
            move || format!("SIM-WIDGET-{node_identifier}-{widget_identifier}")
        })
        .track_focus(&focus_handle)
        .when(editable, |element| {
            element.key_context(crate::SIM_TEXT_INPUT_KEY_CONTEXT)
        })
        .tab_stop(operable)
        .role(role)
        .aria_label(format!(
            "Widget {} value {}{}{}{}",
            widget.identifier,
            display_value,
            if widget.converted_to_input {
                ", promoted to input"
            } else {
                ""
            },
            validation,
            if operable {
                ""
            } else {
                ", unsupported native widget preserved as a read-only placeholder"
            }
        ))
        .aria_value(display_value.clone())
        .when_some(boolean_widget_state(&widget), |element, toggled| {
            element.aria_toggled(toggled)
        })
        .when_some(numeric_widget_state(&widget), |element, state| {
            element
                .aria_numeric_value(state.value)
                .aria_min_numeric_value(state.minimum)
                .aria_max_numeric_value(state.maximum)
                .aria_numeric_value_step(state.step)
        })
        .rounded_sm()
        .border_1()
        .border_color(if matches!(widget.validation, WidgetValidation::Valid) {
            rgb(0x343a46)
        } else {
            rgb(0xd06b70)
        })
        .bg(rgb(0x20242c))
        .px(px(8.0 * scale))
        .py(px(4.0 * scale))
        .text_size(px((12.0 * scale).max(4.0)))
        .child(format!("{}: {display_value}", widget.identifier))
        .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
            let Some(value) = widget_value_for_key(&key_widget, event) else {
                return;
            };
            cx.stop_propagation();
            this.apply_graph_command(
                GraphCommand::SetWidget {
                    node: node_identifier.clone(),
                    widget: key_widget_identifier.clone(),
                    value,
                },
                cx,
            );
        }))
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, _, window, cx| {
                cx.stop_propagation();
                window.focus(&mouse_focus_handle, cx);
                let Some(value) = widget_activation_value(&mouse_widget) else {
                    return;
                };
                this.apply_graph_command(
                    GraphCommand::SetWidget {
                        node: mouse_node_identifier.clone(),
                        widget: mouse_widget_identifier.clone(),
                        value,
                    },
                    cx,
                );
            }),
        );
    element.into_any_element()
}

#[derive(Clone, Copy)]
struct NumericWidgetState {
    value: f64,
    minimum: f64,
    maximum: f64,
    step: f64,
}

fn boolean_widget_state(widget: &GraphWidget) -> Option<Toggled> {
    matches!(widget.kind, GraphWidgetKind::Boolean).then(|| {
        if widget.value.as_bool().unwrap_or(false) {
            Toggled::True
        } else {
            Toggled::False
        }
    })
}

fn numeric_widget_state(widget: &GraphWidget) -> Option<NumericWidgetState> {
    match widget.kind {
        GraphWidgetKind::Integer {
            minimum,
            maximum,
            step,
        } => Some(NumericWidgetState {
            value: widget.value.as_i64().unwrap_or(minimum) as f64,
            minimum: minimum as f64,
            maximum: maximum as f64,
            step: step as f64,
        }),
        GraphWidgetKind::Float {
            minimum,
            maximum,
            step,
        } => Some(NumericWidgetState {
            value: widget.value.as_f64().unwrap_or(minimum),
            minimum,
            maximum,
            step,
        }),
        _ => None,
    }
}

fn widget_activation_value(widget: &GraphWidget) -> Option<Value> {
    match &widget.kind {
        GraphWidgetKind::Boolean => Some(Value::Bool(!widget.value.as_bool().unwrap_or(false))),
        GraphWidgetKind::Integer { step, .. } => {
            Some(Value::from(widget.value.as_i64()?.saturating_add(*step)))
        }
        GraphWidgetKind::Float { step, .. } => Some(Value::from(widget.value.as_f64()? + *step)),
        GraphWidgetKind::Combo { values, dynamic } => {
            if *dynamic && values.is_empty() {
                return None;
            }
            let current = widget.value.as_str().unwrap_or_default();
            let index = values
                .iter()
                .position(|value| value == current)
                .unwrap_or(values.len().saturating_sub(1));
            values
                .get((index + 1) % values.len().max(1))
                .cloned()
                .map(Value::String)
        }
        GraphWidgetKind::Text { .. } | GraphWidgetKind::Preserved { .. } => None,
    }
}

fn widget_value_for_key(widget: &GraphWidget, event: &KeyDownEvent) -> Option<Value> {
    let direction = match event.keystroke.key.as_str() {
        "down" | "left" => -1.0,
        "up" | "right" | "enter" | "space" => 1.0,
        _ => 0.0,
    };
    match &widget.kind {
        GraphWidgetKind::Boolean if direction != 0.0 => {
            Some(Value::Bool(!widget.value.as_bool().unwrap_or(false)))
        }
        GraphWidgetKind::Integer { step, .. } if direction != 0.0 => Some(Value::from(
            widget.value.as_i64()?.saturating_add(if direction < 0.0 {
                step.saturating_neg()
            } else {
                *step
            }),
        )),
        GraphWidgetKind::Float { step, .. } if direction != 0.0 => {
            Some(Value::from(widget.value.as_f64()? + direction * *step))
        }
        GraphWidgetKind::Combo { values, dynamic } if direction != 0.0 && !values.is_empty() => {
            let current = widget.value.as_str().unwrap_or_default();
            let index = values
                .iter()
                .position(|value| value == current)
                .unwrap_or(0);
            let next = if direction < 0.0 {
                index.checked_sub(1).unwrap_or(values.len() - 1)
            } else {
                (index + 1) % values.len()
            };
            values.get(next).cloned().map(Value::String)
        }
        GraphWidgetKind::Text { multiline } => {
            edit_text_widget(widget.value.as_str()?, event, *multiline)
        }
        GraphWidgetKind::Combo {
            dynamic: true,
            values: _,
        } => edit_text_widget(widget.value.as_str()?, event, false),
        GraphWidgetKind::Preserved { .. }
        | GraphWidgetKind::Combo { .. }
        | GraphWidgetKind::Boolean
        | GraphWidgetKind::Integer { .. }
        | GraphWidgetKind::Float { .. } => None,
    }
}

fn edit_text_widget(current: &str, event: &KeyDownEvent, multiline: bool) -> Option<Value> {
    if event.keystroke.modifiers.control
        || event.keystroke.modifiers.alt
        || event.keystroke.modifiers.platform
    {
        return None;
    }
    let mut value = current.to_owned();
    if event.keystroke.key == "backspace" {
        value.pop()?;
    } else if event.keystroke.key == "enter" && multiline {
        value.push('\n');
    } else {
        let text = event.keystroke.key_char.as_deref()?;
        if text.chars().any(char::is_control) {
            return None;
        }
        value.push_str(text);
    }
    Some(Value::String(value))
}

fn widget_display_value(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

pub(crate) fn node_accessibility_label(node: &GraphNode, graph: &GraphLevel) -> String {
    let state = match node.mode {
        GraphNodeMode::Always => "always",
        GraphNodeMode::OnEvent => "on event",
        GraphNodeMode::Never => "never",
        GraphNodeMode::OnTrigger => "on trigger",
        GraphNodeMode::Bypass => "bypass",
    };
    format!(
        "Node {}, type {}, {}, {} inputs, {} outputs{}{}{}",
        node.title,
        node.type_identifier,
        state,
        node.inputs.len(),
        node.outputs.len(),
        if node.pinned { ", pinned" } else { "" },
        if node.collapsed { ", collapsed" } else { "" },
        if graph.selection.nodes.contains(&node.identifier) {
            ", selected"
        } else {
            ""
        }
    )
}

fn render_minimap(
    graph: &GraphLevel,
    hidden_nodes: &BTreeSet<GraphIdentifier>,
    available_viewport: GraphSize,
    cx: &mut Context<GraphWorkspaceItem>,
) -> AnyElement {
    const WIDTH: f32 = 128.0;
    const HEIGHT: f32 = 80.0;
    const INSET: f32 = 6.0;
    let content_bounds = graph_content_bounds(graph, hidden_nodes).unwrap_or(GraphRect {
        origin: GraphPoint::ZERO,
        size: GraphSize {
            width: 1.0,
            height: 1.0,
        },
    });
    let minimap_transform = MinimapTransform::new(content_bounds, WIDTH, HEIGHT, INSET);
    let node_rects = graph
        .nodes
        .values()
        .filter(|node| !hidden_nodes.contains(&node.identifier))
        .map(|node| minimap_transform.project_rect(node.bounds()))
        .collect::<Vec<_>>();
    let graph_viewport = GraphRect {
        origin: graph.viewport.screen_to_graph(GraphPoint::ZERO),
        size: GraphSize {
            width: available_viewport.width / graph.viewport.scale,
            height: available_viewport.height / graph.viewport.scale,
        },
    };
    let viewport_rect = minimap_transform.project_rect(graph_viewport);
    let keyboard_bounds = content_bounds;
    let mouse_transform = minimap_transform;
    div()
        .id("sim-minimap")
        .debug_selector(|| "SIM-MINIMAP".into())
        .focusable()
        .tab_stop(true)
        .role(Role::Button)
        .aria_label(format!(
            "Graph minimap, {} visible nodes; activate to fit, click to center viewport",
            node_rects.len()
        ))
        .absolute()
        .right_3()
        .bottom_3()
        .w(px(WIDTH))
        .h(px(HEIGHT))
        .rounded_md()
        .border_1()
        .border_color(rgb(0x596273))
        .bg(rgb(0x242832).opacity(0.92))
        .children(node_rects.into_iter().enumerate().map(|(index, bounds)| {
            div()
                .id(SharedString::from(format!("sim-minimap-node-{index}")))
                .absolute()
                .left(px(bounds.origin.x))
                .top(px(bounds.origin.y))
                .w(px(bounds.size.width.max(1.0)))
                .h(px(bounds.size.height.max(1.0)))
                .rounded_sm()
                .bg(rgb(0x77a7ff).opacity(0.72))
        }))
        .child(
            div()
                .id("sim-minimap-viewport")
                .debug_selector(|| "SIM-MINIMAP-VIEWPORT".into())
                .absolute()
                .left(px(viewport_rect.origin.x))
                .top(px(viewport_rect.origin.y))
                .w(px(viewport_rect.size.width.max(1.0)))
                .h(px(viewport_rect.size.height.max(1.0)))
                .border_1()
                .border_color(rgb(0xf2c14e)),
        )
        .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
            if event.keystroke.key == "enter" || event.keystroke.key == "space" {
                cx.stop_propagation();
                this.apply_graph_command(
                    GraphCommand::FitViewport {
                        bounds: keyboard_bounds,
                        available: available_viewport,
                        padding: 40.0,
                    },
                    cx,
                );
            }
        }))
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                cx.stop_propagation();
                let viewport_size = window.viewport_size();
                let viewport_width: f32 = viewport_size.width.into();
                let viewport_height: f32 = viewport_size.height.into();
                let position_x: f32 = event.position.x.into();
                let position_y: f32 = event.position.y.into();
                let graph_point = mouse_transform.unproject_point(GraphPoint {
                    x: position_x - (viewport_width - WIDTH - 12.0),
                    y: position_y - (viewport_height - HEIGHT - 12.0),
                });
                let Some(mut viewport) = this
                    .model
                    .document()
                    .and_then(|document| document.active_graph().ok())
                    .map(|graph| graph.viewport.clone())
                else {
                    return;
                };
                viewport.offset = GraphPoint {
                    x: viewport_width / 2.0 - graph_point.x * viewport.scale,
                    y: viewport_height / 2.0 - graph_point.y * viewport.scale,
                };
                this.apply_graph_command(GraphCommand::SetViewport { viewport }, cx);
            }),
        )
        .into_any_element()
}

fn graph_content_bounds(
    graph: &GraphLevel,
    hidden_nodes: &BTreeSet<GraphIdentifier>,
) -> Option<GraphRect> {
    let mut points = graph
        .nodes
        .values()
        .filter(|node| !hidden_nodes.contains(&node.identifier))
        .flat_map(|node| {
            [
                node.position,
                GraphPoint {
                    x: node.position.x + node.size.width,
                    y: node.position.y + node.size.height,
                },
            ]
        })
        .chain(graph.groups.values().flat_map(|group| {
            [
                group.bounds.origin,
                GraphPoint {
                    x: group.bounds.origin.x + group.bounds.size.width,
                    y: group.bounds.origin.y + group.bounds.size.height,
                },
            ]
        }))
        .chain(graph.reroutes.values().map(|reroute| reroute.position));
    let first = points.next()?;
    let mut minimum_x = first.x;
    let mut minimum_y = first.y;
    let mut maximum_x = first.x;
    let mut maximum_y = first.y;
    for point in points {
        minimum_x = minimum_x.min(point.x);
        minimum_y = minimum_y.min(point.y);
        maximum_x = maximum_x.max(point.x);
        maximum_y = maximum_y.max(point.y);
    }
    Some(GraphRect {
        origin: GraphPoint {
            x: minimum_x,
            y: minimum_y,
        },
        size: GraphSize {
            width: (maximum_x - minimum_x).max(1.0),
            height: (maximum_y - minimum_y).max(1.0),
        },
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MinimapTransform {
    content_bounds: GraphRect,
    scale: f32,
    offset: GraphPoint,
}

impl MinimapTransform {
    pub(crate) fn new(content_bounds: GraphRect, width: f32, height: f32, inset: f32) -> Self {
        let available_width = (width - inset * 2.0).max(1.0);
        let available_height = (height - inset * 2.0).max(1.0);
        let scale_x = available_width / content_bounds.size.width.max(1.0);
        let scale_y = available_height / content_bounds.size.height.max(1.0);
        let scale = scale_x.min(scale_y);
        let projected_width = content_bounds.size.width * scale;
        let projected_height = content_bounds.size.height * scale;
        Self {
            content_bounds,
            scale,
            offset: GraphPoint {
                x: inset + (available_width - projected_width) / 2.0,
                y: inset + (available_height - projected_height) / 2.0,
            },
        }
    }

    pub(crate) fn project_rect(self, graph_rect: GraphRect) -> GraphRect {
        GraphRect {
            origin: GraphPoint {
                x: self.offset.x
                    + (graph_rect.origin.x - self.content_bounds.origin.x) * self.scale,
                y: self.offset.y
                    + (graph_rect.origin.y - self.content_bounds.origin.y) * self.scale,
            },
            size: GraphSize {
                width: graph_rect.size.width * self.scale,
                height: graph_rect.size.height * self.scale,
            },
        }
    }

    pub(crate) fn unproject_point(self, minimap_point: GraphPoint) -> GraphPoint {
        let x = ((minimap_point.x - self.offset.x) / self.scale)
            .clamp(0.0, self.content_bounds.size.width);
        let y = ((minimap_point.y - self.offset.y) / self.scale)
            .clamp(0.0, self.content_bounds.size.height);
        GraphPoint {
            x: self.content_bounds.origin.x + x,
            y: self.content_bounds.origin.y + y,
        }
    }
}

fn handle_scroll_wheel(
    this: &mut GraphWorkspaceItem,
    event: &ScrollWheelEvent,
    _window: &mut Window,
    cx: &mut Context<GraphWorkspaceItem>,
) {
    let delta = event.delta.pixel_delta(px(20.0));
    if event.modifiers.control || event.modifiers.alt {
        let vertical: f32 = delta.y.into();
        let factor = if vertical < 0.0 { 1.1 } else { 1.0 / 1.1 };
        this.zoom_viewport(
            factor,
            GraphPoint {
                x: event.position.x.into(),
                y: event.position.y.into(),
            },
            cx,
        );
    } else {
        this.pan_viewport(
            GraphPoint {
                x: delta.x.into(),
                y: delta.y.into(),
            },
            cx,
        );
    }
}
