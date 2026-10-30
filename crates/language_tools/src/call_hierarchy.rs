use std::collections::{HashMap, HashSet};

use editor::{Editor, ToOffset as _, actions::ShowCallHierarchy};
use gpui::{
    App, AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable, IntoElement,
    Render, ScrollStrategy, SharedString, Task, TaskExt as _, WeakEntity, Window, actions, div,
};
use language::ToOffset as _;
use project::{CallHierarchyCalls, CallHierarchyItem, Location, PreparedCallHierarchy, Project};
use ui::{Button, IconName, prelude::*};
use workspace::{Item, SplitDirection, Workspace};

use crate::language_tool_tree::{
    self, LanguageToolNode, LanguageToolNodeId, LanguageToolProviderStatus, LanguageToolSnapshot,
    LanguageToolTreeHost, language_tool_tree, status_message,
};

actions!(call_hierarchy, [ShowIncomingCalls, ShowOutgoingCalls,]);

const MAX_CALL_HIERARCHY_NODES: usize = 10_000;
const MAX_CALL_HIERARCHY_DEPTH: usize = 32;

fn request_is_current(
    current_generation: u64,
    current_direction: CallHierarchyDirection,
    request_generation: u64,
    request_direction: CallHierarchyDirection,
) -> bool {
    current_generation == request_generation && current_direction == request_direction
}

fn is_cycle(ancestry: &HashSet<String>, identity: &str) -> bool {
    ancestry.contains(identity)
}

fn node_limit_reached(node_count: usize) -> bool {
    node_count >= MAX_CALL_HIERARCHY_NODES
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallHierarchyDirection {
    Incoming,
    Outgoing,
}

impl CallHierarchyDirection {
    fn label(self) -> &'static str {
        match self {
            Self::Incoming => "Incoming Calls",
            Self::Outgoing => "Outgoing Calls",
        }
    }
}

#[derive(Clone, Debug)]
enum ChildrenState {
    Unloaded,
    Loading,
    Loaded(Vec<LanguageToolNodeId>),
}

#[derive(Clone, Debug)]
struct ProjectedCallNode {
    id: LanguageToolNodeId,
    item: CallHierarchyItem,
    ancestry: HashSet<String>,
    depth: usize,
    cycle: bool,
    children: ChildrenState,
}

pub struct CallHierarchyView {
    _workspace: WeakEntity<Workspace>,
    project: Entity<Project>,
    origin_editor: WeakEntity<Editor>,
    root_buffer: Entity<language::Buffer>,
    root_position: text::Anchor,
    focus_handle: FocusHandle,
    host: LanguageToolTreeHost,
    direction: CallHierarchyDirection,
    generation: u64,
    next_node_id: u64,
    root_items: Vec<CallHierarchyItem>,
    roots: Vec<LanguageToolNodeId>,
    nodes: HashMap<LanguageToolNodeId, ProjectedCallNode>,
    root_status: LanguageToolProviderStatus,
    status: LanguageToolProviderStatus,
    prepare_task: Task<()>,
    in_flight: HashMap<LanguageToolNodeId, Task<()>>,
}

pub fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(|workspace, _: &ShowCallHierarchy, window, cx| {
            open_for_active_editor(workspace, CallHierarchyDirection::Incoming, window, cx)
        });
        workspace.register_action(|workspace, _: &ShowIncomingCalls, window, cx| {
            open_for_active_editor(workspace, CallHierarchyDirection::Incoming, window, cx)
        });
        workspace.register_action(|workspace, _: &ShowOutgoingCalls, window, cx| {
            open_for_active_editor(workspace, CallHierarchyDirection::Outgoing, window, cx)
        });
    })
    .detach();
}

fn open_for_active_editor(
    workspace: &mut Workspace,
    direction: CallHierarchyDirection,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let Some(editor) = workspace.active_item_as::<Editor>(cx) else {
        workspace.show_error("Show Call Hierarchy requires an active editor", cx);
        return;
    };
    let Some((buffer, position)) = editor.update(cx, |editor, cx| {
        let selection = editor.selections.newest_anchor();
        let multi_buffer = editor.buffer().read(cx);
        let snapshot = multi_buffer.snapshot(cx);
        let head = selection.map(|anchor| anchor.to_offset(&snapshot)).head();
        multi_buffer.text_anchor_for_position(head, cx)
    }) else {
        workspace.show_error("The cursor is not inside a project-backed buffer", cx);
        return;
    };
    let view = cx.new(|cx| {
        CallHierarchyView::new(
            workspace.weak_handle(),
            workspace.project().clone(),
            editor.downgrade(),
            buffer,
            position,
            direction,
            cx,
        )
    });
    workspace.split_item(SplitDirection::Right, Box::new(view.clone()), window, cx);
    let focus_handle = view.read(cx).focus_handle.clone();
    window.focus(&focus_handle, cx);
}

impl CallHierarchyView {
    fn new(
        workspace: WeakEntity<Workspace>,
        project: Entity<Project>,
        origin_editor: WeakEntity<Editor>,
        root_buffer: Entity<language::Buffer>,
        root_position: text::Anchor,
        direction: CallHierarchyDirection,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();
        let mut view = Self {
            _workspace: workspace,
            project,
            origin_editor,
            root_buffer,
            root_position,
            focus_handle: focus_handle.clone(),
            host: LanguageToolTreeHost::with_focus_handle(focus_handle),
            direction,
            generation: 0,
            next_node_id: 0,
            root_items: Vec::new(),
            roots: Vec::new(),
            nodes: HashMap::new(),
            root_status: LanguageToolProviderStatus::Loading,
            status: LanguageToolProviderStatus::Loading,
            prepare_task: Task::ready(()),
            in_flight: HashMap::new(),
        };
        view.refresh(cx);
        view
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        self.in_flight.clear();
        self.root_items.clear();
        self.roots.clear();
        self.nodes.clear();
        self.root_status = LanguageToolProviderStatus::Loading;
        self.status = LanguageToolProviderStatus::Loading;
        self.rebuild_snapshot();
        let project = self.project.clone();
        let buffer = self.root_buffer.clone();
        let position = self.root_position;
        self.prepare_task = cx.spawn(async move |view, cx| {
            let result = project.update(cx, |project, cx| {
                project.prepare_call_hierarchy(&buffer, position, cx)
            });
            let result = result.await;
            view.update(cx, |view, cx| {
                if view.generation != generation {
                    return;
                }
                view.apply_prepared(result, cx);
            })
            .ok();
        });
    }

    fn apply_prepared(
        &mut self,
        result: anyhow::Result<Option<PreparedCallHierarchy>>,
        cx: &mut Context<Self>,
    ) {
        match result {
            Ok(None) => {
                self.root_status = LanguageToolProviderStatus::Unsupported(
                    "The active language server does not support call hierarchy".to_string(),
                );
                self.status = self.root_status.clone();
            }
            Ok(Some(prepared)) if prepared.items.is_empty() => {
                self.root_status = if prepared.malformed_count > 0 {
                    LanguageToolProviderStatus::Error(
                        "The language server returned no valid call hierarchy items".to_string(),
                    )
                } else {
                    LanguageToolProviderStatus::Empty(
                        "No call hierarchy is available at the cursor".to_string(),
                    )
                };
                self.status = self.root_status.clone();
            }
            Ok(Some(prepared)) => {
                let partial = prepared.truncated || prepared.malformed_count > 0;
                self.root_items = prepared.items;
                self.reset_roots(cx);
                self.root_status = if partial {
                    LanguageToolProviderStatus::Partial(format!(
                        "Call hierarchy is partial; {} malformed item(s) were isolated",
                        prepared.malformed_count
                    ))
                } else {
                    LanguageToolProviderStatus::Current
                };
                self.status = self.root_status.clone();
            }
            Err(error) => {
                self.root_status = if self.project.read(cx).is_disconnected(cx) {
                    LanguageToolProviderStatus::Disconnected(
                        "The authoritative project host is disconnected".to_string(),
                    )
                } else {
                    LanguageToolProviderStatus::Error(format!(
                        "Call hierarchy request failed: {error}"
                    ))
                };
                self.status = self.root_status.clone();
            }
        }
        self.rebuild_snapshot();
        self.host.select_first();
        cx.notify();
    }

    fn reset_roots(&mut self, cx: &App) {
        self.in_flight.clear();
        self.roots.clear();
        self.nodes.clear();
        self.next_node_id = 0;
        for item in self.root_items.clone() {
            if node_limit_reached(self.nodes.len()) {
                self.status = LanguageToolProviderStatus::Partial(format!(
                    "Call hierarchy reached the {MAX_CALL_HIERARCHY_NODES}-node limit"
                ));
                break;
            }
            let identity = identity_for_item(&item, cx);
            let id = self.allocate_node_id();
            self.roots.push(id.clone());
            self.nodes.insert(
                id.clone(),
                ProjectedCallNode {
                    id,
                    item,
                    ancestry: HashSet::from_iter([identity]),
                    depth: 0,
                    cycle: false,
                    children: ChildrenState::Unloaded,
                },
            );
        }
    }

    fn allocate_node_id(&mut self) -> LanguageToolNodeId {
        let id = LanguageToolNodeId(format!("call-hierarchy-node-{}", self.next_node_id));
        self.next_node_id = self.next_node_id.wrapping_add(1);
        id
    }

    fn set_direction(&mut self, direction: CallHierarchyDirection, cx: &mut Context<Self>) {
        if self.direction == direction {
            return;
        }
        self.direction = direction;
        self.generation = self.generation.wrapping_add(1);
        self.reset_roots(cx);
        self.status = self.root_status.clone();
        self.rebuild_snapshot();
        self.host.select_first();
        cx.notify();
    }

    fn toggle_node(
        &mut self,
        id: LanguageToolNodeId,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let was_expanded = self.host.expanded().contains(&id);
        self.host.toggle(&id);
        if was_expanded {
            self.in_flight.remove(&id);
            if let Some(node) = self.nodes.get_mut(&id)
                && matches!(node.children, ChildrenState::Loading)
            {
                node.children = ChildrenState::Unloaded;
                self.rebuild_snapshot();
            }
            return;
        }
        if self
            .nodes
            .get(&id)
            .is_some_and(|node| matches!(node.children, ChildrenState::Unloaded))
        {
            self.load_node(id, cx);
        }
        cx.notify();
    }

    fn load_node(&mut self, id: LanguageToolNodeId, cx: &mut Context<Self>) {
        let Some(node) = self.nodes.get_mut(&id) else {
            return;
        };
        if node.cycle || node.depth >= MAX_CALL_HIERARCHY_DEPTH {
            node.children = ChildrenState::Loaded(Vec::new());
            self.rebuild_snapshot();
            return;
        }
        node.children = ChildrenState::Loading;
        let item = node.item.clone();
        let project = self.project.clone();
        let direction = self.direction;
        let generation = self.generation;
        self.rebuild_snapshot();
        let request_id = id.clone();
        let task = cx.spawn(async move |view, cx| {
            let request = project.update(cx, |project, cx| match direction {
                CallHierarchyDirection::Incoming => project.incoming_calls(item, cx),
                CallHierarchyDirection::Outgoing => project.outgoing_calls(item, cx),
            });
            let result = request.await;
            view.update(cx, |view, cx| {
                if !request_is_current(view.generation, view.direction, generation, direction) {
                    return;
                }
                view.apply_calls(&request_id, result, cx);
            })
            .ok();
        });
        self.in_flight.insert(id, task);
    }

    fn apply_calls(
        &mut self,
        parent_id: &LanguageToolNodeId,
        result: anyhow::Result<CallHierarchyCalls>,
        cx: &mut Context<Self>,
    ) {
        let Some(parent) = self.nodes.get(parent_id).cloned() else {
            return;
        };
        let mut child_ids = Vec::new();
        match result {
            Ok(calls) => {
                if calls.truncated || calls.malformed_count > 0 {
                    self.status = LanguageToolProviderStatus::Partial(format!(
                        "Call hierarchy is partial; {} malformed item(s) were isolated",
                        calls.malformed_count
                    ));
                }
                for call in calls.calls {
                    if node_limit_reached(self.nodes.len()) {
                        self.status = LanguageToolProviderStatus::Partial(format!(
                            "Call hierarchy reached the {MAX_CALL_HIERARCHY_NODES}-node limit"
                        ));
                        break;
                    }
                    let identity = identity_for_item(&call.item, cx);
                    let cycle = is_cycle(&parent.ancestry, &identity);
                    let mut ancestry = parent.ancestry.clone();
                    ancestry.insert(identity.clone());
                    let id = self.allocate_node_id();
                    child_ids.push(id.clone());
                    self.nodes.insert(
                        id.clone(),
                        ProjectedCallNode {
                            id,
                            item: call.item,
                            ancestry,
                            depth: parent.depth + 1,
                            cycle,
                            children: if cycle || parent.depth + 1 >= MAX_CALL_HIERARCHY_DEPTH {
                                ChildrenState::Loaded(Vec::new())
                            } else {
                                ChildrenState::Unloaded
                            },
                        },
                    );
                }
            }
            Err(error) => {
                self.status = if self.project.read(cx).is_disconnected(cx) {
                    LanguageToolProviderStatus::Disconnected(
                        "The authoritative project host is disconnected".to_string(),
                    )
                } else {
                    LanguageToolProviderStatus::Error(format!(
                        "Call hierarchy expansion failed: {error}"
                    ))
                };
            }
        }
        if let Some(parent) = self.nodes.get_mut(parent_id) {
            parent.children = ChildrenState::Loaded(child_ids);
        }
        self.rebuild_snapshot();
        cx.notify();
    }

    fn rebuild_snapshot(&mut self) {
        let roots = self
            .roots
            .iter()
            .filter_map(|id| self.project_node(id))
            .collect();
        self.host.replace_snapshot(LanguageToolSnapshot {
            roots,
            status: self.status.clone(),
        });
    }

    fn project_node(&self, id: &LanguageToolNodeId) -> Option<LanguageToolNode> {
        let node = self.nodes.get(id)?;
        let children = match &node.children {
            ChildrenState::Loaded(children) => children
                .iter()
                .filter_map(|child| self.project_node(child))
                .collect(),
            ChildrenState::Unloaded | ChildrenState::Loading => vec![LanguageToolNode {
                id: LanguageToolNodeId(format!("{}:placeholder", id.0)),
                label: if matches!(node.children, ChildrenState::Loading) {
                    "Loading…".to_string()
                } else {
                    "Expand to load".to_string()
                },
                secondary_label: None,
                icon: None,
                accessibility_label: "Call hierarchy children not loaded".to_string(),
                children: Vec::new(),
                enabled: false,
                activation_label: None,
            }],
        };
        let depth_limited = node.depth >= MAX_CALL_HIERARCHY_DEPTH;
        let state = if node.cycle {
            ", cycle reference"
        } else if depth_limited {
            ", depth limit reached"
        } else {
            ""
        };
        Some(LanguageToolNode {
            id: node.id.clone(),
            label: node.item.name.to_string(),
            secondary_label: node
                .item
                .detail
                .as_ref()
                .map(ToString::to_string)
                .or_else(|| node.cycle.then(|| "cycle reference".to_string())),
            icon: Some(IconName::FileCode),
            accessibility_label: format!(
                "{} call hierarchy item {}{state}",
                self.direction.label(),
                node.item.name
            ),
            children,
            enabled: true,
            activation_label: Some("Open call hierarchy location".to_string()),
        })
    }

    fn activate_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.host.can_activate() {
            return;
        }
        let Some(node) = self
            .host
            .selected()
            .and_then(|id| self.nodes.get(id))
            .cloned()
        else {
            return;
        };
        let location = Location {
            buffer: node.item.buffer,
            range: node.item.selection_range,
        };
        if let Ok(task) = self.origin_editor.update(cx, |editor, cx| {
            editor.open_location(location, false, window, cx)
        }) {
            task.detach_and_log_err(cx);
        }
    }

    fn select_node(
        &mut self,
        id: LanguageToolNodeId,
        click_count: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.host.select(id);
        if click_count > 1 {
            self.activate_selected(window, cx);
        }
        cx.notify();
    }
}

fn identity_for_item(item: &CallHierarchyItem, cx: &App) -> String {
    let buffer = item.buffer.read(cx);
    format!(
        "{}:{:?}:{}:{}",
        item.server_id.to_proto(),
        buffer.remote_id(),
        item.selection_range.start.to_offset(buffer),
        item.selection_range.end.to_offset(buffer),
    )
}

impl Render for CallHierarchyView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let selected = self.host.selected().cloned();
        let rows = self.host.visible_rows_snapshot();
        let status = status_message(self.host.status());
        let scroll_handle = self.host.scroll_handle().clone();
        let weak = cx.weak_entity();
        div()
            .key_context("CallHierarchy")
            .track_focus(&self.focus_handle)
            .size_full()
            .flex()
            .flex_col()
            .on_action(cx.listener(|view, _: &ShowIncomingCalls, _, cx| {
                view.set_direction(CallHierarchyDirection::Incoming, cx)
            }))
            .on_action(cx.listener(|view, _: &ShowOutgoingCalls, _, cx| {
                view.set_direction(CallHierarchyDirection::Outgoing, cx)
            }))
            .on_action(
                cx.listener(|view, _: &language_tool_tree::SelectPrevious, _, cx| {
                    view.host.select_previous();
                    view.host.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|view, _: &language_tool_tree::SelectNext, _, cx| {
                    view.host.select_next();
                    view.host.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|view, _: &language_tool_tree::SelectFirst, _, cx| {
                    view.host.select_first();
                    view.host.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|view, _: &language_tool_tree::SelectLast, _, cx| {
                    view.host.select_last();
                    view.host.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|view, _: &language_tool_tree::SelectParent, _, cx| {
                    view.host.select_parent();
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|view, _: &language_tool_tree::SelectFirstChild, _, cx| {
                    view.host.select_first_child();
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|view, _: &language_tool_tree::ToggleExpanded, window, cx| {
                    if let Some(id) = view.host.selected().cloned() {
                        view.toggle_node(id, window, cx);
                    }
                }),
            )
            .on_action(cx.listener(
                |view, _: &language_tool_tree::ActivateSelected, window, cx| {
                    view.activate_selected(window, cx)
                },
            ))
            .on_action(cx.listener(|view, _: &language_tool_tree::Refresh, _, cx| view.refresh(cx)))
            .on_action(
                cx.listener(|view, _: &language_tool_tree::ExpandAll, _, cx| {
                    let unloaded = view
                        .host
                        .visible_rows()
                        .iter()
                        .filter_map(|row| {
                            view.nodes
                                .get(&row.id)
                                .filter(|node| matches!(node.children, ChildrenState::Unloaded))
                                .map(|_| row.id.clone())
                        })
                        .collect::<Vec<_>>();
                    view.host.expand_all();
                    for id in unloaded {
                        view.load_node(id, cx);
                    }
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|view, _: &language_tool_tree::CollapseAll, _, cx| {
                    view.in_flight.clear();
                    for node in view.nodes.values_mut() {
                        if matches!(node.children, ChildrenState::Loading) {
                            node.children = ChildrenState::Unloaded;
                        }
                    }
                    view.rebuild_snapshot();
                    view.host.collapse_all();
                    cx.notify();
                }),
            )
            .child(
                h_flex()
                    .h_9()
                    .px_2()
                    .gap_2()
                    .child("Call Hierarchy")
                    .child(
                        Button::new("call-hierarchy-incoming", "Incoming")
                            .disabled(self.direction == CallHierarchyDirection::Incoming)
                            .on_click(cx.listener(|view, _, _, cx| {
                                view.set_direction(CallHierarchyDirection::Incoming, cx)
                            })),
                    )
                    .child(
                        Button::new("call-hierarchy-outgoing", "Outgoing")
                            .disabled(self.direction == CallHierarchyDirection::Outgoing)
                            .on_click(cx.listener(|view, _, _, cx| {
                                view.set_direction(CallHierarchyDirection::Outgoing, cx)
                            })),
                    ),
            )
            .child(language_tool_tree(
                rows,
                selected,
                status,
                scroll_handle,
                {
                    let weak = weak.clone();
                    move |id, click_count, window, cx| {
                        weak.update(cx, |view, cx| view.select_node(id, click_count, window, cx))
                            .ok();
                    }
                },
                move |id, window, cx| {
                    weak.update(cx, |view, cx| view.toggle_node(id, window, cx))
                        .ok();
                },
                |_, _, _, _| {},
            ))
    }
}

impl EventEmitter<()> for CallHierarchyView {}

impl Focusable for CallHierarchyView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Item for CallHierarchyView {
    type Event = ();

    fn to_item_events(_: &Self::Event, _: &mut dyn FnMut(workspace::item::ItemEvent)) {}

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        "Call Hierarchy".into()
    }

    fn telemetry_event_text(&self) -> Option<&'static str> {
        None
    }

    fn can_split(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_hierarchy_cycles_and_direction_generations_cancel_stale_work() {
        let ancestry: HashSet<String> = HashSet::from_iter(["root".to_string()]);
        assert!(is_cycle(&ancestry, "root"));
        assert!(!is_cycle(&ancestry, "child"));
        assert!(request_is_current(
            7,
            CallHierarchyDirection::Incoming,
            7,
            CallHierarchyDirection::Incoming,
        ));
        assert!(!request_is_current(
            8,
            CallHierarchyDirection::Outgoing,
            7,
            CallHierarchyDirection::Incoming,
        ));
    }

    #[test]
    fn call_hierarchy_large_projection_stops_at_the_reviewed_node_limit() {
        let mut admitted = 0;
        while !node_limit_reached(admitted) {
            admitted += 1;
        }
        assert_eq!(admitted, 10_000);
        assert!(node_limit_reached(admitted));

        let mut host = LanguageToolTreeHost::default();
        host.replace_snapshot(LanguageToolSnapshot {
            roots: (0..admitted)
                .map(|index| LanguageToolNode {
                    id: LanguageToolNodeId(format!("call-{index}")),
                    label: format!("function_{index}"),
                    secondary_label: None,
                    icon: Some(IconName::FileCode),
                    accessibility_label: format!("Incoming Calls item function_{index}"),
                    children: Vec::new(),
                    enabled: true,
                    activation_label: Some("Open call hierarchy location".to_string()),
                })
                .collect(),
            status: LanguageToolProviderStatus::Partial(
                "Call hierarchy reached the 10000-node limit".to_string(),
            ),
        });
        assert_eq!(host.visible_rows().len(), MAX_CALL_HIERARCHY_NODES);
        assert_eq!(host.visible_rows_in_range(9_975..10_000).len(), 25);
        host.select_first();
        host.select_next();
        assert_eq!(host.selected().map(|id| id.0.as_str()), Some("call-1"));
        assert!(
            host.visible_rows()[1]
                .accessibility_label
                .contains("Incoming Calls")
        );
    }
}
