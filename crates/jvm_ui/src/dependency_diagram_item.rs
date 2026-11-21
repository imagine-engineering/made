//! The dependency diagram editor item: a dedicated tab hosting the read-only
//! graph canvas for one selected project or module. The item owns the canvas
//! viewport, selection and inspection overlay; admitted facts come from the
//! Dependencies panel's snapshot at open time and are re-derived without
//! provider queries.

use crate::dependencies_panel::{
    AddGraphSelectionToSet, ClearGraphInspection, ClearGraphSelectionSet,
    CopyDependencyDiagramSource, InspectGraphNeighbours, InspectGraphPathsBetween,
    InspectGraphRootPaths, InspectGraphSelectionSet, PinGraphSelection,
};
use crate::dependency_graph_canvas::CanvasState;
use gpui::{
    App, AppContext as _, Context, DismissEvent, EventEmitter, FocusHandle, Focusable, FontWeight,
    Render, SharedString, actions,
};
use std::collections::BTreeSet;
use workspace::Workspace;

actions!(
    groovy_dependencies_panel,
    [
        ShowDependencyDiagramPopup,
        DismissDependencyDiagramPopup,
        FitDependencyDiagram,
        ZoomDependencyDiagramIn,
        ZoomDependencyDiagramOut,
        ExportDependencyDiagramImage,
        ToggleDependencyDiagramStructure
    ]
);

/// Applies a scope choice from the diagram scope picker.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct SelectDependencyDiagramScope {
    pub label: String,
}

/// Reveals and selects the node with the chosen label, from the search popup.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct RevealDependencyDiagramNode {
    pub label: String,
}

/// The scope choice applied when re-deriving the canvas from admitted facts.
/// Gradle admits one provider-reported configuration; Maven admits edges whose
/// scope matches the named Maven fact-filter scope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DependencyDiagramScope {
    GradleConfiguration(String),
    MavenScope(String),
}

impl DependencyDiagramScope {
    pub fn label(&self) -> String {
        match self {
            Self::GradleConfiguration(configuration) => configuration.clone(),
            Self::MavenScope(label) => label.clone(),
        }
    }
}

/// Everything the item needs to re-derive the canvas from admitted facts
/// without provider queries, plus the current scope choice and the options the
/// provider reports for the module.
#[derive(Clone, Debug)]
pub struct DependencyDiagramDerivation {
    pub canvas: crate::dependency_graph_canvas::CanvasGraph,
    pub target: DependencyDiagramTarget,
    pub selected: BTreeSet<String>,
    pub scope: DependencyDiagramScope,
    pub scope_options: Vec<String>,
}

/// The admitted-facts source identity behind a diagram, so the item can
/// re-derive on scope changes without provider queries.
#[derive(Clone, Debug)]
pub struct DependencyDiagramSource {
    pub project: gpui::Entity<project::Project>,
    pub selection: crate::GraphSelection,
    pub node: String,
    pub view: crate::BuildToolView,
    /// The opening panel, for context-menu command routing through the
    /// panel's own action handlers and enablement logic.
    pub panel: gpui::WeakEntity<crate::dependencies_panel::DependenciesPanel>,
}

/// The bundle the diagram surfaces hand to the item or popup when opening:
/// the initial derivation plus the source needed for later re-derivations.
#[derive(Clone, Debug)]
pub struct DependencyDiagramOpen {
    pub derivation: DependencyDiagramDerivation,
    pub source: DependencyDiagramSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyDiagramProvider {
    Gradle,
    Maven,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyDiagramTarget {
    pub module_label: String,
    pub provider: DependencyDiagramProvider,
    pub configuration: Option<String>,
}

impl DependencyDiagramTarget {
    pub fn title(&self) -> String {
        match (&self.provider, &self.configuration) {
            (DependencyDiagramProvider::Gradle, Some(configuration)) => {
                format!(
                    "Dependencies — {self_module} ({configuration})",
                    self_module = self.module_label
                )
            }
            _ => format!("Dependencies — {}", self.module_label),
        }
    }
}

pub enum DependencyDiagramEvent {}

pub struct DependencyDiagramItem {
    pub(crate) canvas: gpui::Entity<CanvasState>,
    target: DependencyDiagramTarget,
    source: Option<DependencyDiagramSource>,
    scope: DependencyDiagramScope,
    scope_options: Vec<String>,
    structure_visible: bool,
    overview_drag: bool,
    overview_origin: std::sync::Arc<std::sync::Mutex<sim_runtime::graph::GraphPoint>>,
    context_state: std::sync::Arc<crate::dependency_diagram_menu::DependencyDiagramContextState>,
    context_handle: ui::RightClickMenuHandle<ui::ContextMenu>,
    pub(crate) notice: Option<String>,
    focus_handle: FocusHandle,
}

impl EventEmitter<DependencyDiagramEvent> for DependencyDiagramItem {}

impl Focusable for DependencyDiagramItem {
    fn focus_handle(&self, _: &gpui::App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl DependencyDiagramItem {
    pub fn new(
        derivation: DependencyDiagramDerivation,
        source: Option<DependencyDiagramSource>,
        cx: &mut App,
    ) -> Self {
        let mut canvas = CanvasState::new(derivation.canvas);
        canvas.selection = derivation.selected;
        let context_state = std::sync::Arc::new(
            crate::dependency_diagram_menu::DependencyDiagramContextState::default(),
        );
        let context_handle = ui::RightClickMenuHandle::default();
        canvas.context_state = Some(context_state.clone());
        canvas.context_handle = Some(context_handle.clone());
        Self {
            canvas: cx.new(|_| canvas),
            target: derivation.target,
            source,
            scope: derivation.scope,
            scope_options: derivation.scope_options,
            structure_visible: false,
            overview_drag: false,
            overview_origin: std::sync::Arc::new(std::sync::Mutex::new(
                sim_runtime::graph::GraphPoint::ZERO,
            )),
            context_state,
            context_handle,
            notice: None,
            focus_handle: cx.focus_handle(),
        }
    }

    /// Opens the diagram as a focused editor tab in the workspace's active
    /// pane and returns the item handle.
    pub fn open(
        derivation: DependencyDiagramDerivation,
        source: Option<DependencyDiagramSource>,
        workspace: &mut Workspace,
        window: &mut gpui::Window,
        cx: &mut App,
    ) -> gpui::Entity<Self> {
        let item = cx.new(|cx| Self::new(derivation, source, cx));
        workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
        item
    }

    pub fn canvas(&self) -> &gpui::Entity<CanvasState> {
        &self.canvas
    }

    pub fn target(&self) -> &DependencyDiagramTarget {
        &self.target
    }

    pub fn scope(&self) -> &DependencyDiagramScope {
        &self.scope
    }

    pub fn scope_options(&self) -> &[String] {
        &self.scope_options
    }

    /// Fits the graph bounds into the canvas' last painted size.
    pub fn fit_canvas(&mut self, cx: &mut Context<Self>) {
        self.canvas.update(cx, |canvas, cx| {
            let size = *canvas.canvas_size.lock().expect("canvas size");
            canvas.fit(size);
            cx.notify();
        });
    }

    /// Zooms the canvas in or out around its centre by `factor`.
    pub fn zoom_canvas(&mut self, factor: f32, cx: &mut Context<Self>) {
        self.canvas.update(cx, |canvas, cx| {
            let size = *canvas.canvas_size.lock().expect("canvas size");
            let anchor = sim_runtime::graph::GraphPoint {
                x: size.width / 2.0,
                y: size.height / 2.0,
            };
            canvas.zoom(factor, anchor);
            cx.notify();
        });
    }

    /// Copies the exact Mermaid export payload derived from the same admitted
    /// facts, matching the panel's Copy Diagram Source behaviour.
    pub fn copy_diagram_source(&mut self, cx: &mut Context<Self>) {
        let Some(source) = &self.source else {
            self.notice = Some(
                "No provider source is attached to this diagram; Copy diagram source requires a current selection"
                    .into(),
            );
            cx.notify();
            return;
        };
        let project = source.project.read(cx);
        match crate::dependencies_panel::dependency_diagram_source_export(
            project,
            source.view,
            &source.selection,
            cx,
        ) {
            Some(source_text) => {
                let characters = source_text.chars().count();
                cx.write_to_clipboard(gpui::ClipboardItem::new_string(source_text));
                self.notice = Some(format!(
                    "Diagram source copied to the clipboard; {characters} characters; the copied source is the exact input the displayed diagram renders"
                ));
            }
            None => {
                self.notice = Some(
                    "No admitted graph facts are available for Copy diagram source; show a current diagram, then retry".into(),
                );
            }
        }
        cx.notify();
    }

    /// Re-derives the canvas from admitted facts under the chosen scope
    /// without any provider query.
    pub fn apply_scope(&mut self, label: String, cx: &mut Context<Self>) {
        let Some(source) = self.source.clone() else {
            return;
        };
        let scope = match &self.scope {
            DependencyDiagramScope::GradleConfiguration(_) => {
                DependencyDiagramScope::GradleConfiguration(label)
            }
            DependencyDiagramScope::MavenScope(_) => DependencyDiagramScope::MavenScope(label),
        };
        let project = source.project.read(cx);
        match crate::dependencies_panel::derive_dependency_diagram(
            project,
            source.view,
            &source.selection,
            &source.node,
            Some(&scope),
            cx,
        ) {
            Some(derivation) => {
                self.canvas = cx.new(|_| {
                    let mut canvas = CanvasState::new(derivation.canvas);
                    canvas.selection = derivation.selected;
                    canvas
                });
                self.target = derivation.target;
                self.scope = derivation.scope;
                self.scope_options = derivation.scope_options;
                self.notice = None;
            }
            None => {
                self.notice = Some(
                    "No admitted graph facts match the chosen scope; refresh provider graph facts, then retry".into(),
                );
            }
        }
        cx.notify();
    }

    /// Bounded node-label matches for the search popup, in node order.
    pub fn search_matches(&self, query: &str, cx: &App) -> Vec<String> {
        const MAX_SEARCH_MATCHES: usize = 16;
        let query = query.to_lowercase();
        self.canvas
            .read(cx)
            .graph
            .nodes
            .iter()
            .filter(|node| node.label.to_lowercase().contains(&query))
            .map(|node| node.label.clone())
            .take(MAX_SEARCH_MATCHES)
            .collect()
    }

    /// Selects and reveals the first admitted node whose label matches, panning
    /// the canvas so the node sits at its centre.
    pub fn reveal_node_by_label(&mut self, label: &str, cx: &mut Context<Self>) {
        let node_id = self
            .canvas
            .read(cx)
            .graph
            .nodes
            .iter()
            .find(|node| node.label == label)
            .map(|node| node.id.clone());
        let Some(node_id) = node_id else {
            self.notice = Some(
                "The search result is no longer part of the admitted graph; re-derive the diagram, then search again".into(),
            );
            cx.notify();
            return;
        };
        self.canvas.update(cx, |canvas, cx| {
            canvas.reveal_node(&node_id);
            cx.notify();
        });
        self.notice = None;
        cx.notify();
    }

    /// Toggles the minimap-style Structure overview panel.
    pub fn toggle_structure(&mut self, cx: &mut Context<Self>) {
        self.structure_visible = !self.structure_visible;
        cx.notify();
    }

    /// Routes a panel-level command through the opening panel's own action
    /// scope, so menu invocations share the panel's enablement logic.
    fn forward_to_panel<A: gpui::Action>(
        &mut self,
        event: &A,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        let focus = self.source.as_ref().and_then(|source| {
            source
                .panel
                .upgrade()
                .map(|panel| panel.read(cx).routing_focus_handle())
        });
        if let Some(focus) = focus {
            focus.dispatch_action(event, window, cx);
        }
    }

    /// Exports the current canvas as a deterministic SVG image and copies it
    /// to the clipboard; distinct from the Mermaid Copy Diagram Source payload.
    pub fn export_diagram_image(&mut self, cx: &mut Context<Self>) {
        let svg = crate::dependency_graph_canvas::canvas_graph_svg(&self.canvas.read(cx).graph);
        let bytes = svg.chars().count();
        cx.write_to_clipboard(gpui::ClipboardItem::new_string(svg));
        self.notice = Some(format!(
            "Diagram image exported as SVG and copied to the clipboard; {bytes} characters"
        ));
        cx.notify();
    }
}

impl Render for DependencyDiagramItem {
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) -> impl gpui::IntoElement {
        use ui::prelude::*;

        let theme = cx.theme().colors().clone();
        let title = self.target.title();
        let scope_label = self.scope.label();
        let scope_options = self.scope_options.clone();
        let search_matches = self.search_matches("", cx);
        let structure_visible = self.structure_visible;
        let notice = self.notice.clone();

        let toolbar = div()
            .flex()
            .items_center()
            .gap_1()
            .px_2()
            .py_1()
            .border_1()
            .border_color(theme.border)
            .text_color(theme.text)
            .child(
                div()
                    .flex_1()
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .child(title),
            )
            .child(
                IconButton::new("dependency-diagram-fit-icon", ui::IconName::Maximize)
                    .shape(ui::IconButtonShape::Square)
                    .tab_index(0_isize)
                    .aria_label("Fit dependency diagram")
                    .tooltip(ui::Tooltip::text("Fit diagram"))
                    .on_click(cx.listener(|item, _, _, cx| item.fit_canvas(cx))),
            )
            .child(
                IconButton::new("dependency-diagram-zoom-in-icon", ui::IconName::Plus)
                    .shape(ui::IconButtonShape::Square)
                    .tab_index(0_isize)
                    .aria_label("Zoom dependency diagram in")
                    .tooltip(ui::Tooltip::text("Zoom in"))
                    .on_click(cx.listener(|item, _, _, cx| item.zoom_canvas(1.25, cx))),
            )
            .child(
                IconButton::new(
                    "dependency-diagram-zoom-out-icon",
                    ui::IconName::SquareMinus,
                )
                .shape(ui::IconButtonShape::Square)
                .tab_index(0_isize)
                .aria_label("Zoom dependency diagram out")
                .tooltip(ui::Tooltip::text("Zoom out"))
                .on_click(cx.listener(|item, _, _, cx| item.zoom_canvas(0.8, cx))),
            )
            .child(
                ui::PopoverMenu::new("dependency-diagram-scope")
                    .trigger(
                        IconButton::new("dependency-diagram-scope-icon", ui::IconName::Filter)
                            .shape(ui::IconButtonShape::Square)
                            .tab_index(0_isize)
                            .aria_label(format!(
                                "Dependency diagram scope; current scope {scope_label}"
                            ))
                            .tooltip(ui::Tooltip::text(format!("Scope: {scope_label}"))),
                    )
                    .menu(move |window, cx| {
                        Some(ui::ContextMenu::build(window, cx, |menu, _, _| {
                            let mut menu = menu;
                            for option in &scope_options {
                                menu = menu.action(
                                    option.clone(),
                                    Box::new(SelectDependencyDiagramScope {
                                        label: option.clone(),
                                    }),
                                );
                            }
                            menu
                        }))
                    }),
            )
            .child(
                ui::PopoverMenu::new("dependency-diagram-search")
                    .trigger(
                        IconButton::new(
                            "dependency-diagram-search-icon",
                            ui::IconName::MagnifyingGlass,
                        )
                        .shape(ui::IconButtonShape::Square)
                        .tab_index(0_isize)
                        .aria_label("Search dependency diagram nodes")
                        .tooltip(ui::Tooltip::text("Search dependencies")),
                    )
                    .menu({
                        move |window, cx| {
                            Some(ui::ContextMenu::build(window, cx, |menu, _, _| {
                                let mut menu = menu;
                                if search_matches.is_empty() {
                                    menu = menu.action_disabled_when(
                                        true,
                                        "No admitted dependency nodes",
                                        Box::new(RevealDependencyDiagramNode {
                                            label: String::new(),
                                        }),
                                    );
                                }
                                for label in &search_matches {
                                    menu = menu.action(
                                        label.clone(),
                                        Box::new(RevealDependencyDiagramNode {
                                            label: label.clone(),
                                        }),
                                    );
                                }
                                menu
                            }))
                        }
                    }),
            )
            .child(
                IconButton::new(
                    "dependency-diagram-structure-icon",
                    if structure_visible {
                        ui::IconName::EyeOff
                    } else {
                        ui::IconName::Eye
                    },
                )
                .shape(ui::IconButtonShape::Square)
                .tab_index(0_isize)
                .aria_label(if structure_visible {
                    "Hide dependency diagram Structure overview"
                } else {
                    "Show dependency diagram Structure overview"
                })
                .tooltip(ui::Tooltip::text("Structure overview"))
                .on_click(cx.listener(|item, _, _, cx| item.toggle_structure(cx))),
            )
            .child(
                IconButton::new("dependency-diagram-copy-source-icon", ui::IconName::Copy)
                    .shape(ui::IconButtonShape::Square)
                    .tab_index(0_isize)
                    .aria_label("Copy dependency diagram source")
                    .tooltip(ui::Tooltip::text("Copy Diagram Source"))
                    .on_click(cx.listener(|item, _, _, cx| item.copy_diagram_source(cx))),
            )
            .child(
                IconButton::new("dependency-diagram-export-icon", ui::IconName::Image)
                    .shape(ui::IconButtonShape::Square)
                    .tab_index(0_isize)
                    .aria_label("Export dependency diagram image")
                    .tooltip(ui::Tooltip::text("Export Image"))
                    .on_click(cx.listener(|item, _, _, cx| item.export_diagram_image(cx))),
            );

        // Keep the menu availability snapshot fresh: the canvas stages the
        // invocation and the menu builder reads this state.
        let snapshot = {
            let panel = self
                .source
                .as_ref()
                .and_then(|source| source.panel.upgrade());
            let (selection_set_len, has_inspection) = panel
                .as_ref()
                .map(|panel| panel.read(cx).context_menu_availability())
                .unwrap_or((0, false));
            crate::dependency_diagram_menu::DependencyDiagramMenuSnapshot {
                selection_len: self.canvas.read(cx).selection.len(),
                selection_set_len,
                has_inspection,
                has_source: self.source.is_some(),
            }
        };
        *self.context_state.snapshot.lock().expect("menu snapshot") = snapshot;
        let context_state = self.context_state.clone();
        let context_handle = self.context_handle.clone();
        let canvas_element = self.canvas.clone();
        let canvas_wrapper = div()
            .relative()
            .flex_1()
            .min_h_0()
            .child(
                ui::right_click_menu("dependency-diagram-context-menu")
                    .with_handle(context_handle)
                    .maybe_menu(move |window, cx| {
                        crate::dependency_diagram_menu::build_dependency_diagram_context_menu(
                            &context_state,
                            window,
                            cx,
                        )
                    })
                    .trigger(move |_, _, _| canvas_element),
            )
            .when(structure_visible, |wrapper| {
                wrapper.child(self.render_structure_overview(cx))
            });

        div()
            .id("groovy-dependency-diagram-item")
            .size_full()
            .flex()
            .flex_col()
            .bg(theme.background)
            .on_action(cx.listener(|item, _: &FitDependencyDiagram, _, cx| item.fit_canvas(cx)))
            .on_action(
                cx.listener(|item, _: &ZoomDependencyDiagramIn, _, cx| item.zoom_canvas(1.25, cx)),
            )
            .on_action(
                cx.listener(|item, _: &ZoomDependencyDiagramOut, _, cx| item.zoom_canvas(0.8, cx)),
            )
            .on_action(cx.listener(|item, _: &CopyDependencyDiagramSource, _, cx| {
                item.copy_diagram_source(cx)
            }))
            .on_action(
                cx.listener(|item, _: &ExportDependencyDiagramImage, _, cx| {
                    item.export_diagram_image(cx)
                }),
            )
            .on_action(
                cx.listener(|item, _: &ToggleDependencyDiagramStructure, _, cx| {
                    item.toggle_structure(cx)
                }),
            )
            .on_action(
                cx.listener(|item, event: &SelectDependencyDiagramScope, _, cx| {
                    item.apply_scope(event.label.clone(), cx)
                }),
            )
            .on_action(
                cx.listener(|item, event: &RevealDependencyDiagramNode, _, cx| {
                    item.reveal_node_by_label(&event.label, cx)
                }),
            )
            .on_action(
                cx.listener(|item, event: &InspectGraphNeighbours, window, cx| {
                    item.forward_to_panel(event, window, cx)
                }),
            )
            .on_action(
                cx.listener(|item, event: &InspectGraphRootPaths, window, cx| {
                    item.forward_to_panel(event, window, cx)
                }),
            )
            .on_action(
                cx.listener(|item, event: &InspectGraphPathsBetween, window, cx| {
                    item.forward_to_panel(event, window, cx)
                }),
            )
            .on_action(cx.listener(|item, event: &PinGraphSelection, window, cx| {
                item.forward_to_panel(event, window, cx)
            }))
            .on_action(
                cx.listener(|item, event: &AddGraphSelectionToSet, window, cx| {
                    item.forward_to_panel(event, window, cx)
                }),
            )
            .on_action(
                cx.listener(|item, event: &InspectGraphSelectionSet, window, cx| {
                    item.forward_to_panel(event, window, cx)
                }),
            )
            .on_action(
                cx.listener(|item, event: &ClearGraphSelectionSet, window, cx| {
                    item.forward_to_panel(event, window, cx)
                }),
            )
            .on_action(
                cx.listener(|item, event: &ClearGraphInspection, window, cx| {
                    item.forward_to_panel(event, window, cx)
                }),
            )
            .child(toolbar)
            .child(canvas_wrapper)
            .when_some(notice, |root, notice_text| {
                root.child(
                    div()
                        .id("dependency-diagram-notice")
                        .px_2()
                        .py_1()
                        .border_1()
                        .border_color(theme.border)
                        .text_size(px(12.0))
                        .text_color(theme.text_muted)
                        .child(notice_text),
                )
            })
    }
}

impl DependencyDiagramItem {
    /// The minimap-style Structure overview: admitted nodes painted as scaled
    /// rectangles inside the graph bounds, with the current selection
    /// highlighted. Clicking or dragging recentres the canvas viewport on the
    /// pointed graph position.
    fn render_structure_overview(&self, cx: &Context<Self>) -> impl gpui::IntoElement {
        use ui::prelude::*;

        const OVERVIEW_WIDTH: f32 = 180.0;
        const OVERVIEW_HEIGHT: f32 = 120.0;
        let theme = cx.theme().colors().clone();
        let graph = &self.canvas.read(cx).graph;
        let bounds = graph.bounds;
        let scale = if bounds.size.width > 0.0 && bounds.size.height > 0.0 {
            (OVERVIEW_WIDTH / bounds.size.width)
                .min(OVERVIEW_HEIGHT / bounds.size.height)
                .min(1.0)
        } else {
            0.0
        };
        let overview_origin = self.overview_origin.clone();
        let mut overview = div()
            .absolute()
            .right_2()
            .bottom_2()
            .w(px(OVERVIEW_WIDTH))
            .h(px(OVERVIEW_HEIGHT))
            .border_1()
            .border_color(theme.border)
            .bg(theme.background.opacity(0.9))
            .overflow_hidden()
            .on_mouse_down(
                gpui::MouseButton::Left,
                cx.listener(|item, event: &gpui::MouseDownEvent, _, cx| {
                    item.overview_drag = true;
                    item.move_viewport_to_overview(
                        sim_runtime::graph::GraphPoint {
                            x: event.position.x.into(),
                            y: event.position.y.into(),
                        },
                        cx,
                    );
                }),
            )
            .on_mouse_move(cx.listener(|item, event: &gpui::MouseMoveEvent, _, cx| {
                if item.overview_drag {
                    item.move_viewport_to_overview(
                        sim_runtime::graph::GraphPoint {
                            x: event.position.x.into(),
                            y: event.position.y.into(),
                        },
                        cx,
                    );
                }
            }))
            .on_mouse_up(
                gpui::MouseButton::Left,
                cx.listener(|item, _: &gpui::MouseUpEvent, _, cx| {
                    item.overview_drag = false;
                    cx.notify();
                }),
            )
            .child(gpui::canvas(
                move |bounds, _, _| {
                    *overview_origin.lock().expect("overview origin") =
                        sim_runtime::graph::GraphPoint {
                            x: bounds.origin.x.into(),
                            y: bounds.origin.y.into(),
                        };
                },
                |_, _, _, _| {},
            ));
        if scale > 0.0 {
            for node in graph.nodes.iter().take(64) {
                let Some(origin) = graph.positions.get(&node.id).copied() else {
                    continue;
                };
                let selected = self.canvas.read(cx).selection.contains(&node.id);
                let rect = div()
                    .absolute()
                    .left(px((origin.x - bounds.origin.x) * scale))
                    .top(px((origin.y - bounds.origin.y) * scale))
                    .w(px(3.0_f32.max(graph.node_size().width * scale)))
                    .h(px(2.0_f32.max(graph.node_size().height * scale)));
                if selected {
                    overview = overview.child(
                        rect.id(format!("structure-overview-{}", node.id))
                            .bg(theme.border_selected)
                            .role(gpui::Role::Button)
                            .aria_label(format!(
                                "Structure overview selected node {}; activate to centre the viewport",
                                node.label
                            )),
                    );
                } else {
                    overview = overview.child(rect.bg(theme.border));
                }
            }
        }
        overview
    }

    /// Recentres the canvas viewport on the graph position the pointer points
    /// at inside the Structure overview.
    fn move_viewport_to_overview(
        &mut self,
        position: sim_runtime::graph::GraphPoint,
        cx: &mut Context<Self>,
    ) {
        let overview_origin = *self.overview_origin.lock().expect("overview origin");
        let (bounds, canvas_size, scale) = {
            let canvas = self.canvas.read(cx);
            (
                canvas.graph.bounds,
                *canvas.canvas_size.lock().expect("canvas size"),
                self.overview_scale(cx),
            )
        };
        if scale <= 0.0 || bounds.size.width <= 0.0 || bounds.size.height <= 0.0 {
            return;
        }
        let local = sim_runtime::graph::GraphPoint {
            x: position.x - overview_origin.x,
            y: position.y - overview_origin.y,
        };
        let graph_position = sim_runtime::graph::GraphPoint {
            x: bounds.origin.x + local.x / scale,
            y: bounds.origin.y + local.y / scale,
        };
        self.canvas.update(cx, |canvas, cx| {
            canvas.viewport.offset = sim_runtime::graph::GraphPoint {
                x: canvas_size.width / 2.0 - graph_position.x * canvas.viewport.scale,
                y: canvas_size.height / 2.0 - graph_position.y * canvas.viewport.scale,
            };
            cx.notify();
        });
    }

    fn overview_scale(&self, cx: &App) -> f32 {
        const OVERVIEW_WIDTH: f32 = 180.0;
        const OVERVIEW_HEIGHT: f32 = 120.0;
        let bounds = self.canvas.read(cx).graph.bounds;
        if bounds.size.width > 0.0 && bounds.size.height > 0.0 {
            (OVERVIEW_WIDTH / bounds.size.width)
                .min(OVERVIEW_HEIGHT / bounds.size.height)
                .min(1.0)
        } else {
            0.0
        }
    }
}

impl workspace::Item for DependencyDiagramItem {
    type Event = DependencyDiagramEvent;

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        self.target.title().into()
    }

    fn tab_icon(&self, _window: &gpui::Window, _cx: &App) -> Option<ui::Icon> {
        Some(ui::Icon::new(match self.target.provider {
            DependencyDiagramProvider::Gradle => ui::IconName::Gradle,
            DependencyDiagramProvider::Maven => ui::IconName::Maven,
        }))
    }
}

/// Diagram tabs are session-scoped: admitted facts come from provider
/// collections and are not persisted, so restored sessions drop the tab
/// instead of reopening a stale or empty diagram.
impl workspace::SerializableItem for DependencyDiagramItem {
    fn serialized_item_kind() -> &'static str {
        "DependencyDiagramItem"
    }

    fn cleanup(
        _: workspace::WorkspaceId,
        _: Vec<workspace::ItemId>,
        _: &mut gpui::Window,
        _: &mut App,
    ) -> gpui::Task<anyhow::Result<()>> {
        gpui::Task::ready(Ok(()))
    }

    fn deserialize(
        _: gpui::Entity<project::Project>,
        _: gpui::WeakEntity<Workspace>,
        _: workspace::WorkspaceId,
        _: workspace::ItemId,
        _: &mut gpui::Window,
        _: &mut App,
    ) -> gpui::Task<anyhow::Result<gpui::Entity<Self>>> {
        gpui::Task::ready(Err(anyhow::anyhow!(
            "Dependency diagram tabs are session-scoped and are not restored"
        )))
    }

    fn serialize(
        &mut self,
        _: &mut Workspace,
        _: workspace::ItemId,
        _: bool,
        _: &mut Context<Self>,
    ) -> Option<gpui::Task<anyhow::Result<()>>> {
        Some(gpui::Task::ready(Ok(())))
    }

    fn should_serialize(&self, _: &Self::Event) -> bool {
        false
    }
}

/// Escape dismisses the diagram popup; the workspace modal layer hides it on
/// the emitted dismiss event.
impl EventEmitter<DismissEvent> for DependencyDiagramPopup {}

impl Focusable for DependencyDiagramPopup {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl workspace::ModalView for DependencyDiagramPopup {}

/// Presents the same read-only canvas as the editor tab, in a workspace modal
/// dismissed by Escape.
pub struct DependencyDiagramPopup {
    canvas: gpui::Entity<CanvasState>,
    target: DependencyDiagramTarget,
    focus_handle: FocusHandle,
}

impl DependencyDiagramPopup {
    pub fn new(
        graph: crate::dependency_graph_canvas::CanvasGraph,
        target: DependencyDiagramTarget,
        selected: BTreeSet<String>,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut canvas = CanvasState::new(graph);
        canvas.selection = selected;
        Self {
            canvas: cx.new(|_| canvas),
            target,
            focus_handle: cx.focus_handle(),
        }
    }

    pub fn canvas(&self) -> &gpui::Entity<CanvasState> {
        &self.canvas
    }

    pub fn target(&self) -> &DependencyDiagramTarget {
        &self.target
    }
}

impl Render for DependencyDiagramPopup {
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) -> impl gpui::IntoElement {
        use ui::prelude::*;

        let theme = cx.theme().colors().clone();
        let title = self.target.title();
        div()
            .key_context("DependencyDiagramPopup")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &DismissDependencyDiagramPopup, _, cx| {
                cx.emit(DismissEvent);
            }))
            .w(gpui::Pixels::from(960.0))
            .h(gpui::Pixels::from(640.0))
            .flex()
            .flex_col()
            .bg(theme.background)
            .border_1()
            .border_color(theme.border)
            .rounded_md()
            .overflow_hidden()
            .child(
                div()
                    .px_2()
                    .py_1()
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text)
                    .child(title),
            )
            .child(self.canvas.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dependency_graph_canvas::{
        CanvasEdge, CanvasEdgeVariant, CanvasNode, CanvasNodeKind, layout_canvas_graph,
    };
    use gpui::TestAppContext;
    use settings::SettingsStore;
    use theme::LoadThemes;
    use workspace::Item as _;

    fn fixture_graph() -> crate::dependency_graph_canvas::CanvasGraph {
        let mut graph = crate::dependency_graph_canvas::CanvasGraph {
            nodes: vec![
                CanvasNode {
                    id: "root".into(),
                    component_id: "root".into(),
                    label: "org.example:app".into(),
                    version: None,
                    kind: CanvasNodeKind::Root,
                },
                CanvasNode {
                    id: "c1".into(),
                    component_id: "c1".into(),
                    label: "org.example:left".into(),
                    version: Some("1.0".into()),
                    kind: CanvasNodeKind::Module,
                },
                CanvasNode {
                    id: "c2".into(),
                    component_id: "c2".into(),
                    label: "org.example:right".into(),
                    version: Some("2.0".into()),
                    kind: CanvasNodeKind::Module,
                },
            ],
            edges: vec![
                CanvasEdge {
                    id: "e1".into(),
                    from: "root".into(),
                    to: "c1".into(),
                    variant: CanvasEdgeVariant::Dependency,
                    label: "1.0".into(),
                },
                CanvasEdge {
                    id: "e2".into(),
                    from: "root".into(),
                    to: "c2".into(),
                    variant: CanvasEdgeVariant::Dependency,
                    label: "2.0".into(),
                },
            ],
            truncated: false,
            positions: Default::default(),
            bounds: Default::default(),
        };
        layout_canvas_graph(&mut graph);
        graph
    }

    fn init(cx: &mut gpui::App) {
        let settings = SettingsStore::test(cx);
        cx.set_global(settings);
        theme_settings::init(LoadThemes::JustBase, cx);
        editor::init(cx);
        crate::init(cx);
    }

    #[gpui::test]
    async fn diagram_item_opens_focuses_and_keeps_state(cx: &mut TestAppContext) {
        cx.update(|cx| init(cx));
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-diagram",
            serde_json::json!({"pom.xml": "<project/>"}),
        )
        .await;
        let project = project::Project::test(fs, ["/groovy-diagram".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let target = DependencyDiagramTarget {
            module_label: "app".into(),
            provider: DependencyDiagramProvider::Gradle,
            configuration: Some(":core".into()),
        };
        let derivation = DependencyDiagramDerivation {
            canvas: fixture_graph(),
            target: target.clone(),
            selected: Default::default(),
            scope: DependencyDiagramScope::GradleConfiguration(":core".into()),
            scope_options: vec![":core".into()],
        };
        let item = workspace.update_in(cx, |workspace, window, cx| {
            DependencyDiagramItem::open(derivation, None, workspace, window, cx)
        });
        let (text, retained_target) = cx.update(|_, cx| {
            (
                item.read(cx).tab_content_text(0, cx),
                item.read(cx).target().clone(),
            )
        });
        assert_eq!(text, "Dependencies — app (:core)");
        assert_eq!(retained_target, target);

        cx.update(|window, _| window.activate_a11y_for_test());
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
            let tree: serde_json::Value =
                serde_json::from_str(&window.debug_a11y_tree_json().expect("a11y tree"))
                    .expect("a11y json");
            let labels: Vec<String> = tree["nodes"]
                .as_object()
                .expect("nodes")
                .values()
                .filter_map(|node| node["aria"]["label"].as_str().map(|label| label.to_owned()))
                .collect();
            assert!(
                labels.iter().any(|label| label == "Dependency diagram"),
                "canvas exposes an accessible root: {labels:?}"
            );
            assert!(
                labels
                    .iter()
                    .any(|label| label.contains("org.example:left") && label.contains("1.0")),
                "nodes expose group, artifact and version: {labels:?}"
            );
        });

        // Retention: mutate the canvas selection, re-render, and verify the
        // item state persists in the pane.
        item.update(cx, |item, cx| {
            item.canvas.update(cx, |canvas, _| {
                canvas.select_node("c1", false, false);
            });
        });
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });
        let retained = cx.update(|_, cx| item.read(cx).canvas.read(cx).selection.contains("c1"));
        assert!(retained, "canvas selection persists across renders");
    }

    #[gpui::test]
    async fn dependency_diagram_toolbar_scopes_actions_and_reveal(cx: &mut TestAppContext) {
        cx.update(|cx| init(cx));
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-diagram",
            serde_json::json!({"pom.xml": "<project/>"}),
        )
        .await;
        let project = project::Project::test(fs, ["/groovy-diagram".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let derivation = DependencyDiagramDerivation {
            canvas: fixture_graph(),
            target: DependencyDiagramTarget {
                module_label: "app".into(),
                provider: DependencyDiagramProvider::Gradle,
                configuration: Some(":core".into()),
            },
            selected: Default::default(),
            scope: DependencyDiagramScope::GradleConfiguration(":core".into()),
            scope_options: vec![":core".into()],
        };
        let item = workspace.update_in(cx, |workspace, window, cx| {
            DependencyDiagramItem::open(derivation, None, workspace, window, cx)
        });

        // The scope picker reflects the configurations the provider reports.
        let (options, scope_label) = cx.update(|_, cx| {
            (
                item.read(cx).scope_options().to_vec(),
                item.read(cx).scope().label(),
            )
        });
        assert_eq!(options, [":core".to_owned()]);
        assert_eq!(scope_label, ":core", "the derivation's scope is retained");

        // Fit uses the painted canvas size; zoom moves within the bounds.
        item.update(cx, |item, cx| {
            item.canvas.update(cx, |canvas, _| {
                *canvas.canvas_size.lock().expect("canvas size") = sim_runtime::graph::GraphSize {
                    width: 800.0,
                    height: 600.0,
                };
            });
        });
        item.update(cx, |item, cx| item.fit_canvas(cx));
        let fitted_scale = cx.update(|_, cx| item.read(cx).canvas.read(cx).viewport.scale);
        assert!(fitted_scale > 0.0, "fit produces a usable scale");
        item.update(cx, |item, cx| item.zoom_canvas(1.25, cx));
        let zoomed_scale = cx.update(|_, cx| item.read(cx).canvas.read(cx).viewport.scale);
        assert!(zoomed_scale > fitted_scale, "zoom in grows the scale");

        // Structure toggle flips the overview panel.
        item.update(cx, |item, cx| item.toggle_structure(cx));
        assert!(
            cx.update(|_, cx| item.read(cx).structure_visible),
            "Structure overview becomes visible"
        );

        // Export Image writes a deterministic SVG distinct from the Mermaid
        // Copy Diagram Source payload.
        item.update(cx, |item, cx| item.export_diagram_image(cx));
        let notice = cx.update(|_, cx| item.read(cx).notice.clone());
        assert!(
            notice.as_ref().is_some_and(|notice| notice.contains("SVG")),
            "Export Image reports the exported SVG: {notice:?}"
        );
        let svg = crate::dependency_graph_canvas::canvas_graph_svg(&fixture_graph());
        assert!(svg.starts_with("<svg"));
        assert!(!svg.contains("flowchart"), "SVG is not the Mermaid payload");

        // Search lists admitted node labels; selecting one selects and
        // reveals it on the canvas.
        let matches = cx.update(|_, cx| item.read(cx).search_matches("", cx));
        assert_eq!(matches.len(), 3, "all admitted nodes are searchable");
        item.update(cx, |item, cx| {
            item.reveal_node_by_label("org.example:right", cx)
        });
        let revealed = cx.update(|_, cx| item.read(cx).canvas.read(cx).selection.contains("c2"));
        assert!(revealed, "search result is selected and revealed");
    }

    #[gpui::test]
    async fn dependency_diagram_search_matches_reveals_and_bounds(cx: &mut TestAppContext) {
        cx.update(|cx| init(cx));
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-diagram",
            serde_json::json!({"pom.xml": "<project/>"}),
        )
        .await;
        let project = project::Project::test(fs, ["/groovy-diagram".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        // More admitted nodes than the bounded result window.
        let graph = crate::dependency_graph_canvas::CanvasGraph {
            nodes: (0..24)
                .map(|index| CanvasNode {
                    id: format!("n{index}"),
                    component_id: format!("c{index}"),
                    label: if index == 7 {
                        "org.example:special".into()
                    } else {
                        format!("org.example:shared{index}")
                    },
                    version: Some("1.0".into()),
                    kind: CanvasNodeKind::Module,
                })
                .collect(),
            edges: Vec::new(),
            truncated: false,
            positions: (0..24)
                .map(|index| {
                    (
                        format!("n{index}"),
                        sim_runtime::graph::GraphPoint {
                            x: index as f32 * 100.0,
                            y: 0.0,
                        },
                    )
                })
                .collect(),
            bounds: sim_runtime::graph::GraphRect {
                origin: sim_runtime::graph::GraphPoint { x: 0.0, y: 0.0 },
                size: sim_runtime::graph::GraphSize {
                    width: 2400.0,
                    height: 44.0,
                },
            },
        };
        let derivation = DependencyDiagramDerivation {
            canvas: graph,
            target: DependencyDiagramTarget {
                module_label: "app".into(),
                provider: DependencyDiagramProvider::Gradle,
                configuration: Some(":core".into()),
            },
            selected: Default::default(),
            scope: DependencyDiagramScope::GradleConfiguration(":core".into()),
            scope_options: vec![":core".into()],
        };
        let item = workspace.update_in(cx, |workspace, window, cx| {
            DependencyDiagramItem::open(derivation, None, workspace, window, cx)
        });
        item.update(cx, |item, cx| {
            item.canvas.update(cx, |canvas, _| {
                *canvas.canvas_size.lock().expect("canvas size") = sim_runtime::graph::GraphSize {
                    width: 800.0,
                    height: 600.0,
                };
            });
        });

        // Bounded results: the popup lists at most the bounded window even
        // when more nodes are admitted.
        let all = cx.update(|_, cx| item.read(cx).search_matches("", cx));
        assert_eq!(all.len(), 16, "search results are bounded");
        // Matching: the query narrows to the admitted labels containing it.
        let matched = cx.update(|_, cx| item.read(cx).search_matches("special", cx));
        assert_eq!(matched, ["org.example:special".to_owned()]);
        // Empty: an unknown query yields no entries.
        let empty = cx.update(|_, cx| item.read(cx).search_matches("absent-dependency", cx));
        assert!(empty.is_empty(), "an unknown query yields no matches");
        // Reveal: selecting a match selects and centres the node on canvas.
        item.update(cx, |item, cx| {
            item.reveal_node_by_label("org.example:special", cx)
        });
        let (selected, offset) = cx.update(|_, cx| {
            let canvas = item.read(cx).canvas.read(cx);
            (canvas.selection.contains("n7"), canvas.viewport.offset)
        });
        assert!(selected, "the revealed node is selected");
        // The node sits at the canvas centre: offset = centre - position*scale.
        let expected_offset = {
            let scale = cx.update(|_, cx| item.read(cx).canvas.read(cx).viewport.scale);
            sim_runtime::graph::GraphPoint {
                x: 400.0 - 700.0 * scale,
                y: 300.0,
            }
        };
        assert_eq!(offset, expected_offset, "the reveal centres the node");
        // Selecting a label that is no longer admitted reports guidance
        // instead of failing.
        item.update(cx, |item, cx| {
            item.reveal_node_by_label("org.example:absent", cx)
        });
        let notice = cx.update(|_, cx| item.read(cx).notice.clone());
        assert!(
            notice
                .as_ref()
                .is_some_and(|notice| notice.contains("no longer part of the admitted graph")),
            "a stale search result reports guidance: {notice:?}"
        );
    }

    #[gpui::test]
    async fn dependency_diagram_structure_overview_syncs_and_highlights(cx: &mut TestAppContext) {
        cx.update(|cx| init(cx));
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-diagram",
            serde_json::json!({"pom.xml": "<project/>"}),
        )
        .await;
        let project = project::Project::test(fs, ["/groovy-diagram".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let derivation = DependencyDiagramDerivation {
            canvas: fixture_graph(),
            target: DependencyDiagramTarget {
                module_label: "app".into(),
                provider: DependencyDiagramProvider::Gradle,
                configuration: Some(":core".into()),
            },
            selected: Default::default(),
            scope: DependencyDiagramScope::GradleConfiguration(":core".into()),
            scope_options: vec![":core".into()],
        };
        let item = workspace.update_in(cx, |workspace, window, cx| {
            DependencyDiagramItem::open(derivation, None, workspace, window, cx)
        });

        // Toggle on demand from the toolbar entry point.
        item.update(cx, |item, cx| item.toggle_structure(cx));
        assert!(
            cx.update(|_, cx| item.read(cx).structure_visible),
            "the Structure overview becomes visible"
        );

        // Viewport sync: pointing inside the overview recentres the canvas
        // viewport on the pointed graph position. Force a layout pass so the
        // overview's paint-captured origin reflects the laid-out panel.
        cx.update(|window, cx| window.draw(cx).clear(cx));
        let overview_origin = cx.update(|_, cx| {
            *item
                .read(cx)
                .overview_origin
                .lock()
                .expect("overview origin")
        });
        let scale = cx.update(|_, cx| item.read(cx).overview_scale(cx));
        let (bounds_origin, viewport_scale, canvas_size) = cx.update(|_, cx| {
            let canvas = item.read(cx).canvas.read(cx);
            (
                canvas.graph.bounds.origin,
                canvas.viewport.scale,
                *canvas.canvas_size.lock().expect("canvas size"),
            )
        });
        assert!(scale > 0.0, "the fixture graph fits the overview");
        let pointer = sim_runtime::graph::GraphPoint {
            x: overview_origin.x + 60.0,
            y: overview_origin.y + 30.0,
        };
        item.update(cx, |item, cx| item.move_viewport_to_overview(pointer, cx));
        let offset = cx.update(|_, cx| item.read(cx).canvas.read(cx).viewport.offset);
        let expected = sim_runtime::graph::GraphPoint {
            x: canvas_size.width / 2.0 - (bounds_origin.x + 60.0 / scale) * viewport_scale,
            y: canvas_size.height / 2.0 - (bounds_origin.y + 30.0 / scale) * viewport_scale,
        };
        assert!(
            (offset.x - expected.x).abs() < 0.01 && (offset.y - expected.y).abs() < 0.01,
            "the overview recentres the viewport on the pointed graph position; offset {offset:?} expected {expected:?} origin {overview_origin:?} scale {scale} viewport scale {viewport_scale} canvas size {canvas_size:?}"
        );

        // Selection highlight: the selected block is announced in the
        // overview's accessibility tree.
        item.update(cx, |item, cx| {
            item.canvas
                .update(cx, |canvas, _| canvas.select_node("c1", false, false));
        });
        cx.update(|window, _| window.activate_a11y_for_test());
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
            let tree: serde_json::Value =
                serde_json::from_str(&window.debug_a11y_tree_json().expect("a11y tree"))
                    .expect("a11y json");
            let labels: Vec<String> = tree["nodes"]
                .as_object()
                .expect("nodes")
                .values()
                .filter_map(|node| node["aria"]["label"].as_str().map(|label| label.to_owned()))
                .collect();
            assert!(
                labels.iter().any(
                    |label| label.contains("Structure overview selected node org.example:left")
                ),
                "the selected block is highlighted and announced: {labels:?}"
            );
        });
    }

    #[gpui::test]
    async fn dependency_diagram_export_is_identical_and_separate_from_copy_source(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| init(cx));
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-diagram",
            serde_json::json!({"pom.xml": "<project/>"}),
        )
        .await;
        let project = project::Project::test(fs, ["/groovy-diagram".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let derivation = DependencyDiagramDerivation {
            canvas: fixture_graph(),
            target: DependencyDiagramTarget {
                module_label: "app".into(),
                provider: DependencyDiagramProvider::Gradle,
                configuration: Some(":core".into()),
            },
            selected: Default::default(),
            scope: DependencyDiagramScope::GradleConfiguration(":core".into()),
            scope_options: vec![":core".into()],
        };
        // No provider source: Copy Diagram Source must degrade to guidance
        // while Export Image still works from the canvas tuples.
        let item = workspace.update_in(cx, |workspace, window, cx| {
            DependencyDiagramItem::open(derivation, None, workspace, window, cx)
        });

        item.update(cx, |item, cx| item.export_diagram_image(cx));
        let first = cx.update(|_, cx| item.read(cx).notice.clone());
        assert!(
            first.as_ref().is_some_and(|notice| notice.contains("SVG")),
            "Export Image reports the exported SVG: {first:?}"
        );
        item.update(cx, |item, cx| item.export_diagram_image(cx));
        let second = cx.update(|_, cx| item.read(cx).notice.clone());
        assert_eq!(
            first, second,
            "Export Image is deterministic for the same canvas"
        );

        // The exported SVG is derived from the same node/edge tuples the
        // canvas paints: every admitted node label appears, and every edge is
        // stroked.
        let svg = crate::dependency_graph_canvas::canvas_graph_svg(&fixture_graph());
        let canvas = cx.update(|_, cx| item.read(cx).canvas.read(cx).graph.clone());
        for node in &canvas.nodes {
            assert!(
                svg.contains(&crate::dependency_graph_canvas::svg_escape(&node.label)),
                "the export contains node {}",
                node.label
            );
        }
        assert_eq!(
            svg.match_indices("<line ").count(),
            canvas.edges.len(),
            "the export strokes every admitted edge"
        );

        // Copy Diagram Source never emits the SVG payload: without a provider
        // source it degrades to guidance, and its payload space is the
        // Mermaid derivation.
        item.update(cx, |item, cx| item.copy_diagram_source(cx));
        let copy_notice = cx.update(|_, cx| item.read(cx).notice.clone());
        assert_ne!(
            first, copy_notice,
            "Copy Diagram Source is a distinct action"
        );
        assert!(
            copy_notice
                .as_ref()
                .is_some_and(|notice| notice.contains("No provider source is attached")),
            "Copy Diagram Source without a source reports guidance: {copy_notice:?}"
        );
        assert!(
            !svg.contains("flowchart"),
            "the export is not the Mermaid payload"
        );
    }
}
