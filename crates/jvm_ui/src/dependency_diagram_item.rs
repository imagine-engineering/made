//! The dependency diagram editor item: a dedicated tab hosting the read-only
//! graph canvas for one selected project or module. The item owns the canvas
//! viewport, selection and inspection overlay; admitted facts come from the
//! Dependencies panel's snapshot at open time and are re-derived without
//! provider queries.

use crate::dependencies_panel::{
    AddGraphSelectionToSet, ClearGraphInspection, ClearGraphSelectionSet,
    CopyDependencyDiagramSource, DependencyGraphReceipt, InspectGraphAllRootPaths,
    InspectGraphAllUsages, InspectGraphConflictOrigins, InspectGraphNeighbours,
    InspectGraphPathsBetween, InspectGraphRootPaths, InspectGraphSelectionSet, MavenGraphScope,
    PinGraphSelection, canvas_dependency_diagram_source, current_dependency_graph_receipt,
    dependency_diagram_source_export, derive_dependency_diagram,
};
use crate::dependency_diagram_menu::{
    DependencyDiagramContextTarget, DependencyDiagramMenuCommand, DependencyDiagramMenuSnapshot,
    dependency_diagram_menu_available, stage_dependency_diagram_target,
};
use crate::dependency_graph_canvas::{CanvasGraph, CanvasLayout, CanvasNode, CanvasState};
use crate::{FocusNextGroovyControl, FocusPreviousGroovyControl};
use gpui::{
    App, AppContext as _, Context, DismissEvent, EventEmitter, FocusHandle, Focusable, FontWeight,
    Render, SharedString, Task, Window, actions,
};
use picker::{Picker, PickerDelegate};
use sim_runtime::graph::{GraphPoint, GraphSize};
use std::{collections::BTreeSet, sync::Arc};
use workspace::Workspace;

const STRUCTURE_OVERVIEW_SIZE: GraphSize = GraphSize {
    width: 180.0,
    height: 120.0,
};

actions!(
    groovy_dependencies_panel,
    [
        ShowDependencyDiagramPopup,
        DismissDependencyDiagramPopup,
        FitDependencyDiagram,
        ZoomDependencyDiagramIn,
        ZoomDependencyDiagramOut,
        ExportDependencyDiagramImage,
        SaveDependencyDiagramImage,
        ToggleDependencyDiagramStructure,
        UseHorizontalDependencyLayout,
        UseVerticalDependencyLayout,
        ShowDependencyDiagramContextMenu,
        DismissDependencyDiagramMenu
    ]
);

/// Applies a scope choice from the diagram scope picker.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct SelectDependencyDiagramScope {
    pub label: String,
}

#[derive(Clone)]
pub(crate) struct DependencySearchEntry {
    pub id: String,
    pub label: String,
    search_label: String,
    search_coordinate: Option<String>,
}

impl DependencySearchEntry {
    pub(crate) fn new(id: String, label: String, coordinate: Option<String>) -> Self {
        let search_label = label.to_lowercase();
        let search_coordinate = coordinate
            .map(|coordinate| coordinate.to_lowercase())
            .filter(|coordinate| coordinate != &search_label);
        Self {
            id,
            label,
            search_label,
            search_coordinate,
        }
    }
}

pub(crate) struct DependencySearch {
    entries: Vec<DependencySearchEntry>,
    matches: Vec<usize>,
    selected_index: usize,
    reveal: Box<dyn Fn(&str, &mut App)>,
}

impl DependencySearch {
    pub(crate) fn new(
        entries: Vec<DependencySearchEntry>,
        reveal: impl Fn(&str, &mut App) + 'static,
    ) -> Self {
        let mut search = Self {
            entries,
            matches: Vec::new(),
            selected_index: 0,
            reveal: Box::new(reveal),
        };
        search.filter("");
        search
    }

    fn filter(&mut self, query: &str) {
        const MAX_MATCHES: usize = 16;
        let query = query.trim().to_lowercase();
        self.matches.clear();
        self.selected_index = 0;
        if query.is_empty() {
            self.matches.extend(0..self.entries.len().min(MAX_MATCHES));
            return;
        }

        let mut partial_matches = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            let coordinate = entry.search_coordinate.as_deref();
            if entry.search_label == query || coordinate == Some(query.as_str()) {
                self.matches.push(index);
                if self.matches.len() == MAX_MATCHES {
                    return;
                }
            } else if partial_matches.len() < MAX_MATCHES
                && (entry.search_label.contains(&query)
                    || coordinate.is_some_and(|coordinate| coordinate.contains(&query)))
            {
                partial_matches.push(index);
            }
        }
        self.matches.extend(
            partial_matches
                .into_iter()
                .take(MAX_MATCHES - self.matches.len()),
        );
    }

    fn entry(&self, index: usize) -> Option<&DependencySearchEntry> {
        self.entries.get(*self.matches.get(index)?)
    }

    pub(crate) fn popover(self, window: &mut Window, cx: &mut App) -> gpui::Entity<Picker<Self>> {
        cx.new(|cx| Picker::uniform_list(self, window, cx).popover())
    }
}

fn diagram_search_entries(
    graph: &CanvasGraph,
    receipt: Option<&DependencyGraphReceipt>,
) -> Vec<DependencySearchEntry> {
    let entry = |node: &CanvasNode| {
        DependencySearchEntry::new(
            node.id.clone(),
            match &node.version {
                Some(version) => format!("{}:{version}", node.label),
                None => node.label.clone(),
            },
            None,
        )
    };
    if graph.truncated
        && let Some(nodes) = receipt.and_then(DependencyGraphReceipt::canvas_nodes)
    {
        return nodes.iter().map(entry).collect();
    }
    graph.nodes.iter().map(entry).collect()
}

fn diagram_search_control(
    id: &'static str,
    label: &'static str,
    handle: ui::PopoverMenuHandle<Picker<DependencySearch>>,
    menu: impl Fn(&mut Window, &mut App) -> Option<gpui::Entity<Picker<DependencySearch>>> + 'static,
) -> ui::PopoverMenu<Picker<DependencySearch>> {
    use ui::prelude::*;

    ui::PopoverMenu::new(id)
        .with_handle(handle)
        .trigger(
            IconButton::new(
                "dependency-diagram-search-icon",
                ui::IconName::MagnifyingGlass,
            )
            .shape(ui::IconButtonShape::Square)
            .tab_index(0_isize)
            .aria_label(label)
            .tooltip(ui::Tooltip::text("Search dependencies")),
        )
        .menu(menu)
}

impl PickerDelegate for DependencySearch {
    type ListItem = ui::ListItem;

    fn name() -> &'static str {
        "groovy-dependency-search"
    }

    fn match_count(&self) -> usize {
        self.matches.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn set_selected_index(&mut self, index: usize, _: &mut Window, _: &mut Context<Picker<Self>>) {
        self.selected_index = index;
    }

    fn placeholder_text(&self, _: &mut Window, _: &mut App) -> Arc<str> {
        "Search dependencies".into()
    }

    fn no_matches_text(&self, _: &mut Window, _: &mut App) -> Option<SharedString> {
        Some(
            if self.entries.is_empty() {
                "No admitted dependencies"
            } else {
                "No matching dependencies"
            }
            .into(),
        )
    }

    fn update_matches(
        &mut self,
        query: String,
        _: &mut Window,
        _: &mut Context<Picker<Self>>,
    ) -> Task<()> {
        self.filter(&query);
        Task::ready(())
    }

    fn confirm(&mut self, _: bool, _: &mut Window, cx: &mut Context<Picker<Self>>) {
        if let Some(entry) = self.entry(self.selected_index) {
            (self.reveal)(&entry.id, cx);
        }
    }

    fn dismissed(&mut self, _: &mut Window, _: &mut Context<Picker<Self>>) {}

    fn render_match(
        &self,
        index: usize,
        selected: bool,
        _: &mut Window,
        _: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        use ui::prelude::*;

        let entry = self.entry(index)?;
        Some(
            ui::ListItem::new(index)
                .inset(true)
                .toggle_state(selected)
                .aria_role(gpui::Role::MenuItem)
                .aria_label(entry.label.clone())
                .when(selected, ui::ListItem::aria_active_descendant)
                .child(ui::Label::new(entry.label.clone())),
        )
    }
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

    fn maven_scope(&self) -> crate::dependencies_panel::MavenGraphScope {
        match self {
            Self::MavenScope(label) => {
                crate::dependencies_panel::maven_graph_scope_from_label(label).unwrap_or_default()
            }
            Self::GradleConfiguration(_) => Default::default(),
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
    /// The opening panel, for context-menu command routing through the
    /// panel's own action handlers and enablement logic.
    pub panel: gpui::WeakEntity<crate::dependencies_panel::DependenciesPanel>,
}

impl DependencyDiagramSource {
    fn project_neighbourhood(
        &self,
        id: &str,
        scope: &DependencyDiagramScope,
        cx: &App,
    ) -> Option<DependencyDiagramDerivation> {
        derive_dependency_diagram(self.project.read(cx), &self.selection, id, Some(scope), cx)
            .filter(|derivation| derivation.canvas.node(id).is_some())
    }
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
    receipt: Option<DependencyGraphReceipt>,
    _panel_subscription: Option<gpui::Subscription>,
    _canvas_subscription: gpui::Subscription,
    toolbar_target: DependencyDiagramContextTarget,
    search_handle: ui::PopoverMenuHandle<Picker<DependencySearch>>,
    scope: DependencyDiagramScope,
    scope_options: Vec<String>,
    structure_visible: bool,
    overview_drag: bool,
    overview_origin: std::sync::Arc<std::sync::Mutex<sim_runtime::graph::GraphPoint>>,
    context_state: std::sync::Arc<crate::dependency_diagram_menu::DependencyDiagramContextState>,
    context_handle: ui::RightClickMenuHandle<ui::ContextMenu>,
    save_image_task: Option<Task<()>>,
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
        cx: &mut Context<Self>,
    ) -> Self {
        let receipt = source.as_ref().and_then(|source| {
            current_dependency_graph_receipt(source.project.read(cx), &source.selection, cx)
        });
        let panel_subscription = source.as_ref().and_then(|source| {
            let panel = source.panel.upgrade()?;
            Some(cx.observe(&panel, |item, _, cx| item.refresh_receipt(cx)))
        });
        let mut canvas = CanvasState::new(derivation.canvas);
        canvas.selection = derivation.selected;
        canvas.surface_focus_handle = Some(cx.focus_handle().tab_stop(true));
        let context_state = std::sync::Arc::new(
            crate::dependency_diagram_menu::DependencyDiagramContextState::default(),
        );
        let context_handle = ui::RightClickMenuHandle::default();
        canvas.context_state = Some(context_state.clone());
        canvas.context_handle = Some(context_handle.clone());
        let canvas = cx.new(|_| canvas);
        let canvas_subscription = cx.observe(&canvas, |item, _, cx| {
            let selected = item.selected_context_target(cx);
            if selected != item.toolbar_target {
                item.toolbar_target = selected;
                cx.notify();
            }
        });
        Self {
            canvas,
            target: derivation.target,
            source,
            receipt,
            _panel_subscription: panel_subscription,
            _canvas_subscription: canvas_subscription,
            toolbar_target: DependencyDiagramContextTarget::Canvas,
            search_handle: ui::PopoverMenuHandle::default(),
            scope: derivation.scope,
            scope_options: derivation.scope_options,
            structure_visible: false,
            overview_drag: false,
            overview_origin: std::sync::Arc::new(std::sync::Mutex::new(
                sim_runtime::graph::GraphPoint::ZERO,
            )),
            context_state,
            context_handle,
            save_image_task: None,
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

    pub(crate) fn matches_target(&self, selection: &crate::GraphSelection) -> bool {
        self.source
            .as_ref()
            .is_some_and(|source| source.selection.target() == selection.target())
    }

    fn receipt_is_current(&self, cx: &App) -> bool {
        let Some(source) = &self.source else {
            return true;
        };
        self.receipt.is_some()
            && self.receipt
                == current_dependency_graph_receipt(source.project.read(cx), &source.selection, cx)
    }

    fn refresh_receipt(&mut self, cx: &mut Context<Self>) {
        let Some(source) = &mut self.source else {
            return;
        };
        let project = source.project.read(cx);
        let receipt = current_dependency_graph_receipt(project, &source.selection, cx);
        if receipt == self.receipt {
            return;
        }
        let derivation = receipt.as_ref().and_then(|receipt| {
            derive_dependency_diagram(
                project,
                &receipt.selection,
                &source.node,
                Some(&self.scope),
                cx,
            )
        });
        self.search_handle.hide(cx);
        self.context_handle.hide(cx);
        match (receipt, derivation) {
            (Some(receipt), Some(derivation)) => {
                source.selection = receipt.selection.clone();
                self.receipt = Some(receipt);
                self.canvas.update(cx, |canvas, _| {
                    canvas.clear_placeholder_selection();
                });
                self.replace_derivation(derivation, cx);
            }
            _ => {
                self.receipt = None;
                self.scope_options.clear();
                self.canvas.update(cx, |canvas, cx| {
                    canvas.replace_graph(Default::default());
                    cx.notify();
                });
                self.notice = Some("Dependency graph facts are unavailable. Refresh and collect graph facts in the Dependencies panel.".into());
            }
        }
        cx.notify();
    }

    fn replace_derivation(
        &mut self,
        derivation: DependencyDiagramDerivation,
        cx: &mut Context<Self>,
    ) {
        self.canvas.update(cx, |canvas, cx| {
            canvas.replace_graph(derivation.canvas);
            cx.notify();
        });
        self.target = derivation.target;
        self.scope = derivation.scope;
        self.scope_options = derivation.scope_options;
        self.notice = None;
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
            canvas.fit_to_canvas();
            cx.notify();
        });
    }

    fn apply_layout(
        &mut self,
        layout: crate::dependency_graph_canvas::CanvasLayout,
        cx: &mut Context<Self>,
    ) {
        self.canvas.update(cx, |canvas, cx| {
            if canvas.set_layout(layout) {
                cx.notify();
            }
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

    pub fn copy_diagram_source(&mut self, cx: &mut Context<Self>) {
        let Some(source) = self.source.as_ref().filter(|_| self.receipt_is_current(cx)) else {
            self.notice = Some(
                "Copy Diagram Source requires current provider facts. Refresh the diagram and retry."
                    .into(),
            );
            cx.notify();
            return;
        };
        let graph = &self.canvas.read(cx).graph;
        let (payload, description) = if graph.truncated
            || self.scope.maven_scope() != MavenGraphScope::All
            || graph.layout != CanvasLayout::LeftToRight
        {
            (
                canvas_dependency_diagram_source(graph),
                if graph.truncated {
                    "Copied Mermaid source for the current canvas. The provider graph contains additional facts."
                } else {
                    "Copied Mermaid source for the current diagram view."
                },
            )
        } else {
            (
                dependency_diagram_source_export(source.project.read(cx), &source.selection, cx)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "The current provider graph is unavailable for Copy Diagram Source."
                        )
                    }),
                "Copied Mermaid source for the admitted dependency graph.",
            )
        };
        match payload {
            Ok(source_text) => {
                cx.write_to_clipboard(gpui::ClipboardItem::new_string(source_text));
                self.notice = Some(description.into());
            }
            Err(error) => {
                self.notice = Some(error.to_string());
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
        match derive_dependency_diagram(project, &source.selection, &source.node, Some(&scope), cx)
        {
            Some(derivation) => self.replace_derivation(derivation, cx),
            None => {
                self.notice = Some(
                    "No admitted graph facts match the chosen scope; refresh provider graph facts, then retry".into(),
                );
            }
        }
        cx.notify();
    }

    fn search_entries(&self, cx: &App) -> Vec<DependencySearchEntry> {
        if !self.receipt_is_current(cx) {
            return Vec::new();
        }
        diagram_search_entries(&self.canvas.read(cx).graph, self.receipt.as_ref())
    }

    pub fn reveal_node_by_id(&mut self, id: &str, cx: &mut Context<Self>) {
        if !self.receipt_is_current(cx) {
            return;
        }
        let revealed = self.ensure_node_neighbourhood(id, cx)
            && self.canvas.update(cx, |canvas, cx| {
                let revealed = canvas.reveal_node(id);
                if revealed {
                    cx.notify();
                }
                revealed
            });
        self.notice = (!revealed).then(|| {
            "The dependency is no longer part of the admitted graph; refresh the diagram and retry".into()
        });
        cx.notify();
    }

    fn ensure_node_neighbourhood(&mut self, id: &str, cx: &mut Context<Self>) -> bool {
        let graph = &self.canvas.read(cx).graph;
        let node_present = graph.node(id).is_some();
        if self.source.is_some() && (graph.truncated || !node_present) {
            self.project_node_neighbourhood(id, cx)
        } else {
            node_present
        }
    }

    fn project_node_neighbourhood(&mut self, id: &str, cx: &mut Context<Self>) -> bool {
        let Some(source) = &mut self.source else {
            return false;
        };
        let Some(derivation) = source.project_neighbourhood(id, &self.scope, cx) else {
            return false;
        };
        // A synthetic Gradle placeholder's ordinal is local to this graph revision.
        if derivation
            .canvas
            .node(id)
            .is_some_and(|node| node.has_provider_identity())
        {
            source.node = id.to_owned();
        }
        self.replace_derivation(derivation, cx);
        true
    }

    /// Toggles the minimap-style Structure overview panel.
    pub fn toggle_structure(&mut self, cx: &mut Context<Self>) {
        self.structure_visible = !self.structure_visible;
        self.overview_drag = false;
        cx.notify();
    }

    /// Opens the context menu for the focused node or diagram surface.
    fn show_context_menu_for_focus(&mut self, window: &mut gpui::Window, cx: &mut Context<Self>) {
        if self.context_handle.is_deployed() {
            return;
        }
        let canvas = self.canvas.read(cx);
        let surface_focused = canvas
            .surface_focus_handle
            .as_ref()
            .is_some_and(|focus| focus.is_focused(window));
        let focused = canvas.focused_node(window);
        if focused.is_none() && !surface_focused && !self.focus_handle.is_focused(window) {
            cx.propagate();
            return;
        }
        let target = stage_dependency_diagram_target(focused, &canvas.selection);
        let origin = *canvas.canvas_origin.lock().expect("canvas origin");
        let screen = if let Some(position) = focused.and_then(|id| canvas.node_screen_position(id))
        {
            position
        } else {
            let size = *canvas.canvas_size.lock().expect("canvas size");
            sim_runtime::graph::GraphPoint {
                x: size.width / 2.0,
                y: size.height / 2.0,
            }
        };
        self.show_context_menu_at(target, origin.translated(screen), window, cx);
    }

    fn show_context_menu_at(
        &self,
        target: DependencyDiagramContextTarget,
        position: GraphPoint,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        if self.context_handle.is_deployed() {
            return;
        }
        self.context_state.stage(
            crate::dependency_diagram_menu::DependencyDiagramContextInvocation {
                target,
                screen_position: position,
            },
        );
        let handle = self.context_handle.clone();
        let position = gpui::point(gpui::px(position.x), gpui::px(position.y));
        window.defer(cx, move |window, cx| {
            handle.show_at(position, window, cx);
        });
    }

    fn dismiss_context_menu(&mut self, cx: &mut Context<Self>) {
        if !self.context_handle.is_deployed() {
            cx.propagate();
            return;
        }
        self.context_handle.hide(cx);
        cx.notify();
    }

    pub(crate) fn context_menu_snapshot(
        &self,
        target: &DependencyDiagramContextTarget,
        cx: &App,
    ) -> DependencyDiagramMenuSnapshot {
        let canvas = self.canvas.read(cx);
        let Some(source) = self.source.as_ref().filter(|_| self.receipt_is_current(cx)) else {
            return DependencyDiagramMenuSnapshot::default();
        };
        let Some(panel) = source.panel.upgrade() else {
            return DependencyDiagramMenuSnapshot::default();
        };
        let members = target.comparison_nodes();
        let (selection_set_len, has_matching_pin, can_add_to_set) = panel
            .read(cx)
            .diagram_comparison_availability(&source.selection, target.node(), members);
        let is_admitted = |id: &str| {
            canvas
                .graph
                .node(id)
                .is_some_and(|node| node.has_provider_identity())
        };
        let maven_scope = self.scope.maven_scope();
        DependencyDiagramMenuSnapshot {
            selection_set_len,
            has_inspection: !canvas.inspected.is_empty(),
            has_source: true,
            has_admitted_node: target.node().is_some_and(is_admitted),
            has_matching_pin,
            can_add_to_set: can_add_to_set && members.iter().all(|node| is_admitted(node)),
            has_conflict_origins: target.node().is_some_and(|node| {
                self.receipt
                    .as_ref()
                    .is_some_and(|receipt| receipt.has_conflict_origins(node, maven_scope))
            }),
        }
    }

    pub(crate) fn inspect_canvas_target(
        &mut self,
        command: DependencyDiagramMenuCommand,
        target: &DependencyDiagramContextTarget,
        cx: &mut Context<Self>,
    ) {
        use DependencyDiagramMenuCommand::*;

        if !dependency_diagram_menu_available(
            command,
            target,
            &self.context_menu_snapshot(target, cx),
        ) {
            return;
        }
        let Some(source) = self.source.as_ref() else {
            return;
        };
        let result = match command {
            ClearInspection => {
                self.canvas.update(cx, |canvas, cx| {
                    canvas.inspected.clear();
                    canvas.inspected_edges.clear();
                    cx.notify();
                });
                source
                    .panel
                    .update(cx, |panel, cx| {
                        panel.clear_diagram_inspection(&source.selection, cx)
                    })
                    .map(|_| None)
            }
            AllUsages => {
                self.open_analyzer_for_target(target, cx);
                return;
            }
            ClearSelectionSet => source.panel.update(cx, |panel, cx| {
                panel.clear_graph_selection_set(cx);
                Some("Dependency comparison set cleared.".into())
            }),
            _ => {
                let Some(node) = target.node() else {
                    return;
                };
                self.inspect_or_compare_node(command, target, node, cx)
            }
        };
        self.notice = match result {
            Ok(notice) => notice,
            Err(error) => Some(error.to_string()),
        };
        cx.notify();
    }

    fn inspect_or_compare_node(
        &self,
        command: DependencyDiagramMenuCommand,
        target: &DependencyDiagramContextTarget,
        node: &str,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<Option<String>> {
        let source = self
            .source
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("The diagram's Dependencies panel is unavailable."))?;
        match command {
            DependencyDiagramMenuCommand::PinSelection => {
                let pinned = source.panel.update(cx, |panel, cx| {
                    panel.pin_graph_node(&source.selection, node, cx)
                })??;
                return Ok(Some(
                    if pinned {
                        "Dependency pinned."
                    } else {
                        "Dependency pin cleared."
                    }
                    .into(),
                ));
            }
            DependencyDiagramMenuCommand::AddToSelectionSet => {
                source.panel.update(cx, |panel, cx| {
                    panel.add_graph_nodes_to_set(
                        &source.selection,
                        target.comparison_nodes().to_vec(),
                        cx,
                    )
                })??;
                return Ok(Some(
                    "Selected dependencies added to the comparison set.".into(),
                ));
            }
            _ => {}
        }
        let Some(mode) = command.inspection_mode() else {
            return Ok(None);
        };
        let inspection = source.panel.update(cx, |panel, cx| {
            panel.inspect_diagram_graph(&source.selection, node, mode, self.scope.maven_scope(), cx)
        })??;
        let edges = BTreeSet::from_iter(inspection.edges);
        let visible = self.canvas.update(cx, |canvas, cx| {
            let mut nodes = BTreeSet::from_iter(inspection.nodes);
            canvas.inspected_edges.clear();
            for edge in &canvas.graph.edges {
                if edges.contains(&edge.provider_index) {
                    nodes.insert(edge.from.clone());
                    nodes.insert(edge.to.clone());
                    canvas.inspected_edges.insert(edge.provider_index);
                }
            }
            canvas.inspected = nodes.into_iter().collect();
            cx.notify();
            canvas.inspected_edges.len()
        });
        Ok(Some(format!(
            "{}: {} {} of {} retained edges visible.",
            mode.label(),
            inspection.status,
            visible,
            edges.len()
        )))
    }

    fn selected_context_target(&self, cx: &App) -> DependencyDiagramContextTarget {
        let canvas = self.canvas.read(cx);
        let selected = canvas
            .focus
            .as_ref()
            .filter(|id| canvas.selection.contains(*id))
            .or_else(|| canvas.selection.iter().next());
        stage_dependency_diagram_target(selected.map(String::as_str), &canvas.selection)
    }

    fn inspect_canvas_focus(
        &mut self,
        command: DependencyDiagramMenuCommand,
        window: &Window,
        cx: &mut Context<Self>,
    ) {
        let canvas = self.canvas.read(cx);
        let focus = canvas
            .focused_node(window)
            .or(canvas.focus.as_deref())
            .or_else(|| canvas.selection.iter().next().map(String::as_str));
        let target = stage_dependency_diagram_target(focus, &canvas.selection);
        self.inspect_canvas_target(command, &target, cx);
    }

    fn open_analyzer_for_target(
        &mut self,
        target: &DependencyDiagramContextTarget,
        cx: &mut Context<Self>,
    ) {
        let Some(source) = self.source.clone() else {
            self.notice = Some(
                "The diagram's opening panel is unavailable; reopen the diagram from the Dependencies panel, then retry All usages"
                    .into(),
            );
            cx.notify();
            return;
        };
        let Some(node) = target.node() else {
            return;
        };
        let node = node.to_owned();
        let scope = self.scope.clone();
        let item = cx.weak_entity();
        // Opening a pane item reads the current one.
        cx.defer(move |cx| {
            let Some(item) = item.upgrade() else {
                return;
            };
            if !item.read(cx).receipt_is_current(cx) {
                return;
            }
            let result = source.panel.update(cx, |panel, cx| {
                panel.open_analyzer_for_diagram(&source.selection, &node, &scope, cx)
            });
            item.update(cx, |item, cx| {
                item.notice = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) | Err(error) => Some(error.to_string()),
                };
                cx.notify();
            });
        });
    }

    pub(crate) fn select_linked_node(&mut self, id: &str, cx: &mut Context<Self>) -> bool {
        self.refresh_receipt(cx);
        if !self.receipt_is_current(cx) || !self.ensure_node_neighbourhood(id, cx) {
            return false;
        }
        if let Some(source) = &mut self.source
            && self
                .canvas
                .read(cx)
                .graph
                .node(id)
                .is_some_and(|node| node.has_provider_identity())
        {
            source.node = id.to_owned();
        }
        self.canvas.update(cx, |canvas, cx| {
            let size = *canvas.canvas_size.lock().expect("canvas size");
            let node_size = canvas.graph.node_size();
            let scale = canvas.viewport.scale;
            let is_visible = canvas.node_screen_position(id).is_some_and(|position| {
                position.x >= 0.0
                    && position.y >= 0.0
                    && position.x + node_size.width * scale <= size.width
                    && position.y + node_size.height * scale <= size.height
            });
            if is_visible {
                canvas.select_node(id, false, false);
                canvas.focus = Some(id.to_owned());
            } else {
                canvas.reveal_node(id);
            }
            cx.notify();
        });
        true
    }

    /// Copies the current diagram image as SVG.
    pub fn export_diagram_image(&mut self, cx: &mut Context<Self>) {
        if !self.receipt_is_current(cx) {
            return;
        }
        let svg = crate::dependency_graph_canvas::canvas_graph_svg(&self.canvas.read(cx).graph);
        cx.write_to_clipboard(gpui::ClipboardItem::new_string(svg));
        self.notice = Some("Copied the diagram image as SVG.".into());
        cx.notify();
    }

    pub(crate) fn save_diagram_image(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.save_image_task.is_some() {
            return;
        }
        let Some(source) = &self.source else {
            return;
        };
        let project = source.project.read(cx);
        let graph = &self.canvas.read(cx).graph;
        let Some(receipt) = current_dependency_graph_receipt(project, &source.selection, cx)
            .filter(|receipt| self.receipt.as_ref() == Some(receipt) && !graph.nodes.is_empty())
        else {
            self.notice = Some(
                "Refresh and collect current dependency facts before exporting an image.".into(),
            );
            cx.notify();
            return;
        };

        let filesystem = project.fs().clone();
        let directory = project
            .worktree_for_id(source.selection.worktree_id(), cx)
            .and_then(|worktree| {
                worktree
                    .read(cx)
                    .as_local()
                    .map(|worktree| worktree.abs_path().to_path_buf())
            })
            .or_else(std::env::home_dir)
            .unwrap_or_default();
        let svg = crate::dependency_graph_canvas::canvas_graph_svg(graph);
        let prompt = cx.prompt_for_new_path(&directory, Some("dependency-diagram.svg"));
        self.notice = None;
        self.save_image_task = Some(cx.spawn_in(window, async move |item, cx| {
            let result = async {
                let Some(destination) = prompt.await?? else {
                    return Ok(None);
                };
                anyhow::ensure!(
                    destination.is_absolute()
                        && destination
                            .extension()
                            .and_then(std::ffi::OsStr::to_str)
                            .is_some_and(|extension| extension.eq_ignore_ascii_case("svg")),
                    "Choose an absolute path with an .svg extension."
                );
                anyhow::ensure!(
                    item.read_with(cx, |item, cx| {
                        item.receipt.as_ref() == Some(&receipt) && item.receipt_is_current(cx)
                    })?,
                    "The dependency graph changed. Refresh it and export the image again."
                );
                filesystem
                    .create_new_file_atomically(destination.clone(), svg.into_bytes().into())
                    .await?;
                Ok::<_, anyhow::Error>(Some(destination))
            }
            .await;
            if let Err(error) = item.update_in(cx, |item, _, cx| {
                item.save_image_task = None;
                item.notice = match result {
                    Ok(Some(destination)) => Some(format!(
                        "Saved the diagram image to {}.",
                        destination.display()
                    )),
                    Ok(None) => None,
                    Err(error) => Some(format!("Could not export the diagram image. {error}")),
                };
                cx.notify();
            }) {
                log::debug!("Dependency diagram closed during image export: {error:#}");
            }
        }));
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
        let search_item = cx.weak_entity();
        let admitted = self.receipt_is_current(cx);
        let structure_visible = self.structure_visible;
        let notice = self.notice.clone();
        let selection = self.selected_context_target(cx);
        let inspection_available = self.context_menu_snapshot(&selection, cx).has_admitted_node;

        let toolbar = div()
            .flex()
            .flex_wrap()
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
                    .min_w_0()
                    .truncate()
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
            .child(diagram_search_control(
                "dependency-diagram-search",
                "Search dependency diagram nodes",
                self.search_handle.clone(),
                move |window, cx| {
                    let item = search_item.upgrade()?;
                    let entries = item.read(cx).search_entries(cx);
                    let item = search_item.clone();
                    Some(
                        DependencySearch::new(entries, move |id, cx| {
                            if let Some(item) = item.upgrade() {
                                item.update(cx, |item, cx| item.reveal_node_by_id(id, cx));
                            }
                        })
                        .popover(window, cx),
                    )
                },
            ))
            .children(
                [
                    (
                        "dependency-diagram-neighbours",
                        ui::IconName::GitGraph,
                        "Inspect dependency neighbours",
                        DependencyDiagramMenuCommand::InspectNeighbours,
                    ),
                    (
                        "dependency-diagram-shortest-paths",
                        ui::IconName::GitBranch,
                        "Inspect shortest root paths",
                        DependencyDiagramMenuCommand::InspectRootPaths,
                    ),
                ]
                .into_iter()
                .map(|(id, icon, label, command)| {
                    IconButton::new(id, icon)
                        .shape(ui::IconButtonShape::Square)
                        .tab_index(0_isize)
                        .aria_label(label)
                        .tooltip(ui::Tooltip::text(label))
                        .disabled(!inspection_available)
                        .on_click(cx.listener(move |item, _, _, cx| {
                            let selected = item.selected_context_target(cx);
                            item.inspect_canvas_target(command, &selected, cx);
                        }))
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
                    .disabled(!admitted)
                    .tooltip(ui::Tooltip::text("Copy Diagram Source"))
                    .on_click(cx.listener(|item, _, _, cx| item.copy_diagram_source(cx))),
            )
            .child(
                IconButton::new("dependency-diagram-export-icon", ui::IconName::Image)
                    .shape(ui::IconButtonShape::Square)
                    .tab_index(0_isize)
                    .aria_label("Export dependency diagram image to file")
                    .disabled(!admitted || self.save_image_task.is_some())
                    .tooltip(ui::Tooltip::text("Export Image to File"))
                    .on_click(
                        cx.listener(|item, _, window, cx| item.save_diagram_image(window, cx)),
                    ),
            )
            .child(crate::DependenciesPanel::ordered_operations_button());

        let context_state = self.context_state.clone();
        let context_item = cx.weak_entity();
        let context_handle = self.context_handle.clone();
        let canvas_element = self.canvas.clone();
        let canvas_wrapper = div()
            .relative()
            .flex_1()
            .min_h_0()
            .child(
                ui::right_click_menu("dependency-diagram-context-menu")
                    .with_handle(context_handle)
                    .full_height(true)
                    // The canvas must resolve the pointer target before the menu is built.
                    .open_on_right_click(false)
                    .maybe_menu(move |window, cx| {
                        crate::dependency_diagram_menu::build_dependency_diagram_context_menu(
                            &context_state,
                            context_item.clone(),
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
            .key_context("DependencyDiagram")
            .track_focus(&self.focus_handle)
            .tab_group()
            .size_full()
            .flex()
            .flex_col()
            .bg(theme.background)
            .on_action(|_: &FocusNextGroovyControl, window, cx| window.focus_next(cx))
            .on_action(|_: &FocusPreviousGroovyControl, window, cx| window.focus_prev(cx))
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
                cx.listener(|item, _: &SaveDependencyDiagramImage, window, cx| {
                    item.save_diagram_image(window, cx)
                }),
            )
            .on_action(
                cx.listener(|item, _: &ToggleDependencyDiagramStructure, _, cx| {
                    item.toggle_structure(cx)
                }),
            )
            .on_action(
                cx.listener(|item, _: &UseHorizontalDependencyLayout, _, cx| {
                    item.apply_layout(
                        crate::dependency_graph_canvas::CanvasLayout::LeftToRight,
                        cx,
                    )
                }),
            )
            .on_action(cx.listener(|item, _: &UseVerticalDependencyLayout, _, cx| {
                item.apply_layout(
                    crate::dependency_graph_canvas::CanvasLayout::TopToBottom,
                    cx,
                )
            }))
            .on_action(
                cx.listener(|item, event: &SelectDependencyDiagramScope, _, cx| {
                    item.apply_scope(event.label.clone(), cx)
                }),
            )
            .on_action(cx.listener(|item, _: &InspectGraphNeighbours, window, cx| {
                item.inspect_canvas_focus(
                    DependencyDiagramMenuCommand::InspectNeighbours,
                    window,
                    cx,
                );
            }))
            .on_action(cx.listener(|item, _: &InspectGraphRootPaths, window, cx| {
                item.inspect_canvas_focus(
                    DependencyDiagramMenuCommand::InspectRootPaths,
                    window,
                    cx,
                );
            }))
            .on_action(
                cx.listener(|item, _: &InspectGraphAllRootPaths, window, cx| {
                    item.inspect_canvas_focus(
                        DependencyDiagramMenuCommand::InspectAllRootPaths,
                        window,
                        cx,
                    );
                }),
            )
            .on_action(
                cx.listener(|item, _: &InspectGraphConflictOrigins, window, cx| {
                    item.inspect_canvas_focus(
                        DependencyDiagramMenuCommand::InspectConflictOrigins,
                        window,
                        cx,
                    );
                }),
            )
            .on_action(cx.listener(|item, _: &InspectGraphAllUsages, window, cx| {
                item.inspect_canvas_focus(DependencyDiagramMenuCommand::AllUsages, window, cx);
            }))
            .on_action(
                cx.listener(|item, _: &InspectGraphPathsBetween, window, cx| {
                    item.inspect_canvas_focus(
                        DependencyDiagramMenuCommand::PathsBetween,
                        window,
                        cx,
                    );
                }),
            )
            .on_action(cx.listener(|item, _: &PinGraphSelection, window, cx| {
                item.inspect_canvas_focus(DependencyDiagramMenuCommand::PinSelection, window, cx);
            }))
            .on_action(cx.listener(|item, _: &AddGraphSelectionToSet, window, cx| {
                item.inspect_canvas_focus(
                    DependencyDiagramMenuCommand::AddToSelectionSet,
                    window,
                    cx,
                );
            }))
            .on_action(
                cx.listener(|item, _: &InspectGraphSelectionSet, window, cx| {
                    item.inspect_canvas_focus(
                        DependencyDiagramMenuCommand::InspectSelectionSet,
                        window,
                        cx,
                    );
                }),
            )
            .on_action(cx.listener(|item, _: &ClearGraphSelectionSet, window, cx| {
                item.inspect_canvas_focus(
                    DependencyDiagramMenuCommand::ClearSelectionSet,
                    window,
                    cx,
                );
            }))
            .on_action(cx.listener(|item, _: &ClearGraphInspection, _, cx| {
                item.inspect_canvas_target(
                    DependencyDiagramMenuCommand::ClearInspection,
                    &DependencyDiagramContextTarget::Canvas,
                    cx,
                );
            }))
            .on_action(
                cx.listener(|item, _: &ShowDependencyDiagramContextMenu, window, cx| {
                    item.show_context_menu_for_focus(window, cx)
                }),
            )
            .on_action(
                cx.listener(|item, _: &DismissDependencyDiagramMenu, _, cx| {
                    item.dismiss_context_menu(cx)
                }),
            )
            .child(toolbar)
            .child(canvas_wrapper)
            .when_some(notice, |root, notice_text| {
                root.child(
                    div()
                        .id("dependency-diagram-notice")
                        .role(gpui::Role::Status)
                        .aria_live(gpui::accesskit::Live::Polite)
                        .aria_value(notice_text.clone())
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

        let theme = cx.theme().colors().clone();
        let graph = &self.canvas.read(cx).graph;
        let bounds = graph.bounds;
        let scale = self.overview_scale(cx);
        let node_size = graph.node_size();
        let width = 3.0_f32.max(node_size.width * scale);
        let height = 2.0_f32.max(node_size.height * scale);
        let overview_origin = self.overview_origin.clone();
        let mut overview = div()
            .absolute()
            .right_2()
            .bottom_2()
            .w(px(STRUCTURE_OVERVIEW_SIZE.width))
            .h(px(STRUCTURE_OVERVIEW_SIZE.height))
            .border_1()
            .border_color(theme.border)
            .bg(theme.background.opacity(0.9))
            .overflow_hidden()
            .block_mouse_except_scroll()
            .on_mouse_down(
                gpui::MouseButton::Right,
                cx.listener(|item, event: &gpui::MouseDownEvent, window, cx| {
                    item.show_overview_pointer_menu(None, event, window, cx);
                }),
            )
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
                if item.overview_drag && event.pressed_button == Some(gpui::MouseButton::Left) {
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
                cx.listener(Self::finish_overview_drag),
            )
            .on_mouse_up_out(
                gpui::MouseButton::Left,
                cx.listener(Self::finish_overview_drag),
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
                let position = GraphPoint {
                    x: (origin.x - bounds.origin.x) * scale,
                    y: (origin.y - bounds.origin.y) * scale,
                };
                let pointer_node = node.id.clone();
                let rect = div()
                    .absolute()
                    .left(px(position.x))
                    .top(px(position.y))
                    .w(px(width))
                    .h(px(height))
                    .on_mouse_down(
                        gpui::MouseButton::Right,
                        cx.listener(move |item, event: &gpui::MouseDownEvent, window, cx| {
                            item.show_overview_pointer_menu(Some(&pointer_node), event, window, cx);
                        }),
                    );
                if selected {
                    let node_id = node.id.clone();
                    let menu_node = node_id.clone();
                    let menu_anchor = position.translated(GraphPoint {
                        x: width / 2.0,
                        y: height / 2.0,
                    });
                    let focus_color = theme.text;
                    overview = overview.child(
                        rect.id(format!("structure-overview-{}", node.id))
                            .bg(theme.border_selected)
                            .tab_index(0_isize)
                            .focus_visible(move |style| style.border_1().border_color(focus_color))
                            .role(gpui::Role::Button)
                            .aria_label(format!(
                                "Structure overview selected node {}; activate to centre the viewport",
                                node.label
                            ))
                            .on_action(cx.listener(
                                move |item, _: &ShowDependencyDiagramContextMenu, window, cx| {
                                    item.show_overview_context_menu(&menu_node, menu_anchor, window, cx);
                                },
                            ))
                            .on_click(cx.listener(move |item, _, _, cx| {
                                item.center_overview_node(&node_id, cx);
                            })),
                    );
                } else {
                    overview = overview.child(rect.bg(theme.border));
                }
            }
        }
        overview
    }

    fn show_overview_pointer_menu(
        &self,
        node: Option<&str>,
        event: &gpui::MouseDownEvent,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        cx.stop_propagation();
        if !self.receipt_is_current(cx) {
            return;
        }
        let target = stage_dependency_diagram_target(node, &self.canvas.read(cx).selection);
        let position = GraphPoint {
            x: event.position.x.into(),
            y: event.position.y.into(),
        };
        self.show_context_menu_at(target, position, window, cx);
    }

    fn show_overview_context_menu(
        &self,
        id: &str,
        anchor: GraphPoint,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        let canvas = self.canvas.read(cx);
        if !canvas.selection.contains(id) || !self.receipt_is_current(cx) {
            return;
        }
        let target = stage_dependency_diagram_target(Some(id), &canvas.selection);
        let position = match self.overview_origin.lock() {
            Ok(origin) => origin.translated(anchor),
            Err(error) => {
                log::error!("Could not read the dependency diagram overview position: {error}");
                return;
            }
        };
        self.show_context_menu_at(target, position, window, cx);
    }

    fn finish_overview_drag(
        &mut self,
        _: &gpui::MouseUpEvent,
        _: &mut gpui::Window,
        _: &mut Context<Self>,
    ) {
        self.overview_drag = false;
    }

    /// Recentres the canvas viewport on the graph position the pointer points
    /// at inside the Structure overview.
    fn move_viewport_to_overview(
        &mut self,
        position: sim_runtime::graph::GraphPoint,
        cx: &mut Context<Self>,
    ) {
        let overview_origin = *self.overview_origin.lock().expect("overview origin");
        let bounds = self.canvas.read(cx).graph.bounds;
        let scale = self.overview_scale(cx);
        if scale <= 0.0 {
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
            canvas.center_viewport_on(graph_position);
            cx.notify();
        });
    }

    fn center_overview_node(&mut self, id: &str, cx: &mut Context<Self>) {
        if !self.receipt_is_current(cx) {
            return;
        }
        self.canvas.update(cx, |canvas, cx| {
            if canvas.selection.contains(id)
                && let Some(origin) = canvas.graph.positions.get(id)
            {
                let size = canvas.graph.node_size();
                canvas.center_viewport_on(GraphPoint {
                    x: origin.x + size.width / 2.0,
                    y: origin.y + size.height / 2.0,
                });
                cx.notify();
            }
        });
    }

    fn overview_scale(&self, cx: &App) -> f32 {
        let bounds = self.canvas.read(cx).graph.bounds;
        if bounds.size.width > 0.0 && bounds.size.height > 0.0 {
            (STRUCTURE_OVERVIEW_SIZE.width / bounds.size.width)
                .min(STRUCTURE_OVERVIEW_SIZE.height / bounds.size.height)
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
    admitted: Option<(DependencyDiagramSource, DependencyGraphReceipt)>,
    scope: DependencyDiagramScope,
    focus_handle: FocusHandle,
    search_handle: ui::PopoverMenuHandle<Picker<DependencySearch>>,
    _panel_subscription: Option<gpui::Subscription>,
}

impl DependencyDiagramPopup {
    pub fn new(open: DependencyDiagramOpen, cx: &mut Context<Self>) -> Self {
        let DependencyDiagramOpen { derivation, source } = open;
        let receipt =
            current_dependency_graph_receipt(source.project.read(cx), &source.selection, cx);
        let panel_subscription = source.panel.upgrade().map(|panel| {
            cx.observe(&panel, |popup, _, cx| {
                if popup.current_source(cx).is_none() {
                    popup.search_handle.hide(cx);
                    popup.canvas.update(cx, |canvas, cx| {
                        canvas.replace_graph(Default::default());
                        cx.notify();
                    });
                    cx.emit(DismissEvent);
                }
            })
        });
        let mut canvas = CanvasState::new(derivation.canvas);
        canvas.selection = derivation.selected;
        Self {
            canvas: cx.new(|_| canvas),
            target: derivation.target,
            admitted: receipt.map(|receipt| (source, receipt)),
            scope: derivation.scope,
            focus_handle: cx.focus_handle(),
            search_handle: ui::PopoverMenuHandle::default(),
            _panel_subscription: panel_subscription,
        }
    }

    pub fn canvas(&self) -> &gpui::Entity<CanvasState> {
        &self.canvas
    }

    pub fn target(&self) -> &DependencyDiagramTarget {
        &self.target
    }

    fn current_source(
        &self,
        cx: &App,
    ) -> Option<(&DependencyDiagramSource, &DependencyGraphReceipt)> {
        let (source, receipt) = self.admitted.as_ref()?;
        let current =
            current_dependency_graph_receipt(source.project.read(cx), &source.selection, cx)?;
        (current == *receipt).then_some((source, receipt))
    }

    fn reveal_node(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some((source, _)) = self.current_source(cx) else {
            return;
        };
        let canvas = self.canvas.read(cx);
        let projection = if canvas.graph.truncated {
            let Some(derivation) = source.project_neighbourhood(id, &self.scope, cx) else {
                return;
            };
            Some(derivation.canvas)
        } else {
            None
        };
        self.canvas.update(cx, |canvas, cx| {
            if let Some(graph) = projection {
                canvas.replace_graph(graph);
            }
            if canvas.reveal_node(id) {
                cx.notify();
            }
        });
    }

    fn fit_canvas(&self, cx: &mut Context<Self>) {
        self.canvas.update(cx, |canvas, cx| {
            canvas.fit_to_canvas();
            cx.notify();
        });
    }

    fn move_focus(&self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        let previous = window.focused(cx);
        if forward {
            window.focus_next(cx);
        } else {
            window.focus_prev(cx);
        }
        if !self.focus_handle.contains_focused(window, cx)
            && let Some(previous) = previous
        {
            window.focus(&previous, cx);
        }
    }
}

impl Render for DependencyDiagramPopup {
    fn render(
        &mut self,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) -> impl gpui::IntoElement {
        use ui::prelude::*;

        let theme = cx.theme().colors().clone();
        let title = self.target.title();
        let search_popup = cx.weak_entity();
        let viewport = window.viewport_size();
        let margin = window.rem_size();
        let width = (viewport.width - margin * 2.).max(px(0.)).min(px(960.));
        // The shared modal layer offsets its content by five rem from the top.
        let height = (viewport.height - margin * 6.).max(px(0.)).min(px(640.));
        div()
            .key_context("DependencyDiagramPopup")
            .track_focus(&self.focus_handle)
            .tab_group()
            .on_action(
                cx.listener(|popup, _: &FocusNextGroovyControl, window, cx| {
                    popup.move_focus(true, window, cx);
                }),
            )
            .on_action(
                cx.listener(|popup, _: &FocusPreviousGroovyControl, window, cx| {
                    popup.move_focus(false, window, cx);
                }),
            )
            .on_action(
                cx.listener(|popup, _: &DismissDependencyDiagramPopup, _, cx| {
                    if popup.search_handle.is_deployed() {
                        popup.search_handle.hide(cx);
                    } else {
                        cx.emit(DismissEvent);
                    }
                }),
            )
            .on_action(cx.listener(|popup, _: &FitDependencyDiagram, _, cx| popup.fit_canvas(cx)))
            .w(width)
            .h(height)
            .flex()
            .flex_col()
            .bg(theme.background)
            .border_1()
            .border_color(theme.border)
            .rounded_md()
            .overflow_hidden()
            .child(
                h_flex()
                    .gap_1()
                    .px_2()
                    .py_1()
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text)
                    .child(div().flex_1().min_w_0().truncate().child(title))
                    .child(diagram_search_control(
                        "dependency-preview-search",
                        "Search dependency preview nodes",
                        self.search_handle.clone(),
                        move |window, cx| {
                            let popup = search_popup.upgrade()?;
                            let popup = popup.read(cx);
                            let (_, receipt) = popup.current_source(cx)?;
                            let entries =
                                diagram_search_entries(&popup.canvas.read(cx).graph, Some(receipt));
                            let popup = search_popup.clone();
                            Some(
                                DependencySearch::new(entries, move |id, cx| {
                                    if let Some(popup) = popup.upgrade() {
                                        popup.update(cx, |popup, cx| popup.reveal_node(id, cx));
                                    }
                                })
                                .popover(window, cx),
                            )
                        },
                    ))
                    .child(
                        ui::IconButton::new("dependency-popup-fit", ui::IconName::Maximize)
                            .shape(ui::IconButtonShape::Square)
                            .tab_index(0_isize)
                            .aria_label("Fit dependency diagram")
                            .tooltip(ui::Tooltip::text("Fit diagram"))
                            .on_click(cx.listener(|popup, _, _, cx| popup.fit_canvas(cx))),
                    ),
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
    use gpui::{InputEvent as _, TestAppContext, px};
    use settings::SettingsStore;
    use theme::LoadThemes;
    use workspace::Item as _;

    #[test]
    fn dependency_search_keeps_exact_coordinates_ahead_of_partial_matches() {
        let coordinate = "org.example:core:1.0";
        let mut entries = (0..20)
            .map(|index| {
                DependencySearchEntry::new(
                    format!("partial-{index}"),
                    format!("{coordinate}-preview-{index}"),
                    None,
                )
            })
            .collect::<Vec<_>>();
        entries.extend(
            ["exact-first", "exact-second"]
                .map(|id| DependencySearchEntry::new(id.into(), coordinate.into(), None)),
        );
        let mut search = DependencySearch::new(entries, |_, _| {});

        search.filter(" ORG.EXAMPLE:CORE:1.0 ");
        assert_eq!(search.match_count(), 16);
        assert_eq!(
            search.entry(0).expect("first exact result").id,
            "exact-first"
        );
        assert_eq!(
            search.entry(1).expect("second exact result").id,
            "exact-second"
        );
        assert_eq!(
            search.entry(2).expect("first partial result").id,
            "partial-0"
        );
    }

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
                    provider_index: 0,
                    from: "root".into(),
                    to: "c1".into(),
                    variant: CanvasEdgeVariant::Dependency,
                    label: "1.0".into(),
                },
                CanvasEdge {
                    id: "e2".into(),
                    provider_index: 1,
                    from: "root".into(),
                    to: "c2".into(),
                    variant: CanvasEdgeVariant::Dependency,
                    label: "2.0".into(),
                },
            ],
            truncated: false,
            positions: Default::default(),
            bounds: Default::default(),
            layout: Default::default(),
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
    async fn dependency_diagram_popup_fits_a_narrow_workspace(cx: &mut TestAppContext) {
        cx.update(init);
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-diagram-popup",
            serde_json::json!({"pom.xml": "<project/>"}),
        )
        .await;
        let project = project::Project::test(fs, ["/groovy-diagram-popup".as_ref()], cx).await;
        let window = cx.open_window(gpui::size(gpui::px(480.), gpui::px(360.)), |window, cx| {
            workspace::MultiWorkspace::test_new(project, window, cx)
        });
        let container = window.root(cx).expect("mounted workspace");
        let cx = &mut gpui::VisualTestContext::from_window(*window, cx);
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.toggle_modal(window, cx, |_, cx| DependencyDiagramPopup {
                canvas: cx.new(|_| CanvasState::new(fixture_graph())),
                target: DependencyDiagramTarget {
                    module_label: "app".into(),
                    provider: DependencyDiagramProvider::Maven,
                    configuration: None,
                },
                admitted: None,
                scope: DependencyDiagramScope::MavenScope("All scopes".into()),
                focus_handle: cx.focus_handle(),
                search_handle: ui::PopoverMenuHandle::default(),
                _panel_subscription: None,
            });
        });
        cx.run_until_parked();
        let canvas = cx.update(|window, cx| {
            window.activate_a11y_for_test();
            window.draw(cx).clear(cx);
            let popup = workspace
                .read(cx)
                .active_modal::<DependencyDiagramPopup>(cx)
                .expect("diagram popup");
            let canvas = popup.read(cx).canvas().clone();
            let canvas_state = canvas.read(cx);
            let origin = *canvas_state.canvas_origin.lock().expect("painted origin");
            let size = *canvas_state.canvas_size.lock().expect("painted size");
            let viewport = window.viewport_size();
            assert!(size.width > 0. && size.height > 0.);
            assert!(origin.x >= 0. && origin.x + size.width <= f32::from(viewport.width));
            assert!(origin.y >= 0. && origin.y + size.height <= f32::from(viewport.height));
            canvas
        });
        let focused_label = |cx: &mut gpui::VisualTestContext| {
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let tree: serde_json::Value =
                    serde_json::from_str(&window.debug_a11y_tree_json().expect("a11y tree"))
                        .expect("a11y json");
                let focus = tree["gpui_focus"].as_str()?;
                tree["nodes"][focus]["aria"]["label"]
                    .as_str()
                    .map(str::to_owned)
            })
        };
        cx.simulate_keystrokes("tab");
        assert_eq!(
            focused_label(cx).as_deref(),
            Some("Search dependency preview nodes")
        );
        cx.simulate_keystrokes("tab");
        assert_eq!(focused_label(cx).as_deref(), Some("Fit dependency diagram"));
        cx.simulate_keystrokes("tab");
        assert_eq!(
            focused_label(cx).as_deref(),
            Some("Dependency org.example:app")
        );
        let (scale, selection, offscreen_offset) = canvas.update(cx, |canvas, cx| {
            canvas.select_node("c2", false, false);
            let size = *canvas.canvas_size.lock().expect("painted size");
            let left = canvas.node_screen_position("c1").expect("left dependency");
            canvas.pan(GraphPoint {
                x: size.width + 16.0 - left.x,
                y: 0.0,
            });
            cx.notify();
            (
                canvas.viewport.scale,
                canvas.selection.clone(),
                canvas.viewport.offset,
            )
        });
        cx.simulate_keystrokes("tab");
        assert_eq!(
            focused_label(cx).as_deref(),
            Some("Dependency org.example:left 1.0")
        );
        let revealed_offset = canvas.read_with(cx, |canvas, _| {
            let size = *canvas.canvas_size.lock().expect("painted size");
            let left = canvas.node_screen_position("c1").expect("left dependency");
            let node_size = canvas.graph.node_size();
            assert!(left.x >= 0.0 && left.x + node_size.width * scale <= size.width + 0.01);
            assert!(left.y >= 0.0 && left.y + node_size.height * scale <= size.height + 0.01);
            assert_eq!(canvas.viewport.scale, scale);
            assert_eq!(canvas.selection, selection);
            assert!(canvas.viewport.offset.x < offscreen_offset.x);
            canvas.viewport.offset
        });
        let manual_offset = revealed_offset.translated(GraphPoint { x: 40.0, y: 0.0 });
        canvas.update(cx, |canvas, cx| {
            canvas.pan(GraphPoint { x: 40.0, y: 0.0 });
            cx.notify();
        });
        assert_eq!(
            focused_label(cx).as_deref(),
            Some("Dependency org.example:left 1.0")
        );
        canvas.read_with(cx, |canvas, _| {
            assert_eq!(canvas.viewport.offset, manual_offset)
        });
        cx.simulate_keystrokes("shift-tab");
        assert_eq!(
            focused_label(cx).as_deref(),
            Some("Dependency org.example:app")
        );
        cx.simulate_keystrokes("shift-tab");
        assert_eq!(focused_label(cx).as_deref(), Some("Fit dependency diagram"));
        canvas.update(cx, |canvas, cx| {
            canvas.viewport.scale = 8.;
            cx.notify();
        });
        cx.dispatch_action(FitDependencyDiagram);
        canvas.read_with(cx, |canvas, _| assert!(canvas.viewport.scale < 8.));
        cx.simulate_keystrokes("escape");
        workspace.read_with(cx, |workspace, cx| {
            assert!(
                workspace
                    .active_modal::<DependencyDiagramPopup>(cx)
                    .is_none()
            );
        });
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
                    .any(|label| label == "Inspect dependency neighbours")
            );
            assert!(
                labels
                    .iter()
                    .any(|label| label == "Inspect shortest root paths")
            );
            assert!(
                labels
                    .iter()
                    .any(|label| label.contains("org.example:left") && label.contains("1.0")),
                "nodes expose group, artifact and version: {labels:?}"
            );
        });

        cx.update(|window, cx| {
            let focus = item
                .read(cx)
                .canvas
                .read(cx)
                .focus_handle_for_node("c2")
                .expect("keyboard node focus");
            window.focus(&focus, cx);
            window.draw(cx).clear(cx);
            let tree: serde_json::Value = serde_json::from_str(
                &window
                    .debug_a11y_tree_json()
                    .expect("focused dependency details"),
            )
            .expect("a11y json");
            assert!(
                tree["nodes"]
                    .as_object()
                    .expect("nodes")
                    .values()
                    .any(|node| {
                        node["aria"]["label"] == "Dependency details"
                            && node["aria"]["value"] == "org.example:right 2.0"
                    })
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
        let matches = cx.update(|_, cx| item.read(cx).search_entries(cx));
        assert_eq!(matches.len(), 3, "all admitted nodes are searchable");
        item.update(cx, |item, cx| item.reveal_node_by_id("c2", cx));
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
                    label: if index >= 22 {
                        "org.example:special".into()
                    } else {
                        format!("org.example:shared{index}")
                    },
                    version: Some(format!("{index}.0")),
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
            layout: Default::default(),
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

        let entries = cx.update(|_, cx| item.read(cx).search_entries(cx));
        let mut search = DependencySearch::new(entries, |_, _| {});
        assert_eq!(search.match_count(), 16);
        search.filter("absent-dependency");
        assert_eq!(search.match_count(), 0);
        search.filter(" SPECIAL ");
        assert_eq!(search.match_count(), 2);
        let result = search.entry(1).expect("second occurrence");
        assert_eq!(result.id, "n23");
        assert_eq!(result.label, "org.example:special:23.0");
        item.update(cx, |item, cx| item.reveal_node_by_id(&result.id, cx));
        let (selected, offset) = cx.update(|_, cx| {
            let canvas = item.read(cx).canvas.read(cx);
            (canvas.selection.contains("n23"), canvas.viewport.offset)
        });
        assert!(selected, "the revealed node is selected");
        // The node sits at the canvas centre: offset = centre - position*scale.
        let expected_offset = {
            let scale = cx.update(|_, cx| item.read(cx).canvas.read(cx).viewport.scale);
            sim_runtime::graph::GraphPoint {
                x: 400.0 - 2300.0 * scale,
                y: 300.0,
            }
        };
        assert_eq!(offset, expected_offset, "the reveal centres the node");
        // Selecting a label that is no longer admitted reports guidance
        // instead of failing.
        item.update(cx, |item, cx| item.reveal_node_by_id("absent", cx));
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
        assert!(
            canvas_size.width > 0.0 && canvas_size.height > 0.0,
            "the diagram fills the editor body: {canvas_size:?}"
        );
        let pointer = GraphPoint {
            x: overview_origin.x + 60.0,
            y: overview_origin.y + 30.0,
        };
        let press = gpui::point(px(pointer.x), px(pointer.y));
        cx.simulate_mouse_down(press, gpui::MouseButton::Left, Default::default());
        let offset = cx.update(|_, cx| item.read(cx).canvas.read(cx).viewport.offset);
        let expected = sim_runtime::graph::GraphPoint {
            x: canvas_size.width / 2.0 - (bounds_origin.x + 60.0 / scale) * viewport_scale,
            y: canvas_size.height / 2.0 - (bounds_origin.y + 30.0 / scale) * viewport_scale,
        };
        assert!(
            (offset.x - expected.x).abs() < 0.01 && (offset.y - expected.y).abs() < 0.01,
            "the overview recentres the viewport on the pointed graph position; offset {offset:?} expected {expected:?} origin {overview_origin:?} scale {scale} viewport scale {viewport_scale} canvas size {canvas_size:?}"
        );
        let drag = gpui::point(press.x + px(30.0), press.y);
        cx.simulate_mouse_move(drag, gpui::MouseButton::Left, Default::default());
        let dragged_offset = cx.update(|_, cx| item.read(cx).canvas.read(cx).viewport.offset);
        assert!((dragged_offset.x - (expected.x - 30.0 / scale * viewport_scale)).abs() < 0.01);
        assert!((dragged_offset.y - expected.y).abs() < 0.01);
        let outside = gpui::point(drag.x, px(overview_origin.y - 10.0));
        cx.simulate_mouse_move(outside, gpui::MouseButton::Left, Default::default());
        cx.simulate_mouse_up(outside, gpui::MouseButton::Left, Default::default());
        cx.simulate_mouse_move(press, None, Default::default());
        cx.update(|_, cx| {
            let item = item.read(cx);
            assert_eq!(item.canvas.read(cx).viewport.offset, dragged_offset);
            assert!(!item.overview_drag);
        });

        // Selection highlight: the selected block is announced in the
        // overview's accessibility tree.
        item.update(cx, |item, cx| {
            item.canvas.update(cx, |canvas, cx| {
                canvas.select_node("c1", false, false);
                canvas.select_node("c2", true, false);
                cx.notify();
            });
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

        let label =
            "Structure overview selected node org.example:left; activate to centre the viewport";
        let overview_block_has_focus = |cx: &mut gpui::VisualTestContext, label: &str| {
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let tree: serde_json::Value =
                    serde_json::from_str(&window.debug_a11y_tree_json().expect("a11y tree"))
                        .expect("a11y json");
                tree["gpui_focus"].as_str().is_some_and(|id| {
                    tree["nodes"][id]["aria"]["label"] == label
                        && tree["nodes"][id]["aria"]["role"] == "Button"
                })
            })
        };
        cx.update(|window, cx| {
            let focus = item.read(cx).focus_handle.clone();
            window.focus(&focus, cx);
        });
        assert!(
            (0..32).any(|_| {
                cx.simulate_keystrokes("tab");
                overview_block_has_focus(cx, label)
            }),
            "Tab reaches the selected overview block"
        );
        cx.simulate_keystrokes("tab");
        assert!(overview_block_has_focus(
            cx,
            "Structure overview selected node org.example:right; activate to centre the viewport"
        ));
        cx.simulate_keystrokes("shift-tab");
        assert!(overview_block_has_focus(cx, label));
        let selection_target = DependencyDiagramContextTarget::Selection {
            node: "c1".into(),
            members: ["c1".into(), "c2".into()].into(),
        };
        cx.update(|window, cx| {
            let keystroke = gpui::Keystroke::parse("shift-f10").expect("context-menu key");
            window.dispatch_event(
                gpui::KeyDownEvent {
                    keystroke,
                    is_held: false,
                    prefer_character_input: false,
                }
                .to_platform_input(),
                cx,
            );
            let item = item.read(cx);
            assert_eq!(item.canvas.read(cx).focus.as_deref(), Some("c2"));
            let staged = item.context_state.invocation.lock().expect("staged menu");
            let invocation = staged.as_ref().expect("keyboard menu invocation");
            assert_eq!(invocation.target, selection_target);
            let position = invocation.screen_position;
            assert!(
                (overview_origin.x..overview_origin.x + STRUCTURE_OVERVIEW_SIZE.width)
                    .contains(&position.x)
            );
            assert!(
                (overview_origin.y..overview_origin.y + STRUCTURE_OVERVIEW_SIZE.height)
                    .contains(&position.y)
            );
        });
        assert!(item.read_with(cx, |item, _| item.context_handle.is_deployed()));
        cx.simulate_keystrokes("escape");
        assert!(!item.read_with(cx, |item, _| item.context_handle.is_deployed()));
        assert!(overview_block_has_focus(cx, label));

        let overview_node_center = |node: &str, cx: &mut gpui::VisualTestContext| {
            cx.update(|_, cx| {
                let graph = &item.read(cx).canvas.read(cx).graph;
                let origin = graph.positions.get(node).expect("overview node");
                let size = graph.node_size();
                overview_origin.translated(GraphPoint {
                    x: (origin.x - bounds_origin.x) * scale + (size.width * scale).max(3.0) / 2.0,
                    y: (origin.y - bounds_origin.y) * scale + (size.height * scale).max(2.0) / 2.0,
                })
            })
        };
        let viewport = cx.update(|_, cx| item.read(cx).canvas.read(cx).viewport.clone());
        for (click, expected_target) in [
            (overview_node_center("c1", cx), selection_target),
            (
                overview_node_center("root", cx),
                DependencyDiagramContextTarget::Node("root".into()),
            ),
            (
                overview_origin.translated(GraphPoint {
                    x: STRUCTURE_OVERVIEW_SIZE.width - 6.0,
                    y: STRUCTURE_OVERVIEW_SIZE.height - 6.0,
                }),
                DependencyDiagramContextTarget::Canvas,
            ),
        ] {
            let position = gpui::point(px(click.x), px(click.y));
            cx.update(|window, cx| {
                window.dispatch_event(
                    gpui::MouseDownEvent {
                        button: gpui::MouseButton::Right,
                        position,
                        modifiers: Default::default(),
                        click_count: 1,
                        first_mouse: false,
                    }
                    .to_platform_input(),
                    cx,
                );
                let item = item.read(cx);
                let staged = item.context_state.invocation.lock().expect("staged menu");
                let invocation = staged.as_ref().expect("overview pointer menu invocation");
                assert_eq!(invocation.target, expected_target);
                assert_eq!(invocation.screen_position, click);
            });
            cx.simulate_mouse_up(position, gpui::MouseButton::Right, Default::default());
            assert!(item.read_with(cx, |item, _| item.context_handle.is_deployed()));
            cx.simulate_keystrokes("escape");
            assert!(!item.read_with(cx, |item, _| item.context_handle.is_deployed()));
            assert!(overview_block_has_focus(cx, label));
            cx.update(|_, cx| {
                let canvas = item.read(cx).canvas.read(cx);
                assert_eq!(canvas.selection, BTreeSet::from(["c1".into(), "c2".into()]));
                assert_eq!(canvas.focus.as_deref(), Some("c2"));
                assert_eq!(canvas.viewport, viewport);
            });
        }

        for activation_key in ["enter", "space"] {
            item.update(cx, |item, cx| item.move_viewport_to_overview(pointer, cx));
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let keystroke = gpui::Keystroke::parse(activation_key).expect("activation key");
                window.dispatch_event(
                    gpui::KeyDownEvent {
                        keystroke: keystroke.clone(),
                        is_held: false,
                        prefer_character_input: false,
                    }
                    .to_platform_input(),
                    cx,
                );
                window.dispatch_event(gpui::KeyUpEvent { keystroke }.to_platform_input(), cx);
            });
            cx.update(|_, cx| {
                let canvas = item.read(cx).canvas.read(cx);
                let origin = canvas.graph.positions.get("c1").expect("selected block");
                let node_size = canvas.graph.node_size();
                let screen = canvas.viewport.graph_to_screen(GraphPoint {
                    x: origin.x + node_size.width / 2.0,
                    y: origin.y + node_size.height / 2.0,
                });
                assert!((screen.x - canvas_size.width / 2.0).abs() < 0.01);
                assert!((screen.y - canvas_size.height / 2.0).abs() < 0.01);
                assert_eq!(canvas.selection, BTreeSet::from(["c1".into(), "c2".into()]));
                assert_eq!(canvas.viewport.scale, viewport_scale);
            });
        }
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
            svg.match_indices("<polyline ").count(),
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
                .is_some_and(|notice| notice.contains("requires current provider facts")),
            "Copy Diagram Source without a source reports guidance: {copy_notice:?}"
        );
        assert!(
            !svg.contains("flowchart"),
            "the export is not the Mermaid payload"
        );
    }

    #[gpui::test]
    async fn dependency_diagram_menu_routes_keyboard_and_pointer_targets(cx: &mut TestAppContext) {
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
        // Draw once so the right-click menu handle is attached to the rendered
        // element tree before the keyboard gesture.
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });
        let context_handle = cx.update(|_, cx| item.read(cx).context_handle.clone());

        let node_focus = item.read_with(cx, |item, cx| {
            item.canvas()
                .read(cx)
                .focus_handle_for_node("c1")
                .expect("canvas node focus handle")
        });
        cx.update(|window, cx| {
            window.focus(&node_focus, cx);
            window.draw(cx).clear(cx);
        });
        cx.simulate_keystrokes("shift-f10");
        assert!(
            context_handle.is_deployed(),
            "opens from keyboard node focus"
        );

        cx.update(|window, cx| context_handle.focus(window, cx));
        cx.simulate_keystrokes("down");
        let navigating = cx.update(|window, cx| {
            context_handle.is_deployed() && context_handle.is_focused(window, cx)
        });
        assert!(navigating, "arrow keys navigate the deployed menu");
        cx.simulate_keystrokes("shift-f10");
        cx.simulate_keystrokes("escape");
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });
        let (dismissed, restored) =
            cx.update(|window, _| (!context_handle.is_deployed(), node_focus.is_focused(window)));
        assert!(dismissed, "Escape dismisses the keyboard context menu");
        assert!(
            restored,
            "focus returns to the exact dependency after Escape"
        );

        let viewport = item.update(cx, |item, cx| {
            item.canvas.update(cx, |canvas, cx| {
                canvas.select_node("c1", false, false);
                canvas.select_node("c2", true, false);
                cx.notify();
                canvas.viewport.clone()
            })
        });
        let item_focus = cx.update(|window, cx| {
            window.activate_a11y_for_test();
            let focus = item.read(cx).focus_handle(cx);
            window.focus(&focus, cx);
            window.draw(cx).clear(cx);
            focus
        });
        let aria_focus = |cx: &mut gpui::VisualTestContext, label: &str, role: &str| {
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let tree: serde_json::Value =
                    serde_json::from_str(&window.debug_a11y_tree_json().expect("diagram"))
                        .expect("accessibility JSON");
                tree["gpui_focus"].as_str().is_some_and(|focus| {
                    let aria = &tree["nodes"][focus]["aria"];
                    aria["label"] == label && aria["role"] == role
                })
            })
        };
        assert!((0..32).any(|_| {
            cx.simulate_keystrokes("tab");
            aria_focus(cx, "Fit dependency diagram", "Button")
        }));
        cx.simulate_keystrokes("shift-f10");
        assert!(
            !context_handle.is_deployed(),
            "toolbar focus must not open a menu for a previously selected dependency"
        );
        assert!(aria_focus(cx, "Fit dependency diagram", "Button"));

        assert!(
            (0..32).any(|_| {
                cx.simulate_keystrokes("tab");
                aria_focus(cx, "Dependency diagram", "Application")
            }),
            "Tab reaches the diagram canvas with nodes selected"
        );
        let canvas_focus = cx.update(|window, cx| window.focused(cx).expect("canvas focus"));
        for focus in [canvas_focus, item_focus] {
            cx.update(|window, cx| {
                window.focus(&focus, cx);
                window.draw(cx).clear(cx);
            });
            cx.simulate_keystrokes("shift-f10");
            assert!(
                context_handle.is_deployed(),
                "the focused diagram can open the blank-canvas menu"
            );
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let tree: serde_json::Value =
                    serde_json::from_str(&window.debug_a11y_tree_json().expect("diagram menu"))
                        .expect("accessibility JSON");
                assert!(
                    tree["nodes"]
                        .as_object()
                        .expect("menu nodes")
                        .values()
                        .any(|node| node["aria"]["label"] == "Layout"),
                    "the menu offers the canvas-only Layout submenu"
                );
            });
            cx.simulate_keystrokes("escape");
            assert!(!context_handle.is_deployed());
            assert!(cx.update(|window, _| focus.is_focused(window)));
        }

        let pointer_targets = item.read_with(cx, |item, cx| {
            let canvas = item.canvas.read(cx);
            let origin = *canvas.canvas_origin.lock().expect("canvas origin");
            let point_for_node =
                |id| origin.translated(canvas.node_screen_position(id).expect("canvas node"));
            let size = *canvas.canvas_size.lock().expect("canvas size");
            [
                (
                    point_for_node("c1"),
                    DependencyDiagramContextTarget::Selection {
                        node: "c1".into(),
                        members: ["c1".into(), "c2".into()].into(),
                    },
                ),
                (
                    point_for_node("root"),
                    DependencyDiagramContextTarget::Node("root".into()),
                ),
                (
                    origin.translated(GraphPoint {
                        x: size.width - 12.0,
                        y: size.height - 12.0,
                    }),
                    DependencyDiagramContextTarget::Canvas,
                ),
            ]
        });
        for (target_point, expected_target) in pointer_targets {
            let position = gpui::point(px(target_point.x), px(target_point.y));
            let invocation = cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                window.dispatch_event(
                    gpui::MouseDownEvent {
                        button: gpui::MouseButton::Right,
                        position,
                        modifiers: Default::default(),
                        click_count: 1,
                        first_mouse: false,
                    }
                    .to_platform_input(),
                    cx,
                );
                item.read(cx)
                    .context_state
                    .invocation
                    .lock()
                    .expect("staged invocation")
                    .clone()
            });
            cx.simulate_mouse_up(position, gpui::MouseButton::Right, Default::default());
            assert!(
                context_handle.is_deployed(),
                "right-click opens the diagram menu"
            );
            assert_eq!(
                invocation.expect("pointer invocation").target,
                expected_target
            );
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let tree: serde_json::Value =
                    serde_json::from_str(&window.debug_a11y_tree_json().expect("diagram menu"))
                        .expect("accessibility JSON");
                let nodes = tree["nodes"].as_object().expect("menu nodes");
                let has_label = |label| nodes.values().any(|node| node["aria"]["label"] == label);
                assert!(has_label("Fit Diagram"));
                assert_eq!(
                    has_label("Layout"),
                    expected_target == DependencyDiagramContextTarget::Canvas
                );
            });
            cx.simulate_keystrokes("escape");
            assert!(!context_handle.is_deployed());
        }
        item.read_with(cx, |item, cx| {
            let canvas = item.canvas.read(cx);
            assert_eq!(canvas.selection, BTreeSet::from(["c1".into(), "c2".into()]));
            assert_eq!(canvas.viewport.offset, viewport.offset);
            assert_eq!(canvas.viewport.scale, viewport.scale);
        });
    }

    #[gpui::test]
    async fn analyzer_diagram_linkage_all_usages_requires_opening_panel(cx: &mut TestAppContext) {
        cx.update(|cx| init(cx));
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-diagram-linkage",
            serde_json::json!({"pom.xml": "<project/>"}),
        )
        .await;
        let project = project::Project::test(fs, ["/groovy-diagram-linkage".as_ref()], cx).await;
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
        // All usages without an opening panel reports guidance instead of
        // dispatching; the same rejection covers a stale diagram whose panel
        // is gone.
        item.update(cx, |item, cx| {
            item.open_analyzer_for_target(&DependencyDiagramContextTarget::Node("c1".into()), cx);
        });
        let notice = cx.update(|_, cx| item.read(cx).notice.clone());
        assert!(
            notice
                .as_deref()
                .is_some_and(|notice| notice.contains("opening panel is unavailable")),
            "All usages without an opening panel reports guidance: {notice:?}"
        );
    }
}
