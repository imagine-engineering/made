//! The dependency diagram editor item: a dedicated tab hosting the read-only
//! graph canvas for one selected project or module. The item owns the canvas
//! viewport, selection and inspection overlay; admitted facts come from the
//! Dependencies panel's snapshot at open time and are re-derived without
//! provider queries.

use crate::dependencies_panel::{
    AddGraphSelectionToSet, ClearGraphInspection, ClearGraphSelectionSet,
    CopyDependencyDiagramSource, DependencyGraphReceipt, InspectGraphAllRootPaths,
    InspectGraphAllUsages, InspectGraphConflictOrigins, InspectGraphNeighbours,
    InspectGraphPathsBetween, InspectGraphRootPaths, InspectGraphSelectionSet, PinGraphSelection,
    current_dependency_graph_receipt, derive_dependency_diagram,
};
use crate::dependency_diagram_menu::{
    DependencyDiagramContextTarget, DependencyDiagramMenuCommand, DependencyDiagramMenuSnapshot,
    dependency_diagram_menu_available, stage_dependency_diagram_target,
};
use crate::dependency_graph_canvas::{CanvasNodeKind, CanvasState};
use gpui::{
    App, AppContext as _, Context, DismissEvent, EventEmitter, FocusHandle, Focusable, FontWeight,
    Render, SharedString, Task, Window, actions,
};
use picker::{Picker, PickerDelegate};
use std::{collections::BTreeSet, sync::Arc};
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
        let query = query.trim().to_lowercase();
        self.matches = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.label.to_lowercase().contains(&query))
            .map(|(index, _)| index)
            .take(16)
            .collect();
        self.selected_index = 0;
    }

    fn entry(&self, index: usize) -> Option<&DependencySearchEntry> {
        self.entries.get(*self.matches.get(index)?)
    }

    pub(crate) fn popover(self, window: &mut Window, cx: &mut App) -> gpui::Entity<Picker<Self>> {
        cx.new(|cx| Picker::uniform_list(self, window, cx).popover())
    }
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
                source.view,
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
            let size = *canvas.canvas_size.lock().expect("canvas size");
            canvas.fit(size);
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
        match derive_dependency_diagram(
            project,
            source.view,
            &source.selection,
            &source.node,
            Some(&scope),
            cx,
        ) {
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
        self.canvas
            .read(cx)
            .graph
            .nodes
            .iter()
            .map(|node| DependencySearchEntry {
                id: node.id.clone(),
                label: match &node.version {
                    Some(version) => format!("{}:{version}", node.label),
                    None => node.label.clone(),
                },
            })
            .collect()
    }

    pub fn reveal_node_by_id(&mut self, id: &str, cx: &mut Context<Self>) {
        if !self.receipt_is_current(cx) {
            return;
        }
        let revealed = self.canvas.update(cx, |canvas, cx| {
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

    /// Toggles the minimap-style Structure overview panel.
    pub fn toggle_structure(&mut self, cx: &mut Context<Self>) {
        self.structure_visible = !self.structure_visible;
        cx.notify();
    }

    /// Opens the context menu for the keyboard-focused node: the platform
    /// context-menu gesture stages the same target a right-click would and
    /// shows the menu at the node's screen position.
    fn show_context_menu_for_focus(&mut self, window: &mut gpui::Window, cx: &mut Context<Self>) {
        let canvas = self.canvas.read(cx);
        let focused = canvas
            .focused_node(window)
            .or(canvas.focus.as_deref())
            .or_else(|| canvas.selection.iter().next().map(String::as_str));
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
        let position = gpui::point(gpui::px(origin.x + screen.x), gpui::px(origin.y + screen.y));
        self.context_state.stage(
            crate::dependency_diagram_menu::DependencyDiagramContextInvocation {
                target,
                screen_position: sim_runtime::graph::GraphPoint {
                    x: position.x.into(),
                    y: position.y.into(),
                },
            },
        );
        let handle = self.context_handle.clone();
        window.defer(cx, move |window, cx| {
            handle.show_at(position, window, cx);
        });
    }

    /// Dismisses the staged context menu and restores focus to the diagram
    /// item: the Escape path for the keyboard context-menu gesture.
    fn dismiss_context_menu(&mut self, window: &mut gpui::Window, cx: &mut Context<Self>) {
        if !self.context_handle.is_deployed() {
            cx.propagate();
            return;
        }
        self.context_handle.hide(cx);
        window.focus(&self.focus_handle, cx);
        // Re-assert the restoration on the next frame so a deferred focus
        // claim from the dismissed overlay cannot outlast the gesture.
        let focus_handle = self.focus_handle.clone();
        window.on_next_frame(move |window, cx| window.focus(&focus_handle, cx));
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
        let (selection_set_len, has_matching_pin, accepts_set) = panel
            .read(cx)
            .diagram_comparison_availability(&source.selection, target.node());
        let maven_scope = match &self.scope {
            DependencyDiagramScope::MavenScope(scope) => {
                crate::dependencies_panel::maven_graph_scope_from_label(scope)
            }
            DependencyDiagramScope::GradleConfiguration(_) => Default::default(),
        };
        DependencyDiagramMenuSnapshot {
            selection_set_len,
            has_inspection: !canvas.inspected.is_empty(),
            has_source: true,
            has_admitted_node: target.node().is_some_and(|id| {
                canvas
                    .graph
                    .node(id)
                    .is_some_and(|node| node.kind != CanvasNodeKind::Unresolved)
            }),
            has_matching_pin,
            can_add_to_set: accepts_set && selection_set_len < 8,
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
                    .update(cx, |panel, cx| panel.clear_graph_inspection(cx))
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
                let nodes = if matches!(target, DependencyDiagramContextTarget::Selection(_)) {
                    self.canvas.read(cx).selection.iter().cloned().collect()
                } else {
                    vec![node.to_owned()]
                };
                source.panel.update(cx, |panel, cx| {
                    panel.add_graph_nodes_to_set(&source.selection, nodes, cx)
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
            panel.inspect_diagram_graph(&source.selection, node, mode, cx)
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

    pub(crate) fn select_analyzer_node(&mut self, id: &str, cx: &mut Context<Self>) {
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
            } else {
                canvas.reveal_node(id);
            }
            cx.notify();
        });
    }

    /// Exports the current canvas as a deterministic SVG image and copies it
    /// to the clipboard; distinct from the Mermaid Copy Diagram Source payload.
    pub fn export_diagram_image(&mut self, cx: &mut Context<Self>) {
        if !self.receipt_is_current(cx) {
            return;
        }
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
            .child(
                ui::PopoverMenu::new("dependency-diagram-search")
                    .with_handle(self.search_handle.clone())
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
                    .menu(move |window, cx| {
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
                    }),
            )
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
                    .aria_label("Export dependency diagram image")
                    .disabled(!admitted)
                    .tooltip(ui::Tooltip::text("Export Image"))
                    .on_click(cx.listener(|item, _, _, cx| item.export_diagram_image(cx))),
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
                cx.listener(|item, _: &DismissDependencyDiagramMenu, window, cx| {
                    item.dismiss_context_menu(window, cx)
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
    _panel_subscription: Option<gpui::Subscription>,
}

impl DependencyDiagramPopup {
    pub fn new(open: DependencyDiagramOpen, cx: &mut Context<Self>) -> Self {
        let DependencyDiagramOpen { derivation, source } = open;
        let receipt =
            current_dependency_graph_receipt(source.project.read(cx), &source.selection, cx);
        let panel_subscription = source.panel.upgrade().map(|panel| {
            cx.observe(&panel, move |popup, _, cx| {
                let current = current_dependency_graph_receipt(
                    source.project.read(cx),
                    &source.selection,
                    cx,
                );
                if current != receipt {
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
            focus_handle: cx.focus_handle(),
            _panel_subscription: panel_subscription,
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
                .is_some_and(|notice| notice.contains("No provider source is attached")),
            "Copy Diagram Source without a source reports guidance: {copy_notice:?}"
        );
        assert!(
            !svg.contains("flowchart"),
            "the export is not the Mermaid payload"
        );
    }

    #[gpui::test]
    async fn dependency_diagram_menu_keyboard_gesture_navigates_and_restores_focus(
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
        let item = workspace.update_in(cx, |workspace, window, cx| {
            DependencyDiagramItem::open(derivation, None, workspace, window, cx)
        });
        // Draw once so the right-click menu handle is attached to the rendered
        // element tree before the keyboard gesture.
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });
        let context_handle = cx.update(|_, cx| item.read(cx).context_handle.clone());

        // Focus the item and one of its nodes; the platform keyboard
        // context-menu gesture stages the focused node's target and shows the
        // menu at the node's screen position.
        item.update(cx, |item, cx| {
            item.canvas
                .update(cx, |canvas, _| canvas.focus = Some("c1".into()));
        });
        cx.update(|window, cx| {
            window.focus(&item.read(cx).focus_handle(cx), cx);
            // Flush the pending focus so the menu's dismissal restores this
            // exact handle (focus becomes effective on the next frame).
            window.draw(cx).clear(cx);
        });
        cx.update(|window, cx| {
            item.update(cx, |item, cx| item.show_context_menu_for_focus(window, cx));
        });
        assert!(
            context_handle.is_deployed(),
            "the keyboard gesture deploys the context menu at the focused node"
        );

        // Escape dismisses the menu and restores focus to the diagram item:
        // the item's key context binds Escape to the dismissal action, which
        // hides the menu and re-focuses the item handle. The navigation
        // keystroke runs first, like a user arrowing through the menu before
        // cancelling.
        cx.simulate_keystrokes("down");
        cx.simulate_keystrokes("escape");
        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });
        let (dismissed, restored) = cx.update(|window, cx| {
            (
                !context_handle.is_deployed(),
                item.read(cx).focus_handle(cx).is_focused(window),
            )
        });
        assert!(dismissed, "Escape dismisses the keyboard context menu");
        assert!(restored, "focus returns to the diagram item after Escape");

        // Second gesture: the shared ui::ContextMenu keeps arrow-key
        // navigation inside the deployed menu without dismissing it. The
        // handle's focus API mirrors the platform gesture handing focus to
        // the menu (the test window does not flush the deferred focus frames).
        cx.update(|window, cx| {
            item.update(cx, |item, cx| item.show_context_menu_for_focus(window, cx));
        });
        assert!(
            context_handle.is_deployed(),
            "the gesture re-deploys the menu"
        );
        cx.update(|window, cx| context_handle.focus(window, cx));
        cx.simulate_keystrokes("down");
        let navigating = cx.update(|window, cx| {
            context_handle.is_deployed() && context_handle.is_focused(window, cx)
        });
        assert!(navigating, "arrow keys navigate the deployed menu");
        cx.simulate_keystrokes("escape");
        assert!(
            !context_handle.is_deployed(),
            "Escape dismisses the navigated menu"
        );
        item.update(cx, |item, cx| {
            item.canvas.update(cx, |canvas, cx| {
                canvas.focus = None;
                canvas.selection.clear();
                cx.notify();
            });
        });
        cx.update(|window, cx| {
            item.update(cx, |item, cx| item.show_context_menu_for_focus(window, cx));
        });
        assert!(
            context_handle.is_deployed(),
            "the gesture can open the blank-canvas menu"
        );
        cx.simulate_keystrokes("escape");
        assert!(!context_handle.is_deployed());
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
