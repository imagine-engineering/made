//! Read-only dependency graph canvas model: adapters from admitted provider
//! graph facts to canvas nodes/edges, and a deterministic layered layout.
//!
//! Node and edge identity follows the provider occurrence scheme already used
//! by the Dependencies panel's fact rows: node ids are provider component or
//! occurrence ids, and synthetic unresolved targets are numbered
//! deterministically. The same admitted facts always produce the same canvas,
//! including positions, so the viewport stays stable across re-derivation.

use gpui::px;
use project::jvm_project::{GradleResolutionGraph, MavenDependencyGraph};
use sim_runtime::graph::{GraphPoint, GraphRect, GraphSize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Visible canvas budgets. They mirror the existing diagram budgets so larger
/// admitted facts stay bounded and explicitly marked as truncated.
pub const MAX_CANVAS_NODES: usize = 64;
pub const MAX_CANVAS_EDGES: usize = 128;

const NODE_WIDTH: f32 = 168.0;
const NODE_HEIGHT: f32 = 44.0;
const HORIZONTAL_GAP: f32 = 96.0;
const VERTICAL_GAP: f32 = 48.0;
const PADDING: f32 = 24.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanvasNodeKind {
    Root,
    Module,
    Project,
    Unresolved,
}

#[derive(Clone, Debug)]
pub struct CanvasNode {
    /// Provider component or occurrence identity; unresolved targets use
    /// numbered synthetic ids (`unresolved:{index}`).
    pub id: String,
    pub component_id: String,
    pub label: String,
    pub version: Option<String>,
    pub kind: CanvasNodeKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanvasEdgeVariant {
    Dependency,
    Constraint,
    Unresolved,
    Omitted,
}

impl CanvasEdgeVariant {
    fn label(self) -> &'static str {
        match self {
            Self::Dependency => "Dependency",
            Self::Constraint => "Constraint",
            Self::Unresolved => "Unresolved dependency",
            Self::Omitted => "Omitted dependency",
        }
    }

    fn dash_pattern(self) -> Option<(f32, f32)> {
        match self {
            Self::Dependency => None,
            Self::Constraint => Some((3.0, 3.0)),
            Self::Unresolved => Some((7.0, 4.0)),
            Self::Omitted => Some((2.0, 5.0)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CanvasEdge {
    pub id: String,
    pub provider_index: usize,
    pub from: String,
    pub to: String,
    pub variant: CanvasEdgeVariant,
    pub label: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum CanvasLayout {
    #[default]
    LeftToRight,
    TopToBottom,
}

impl CanvasLayout {
    fn transpose(self, point: GraphPoint) -> GraphPoint {
        match self {
            Self::LeftToRight => point,
            Self::TopToBottom => GraphPoint {
                x: point.y,
                y: point.x,
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CanvasGraph {
    pub nodes: Vec<CanvasNode>,
    pub edges: Vec<CanvasEdge>,
    /// True when admitted facts exceeded the visible budgets and only the
    /// first nodes/edges in provider order are presented.
    pub truncated: bool,
    /// Deterministic post-layout positions, in graph coordinates.
    pub positions: BTreeMap<String, GraphPoint>,
    pub bounds: GraphRect,
    pub(crate) layout: CanvasLayout,
}

impl CanvasGraph {
    pub fn node(&self, id: &str) -> Option<&CanvasNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    pub fn node_size(&self) -> GraphSize {
        GraphSize {
            width: NODE_WIDTH,
            height: NODE_HEIGHT,
        }
    }
}

fn component_label(group: Option<&str>, name: &str) -> String {
    match group {
        Some(group) => format!("{group}:{name}"),
        None => name.to_owned(),
    }
}

/// Adapts an admitted Gradle resolution graph. Unresolved edges receive one
/// numbered synthetic node each, matching the panel's per-edge placeholders.
pub fn canvas_graph_from_gradle(graph: &GradleResolutionGraph) -> CanvasGraph {
    let mut nodes = Vec::new();
    for component in &graph.components {
        let (label, kind) = match (&component.group, &component.module, component.kind) {
            (_, _, project::jvm_project::GradleDependencyKind::Project) => (
                component
                    .project_path
                    .clone()
                    .or_else(|| component.build_tree_path.clone())
                    .unwrap_or_else(|| component.id.clone()),
                CanvasNodeKind::Project,
            ),
            (Some(group), Some(module), _) => {
                (component_label(Some(group), module), CanvasNodeKind::Module)
            }
            (_, Some(module), _) => (module.clone(), CanvasNodeKind::Module),
            _ => (component.id.clone(), CanvasNodeKind::Module),
        };
        nodes.push(CanvasNode {
            id: component.id.clone(),
            component_id: component.id.clone(),
            label,
            version: component.version.clone(),
            kind,
        });
    }
    if let Some(root) = nodes
        .iter_mut()
        .find(|node| node.id == graph.root_component)
    {
        root.kind = CanvasNodeKind::Root;
    }

    let mut edges = Vec::new();
    let mut unresolved_count = 0usize;
    for (provider_index, edge) in graph.edges.iter().enumerate() {
        let (requested_label, requested_version) = match &edge.requested {
            project::jvm_project::GradleDependencySelector::Module {
                group,
                module,
                version,
                ..
            } => (
                component_label(Some(group), module),
                (!version.is_empty()).then(|| version.clone()),
            ),
            project::jvm_project::GradleDependencySelector::Project { project_path, .. } => {
                (project_path.clone(), None)
            }
        };
        let target = match (&edge.resolved, &edge.selected) {
            (true, Some(selected)) => selected.clone(),
            _ => {
                let target = format!("unresolved:{unresolved_count}");
                unresolved_count += 1;
                nodes.push(CanvasNode {
                    id: target.clone(),
                    component_id: String::new(),
                    label: requested_label.clone(),
                    version: requested_version.clone(),
                    kind: CanvasNodeKind::Unresolved,
                });
                target
            }
        };
        let variant = if !edge.resolved {
            CanvasEdgeVariant::Unresolved
        } else if edge.constraint {
            CanvasEdgeVariant::Constraint
        } else {
            CanvasEdgeVariant::Dependency
        };
        edges.push(CanvasEdge {
            id: format!("gradle-edge:{provider_index}"),
            provider_index,
            from: edge.from.clone(),
            to: target,
            variant,
            label: match requested_version {
                Some(version) => format!("{requested_label} {version}"),
                None => requested_label,
            },
        });
    }

    assemble_canvas_graph(nodes, edges)
}

/// Adapts an admitted Maven dependency graph. Nodes are occurrences (the
/// provider reports one node per placement), preserving repeated artifacts.
pub fn canvas_graph_from_maven(
    graph: &MavenDependencyGraph,
    include_edge: impl Fn(&project::jvm_project::MavenDependencyEdge) -> bool,
) -> CanvasGraph {
    let mut components = HashMap::new();
    for component in &graph.components {
        let artifact = if component.classifier.is_empty() {
            component.artifact_id.clone()
        } else {
            format!("{}:{}", component.artifact_id, component.classifier)
        };
        let extension = if component.extension.is_empty() || component.extension == "jar" {
            String::new()
        } else {
            format!(".{}", component.extension)
        };
        components.insert(
            component.id.clone(),
            (
                component_label(Some(&component.group_id), &format!("{artifact}{extension}")),
                Some(component.version.clone()),
            ),
        );
    }
    let mut nodes = Vec::new();
    for node in &graph.nodes {
        let (label, version) = components
            .get(&node.component)
            .cloned()
            .unwrap_or_else(|| (node.id.clone(), None));
        nodes.push(CanvasNode {
            id: node.id.clone(),
            component_id: node.component.clone(),
            label,
            version,
            kind: CanvasNodeKind::Module,
        });
    }
    if let Some(root) = nodes.iter_mut().find(|node| node.id == graph.root_node) {
        root.kind = CanvasNodeKind::Root;
    }
    let mut edges = Vec::new();
    for (provider_index, edge) in graph.edges.iter().enumerate() {
        if !include_edge(edge) {
            continue;
        }
        let mut label = edge.scope.clone();
        if edge.optional {
            label.push_str("; optional");
        }
        if let Some(constraint) = &edge.version_constraint {
            label.push_str(&format!(
                "; constrained to {constraint}{}",
                edge.premanaged_version
                    .as_ref()
                    .map(|premanaged| format!("; premanaged {premanaged}"))
                    .unwrap_or_default()
            ));
        }
        edges.push(CanvasEdge {
            id: format!(
                "maven-edge:{}:{}:{}",
                edge.from_node, edge.target_node, edge.position
            ),
            provider_index,
            from: edge.from_node.clone(),
            to: edge.target_node.clone(),
            variant: if edge.omitted {
                CanvasEdgeVariant::Omitted
            } else {
                CanvasEdgeVariant::Dependency
            },
            label,
        });
    }

    assemble_canvas_graph(nodes, edges)
}

/// Applies the visible budgets in provider order and lays the graph out.
fn assemble_canvas_graph(mut nodes: Vec<CanvasNode>, mut edges: Vec<CanvasEdge>) -> CanvasGraph {
    let mut truncated = false;
    if nodes.len() > MAX_CANVAS_NODES {
        let retained: HashSet<String> = nodes
            .iter()
            .take(MAX_CANVAS_NODES)
            .map(|node| node.id.clone())
            .collect();
        nodes.truncate(MAX_CANVAS_NODES);
        edges.retain(|edge| retained.contains(&edge.from) && retained.contains(&edge.to));
        truncated = true;
    }
    if edges.len() > MAX_CANVAS_EDGES {
        edges.truncate(MAX_CANVAS_EDGES);
        truncated = true;
    }
    let mut graph = CanvasGraph {
        nodes,
        edges,
        truncated,
        ..CanvasGraph::default()
    };
    layout_canvas_graph(&mut graph);
    graph
}

/// Deterministic layered layout: cycle-breaking DFS, longest-path layering
/// over the acyclic subgraph, barycentric ordering with lexicographic
/// tie-breaks, and fixed compact node sizes. Identical facts always produce
/// identical positions.
pub fn layout_canvas_graph(graph: &mut CanvasGraph) {
    let ids: Vec<String> = graph.nodes.iter().map(|node| node.id.clone()).collect();
    let id_set: HashSet<String> = ids.iter().cloned().collect();
    let mut outgoing: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut incoming: HashMap<String, usize> = HashMap::new();
    let mut forward_edges: Vec<(String, String)> = Vec::new();
    for id in &ids {
        incoming.entry(id.clone()).or_insert(0);
    }
    for edge in &graph.edges {
        if !id_set.contains(&edge.from) || !id_set.contains(&edge.to) {
            continue;
        }
        if edge.from != edge.to
            && !reaches(&outgoing, &edge.to, &edge.from)
            && outgoing
                .entry(edge.from.clone())
                .or_default()
                .insert(edge.to.clone())
        {
            forward_edges.push((edge.from.clone(), edge.to.clone()));
            *incoming.entry(edge.to.clone()).or_insert(0) += 1;
        }
    }

    // Longest-path layering in topological order (Kahn's algorithm over the
    // acyclic subgraph). Cyclic-only nodes stay at layer 0.
    let mut layers: HashMap<String, usize> = HashMap::new();
    let mut ready: Vec<String> = ids
        .iter()
        .filter(|id| incoming.get(*id).copied().unwrap_or(0) == 0)
        .cloned()
        .collect();
    let mut remaining: HashMap<String, usize> = incoming.clone();
    // Deterministic Kahn ordering: a BTreeSet queue ordered by id.
    let mut queue: BTreeSet<String> = ready.drain(..).collect();
    while let Some(id) = queue.iter().next().cloned() {
        queue.remove(&id);
        let layer = *layers.entry(id.clone()).or_insert(0);
        for next in outgoing.get(&id).cloned().unwrap_or_default() {
            let entry = layers.entry(next.clone()).or_insert(0);
            *entry = (*entry).max(layer + 1);
            let count = remaining.entry(next.clone()).or_insert(0);
            *count = count.saturating_sub(1);
            if *count == 0 {
                queue.insert(next);
            }
        }
    }
    for id in &ids {
        layers.entry(id.clone()).or_insert(0);
    }

    // Order within layers: lexicographic seed, then barycentric sweeps.
    let mut layer_members: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for id in &ids {
        layer_members
            .entry(layers[id])
            .or_default()
            .push(id.clone());
    }
    for members in layer_members.values_mut() {
        members.sort();
    }
    let mut order_index: HashMap<String, usize> = HashMap::new();
    for members in layer_members.values() {
        for (index, id) in members.iter().enumerate() {
            order_index.insert(id.clone(), index);
        }
    }
    let predecessors: HashMap<&str, Vec<&str>> = predecessor_map(&forward_edges);
    let successors = successor_map(&forward_edges);
    for _ in 0..2 {
        // Downward sweep: order by the mean predecessor order index.
        for members in layer_members.values_mut() {
            sort_by_barycenter(members, &predecessors, &order_index);
        }
        rebuild_order_index(&layer_members, &mut order_index);
        // Upward sweep: order by the mean successor order index.
        for members in layer_members.values_mut().rev() {
            sort_by_barycenter(members, &successors, &order_index);
        }
        rebuild_order_index(&layer_members, &mut order_index);
    }

    graph.positions.clear();
    let block = graph.layout.transpose(GraphPoint {
        x: NODE_WIDTH,
        y: NODE_HEIGHT,
    });
    for (layer, members) in &layer_members {
        for (index, id) in members.iter().enumerate() {
            graph.positions.insert(
                id.clone(),
                graph.layout.transpose(GraphPoint {
                    x: PADDING + *layer as f32 * (block.x + HORIZONTAL_GAP),
                    y: PADDING + index as f32 * (block.y + VERTICAL_GAP),
                }),
            );
        }
    }
    recompute_bounds(graph);
}

fn reaches(outgoing: &BTreeMap<String, BTreeSet<String>>, from: &str, to: &str) -> bool {
    let mut seen = HashSet::new();
    let mut queue = vec![from];
    while let Some(id) = queue.pop() {
        if !seen.insert(id) {
            continue;
        }
        for next in outgoing.get(id).into_iter().flatten() {
            if next == to {
                return true;
            }
            if !seen.contains(next.as_str()) {
                queue.push(next);
            }
        }
    }
    false
}

fn predecessor_map(edges: &[(String, String)]) -> HashMap<&str, Vec<&str>> {
    let mut map: HashMap<&str, Vec<&str>> = HashMap::new();
    for (from, to) in edges {
        map.entry(to).or_default().push(from.as_str());
    }
    map
}

fn successor_map(edges: &[(String, String)]) -> HashMap<&str, Vec<&str>> {
    let mut map: HashMap<&str, Vec<&str>> = HashMap::new();
    for (from, to) in edges {
        map.entry(from).or_default().push(to.as_str());
    }
    map
}

fn sort_by_barycenter(
    members: &mut [String],
    neighbours: &HashMap<&str, Vec<&str>>,
    order_index: &HashMap<String, usize>,
) {
    members.sort_by(|a, b| {
        let key_a = mean_position(neighbours, a, order_index);
        let key_b = mean_position(neighbours, b, order_index);
        key_a
            .0
            .cmp(&key_b.0)
            .then(
                key_a
                    .1
                    .partial_cmp(&key_b.1)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.cmp(b))
    });
}

fn mean_position(
    neighbours: &HashMap<&str, Vec<&str>>,
    id: &str,
    order_index: &HashMap<String, usize>,
) -> (bool, f64) {
    let positions: Vec<f64> = neighbours
        .get(id)
        .map(|neighbours| {
            neighbours
                .iter()
                .filter_map(|neighbour| order_index.get(*neighbour).map(|index| *index as f64))
                .collect()
        })
        .unwrap_or_default();
    if positions.is_empty() {
        // Neighbours outside the canvas order sort last, deterministically.
        (true, 0.0)
    } else {
        let mean = positions.iter().sum::<f64>() / positions.len() as f64;
        (false, mean)
    }
}

fn rebuild_order_index(
    layer_members: &BTreeMap<usize, Vec<String>>,
    order_index: &mut HashMap<String, usize>,
) {
    for members in layer_members.values() {
        for (index, id) in members.iter().enumerate() {
            order_index.insert(id.clone(), index);
        }
    }
}

fn recompute_bounds(graph: &mut CanvasGraph) {
    let Some(first) = graph.nodes.first().and_then(|node| {
        graph
            .positions
            .get(&node.id)
            .copied()
            .map(|position| (node.id.clone(), position))
    }) else {
        graph.bounds = GraphRect {
            origin: GraphPoint::ZERO,
            size: GraphSize {
                width: NODE_WIDTH,
                height: NODE_HEIGHT,
            },
        };
        return;
    };
    let mut min = first.1;
    let mut max = GraphPoint {
        x: first.1.x + NODE_WIDTH,
        y: first.1.y + NODE_HEIGHT,
    };
    for node in &graph.nodes {
        let Some(position) = graph.positions.get(&node.id).copied() else {
            continue;
        };
        min.x = min.x.min(position.x);
        min.y = min.y.min(position.y);
        max.x = max.x.max(position.x + NODE_WIDTH);
        max.y = max.y.max(position.y + NODE_HEIGHT);
    }
    for point in edge_paths(graph).into_iter().flatten() {
        min.x = min.x.min(point.x);
        min.y = min.y.min(point.y);
        max.x = max.x.max(point.x);
        max.y = max.y.max(point.y);
    }
    graph.bounds = GraphRect {
        origin: min,
        size: GraphSize {
            width: (max.x - min.x).max(NODE_WIDTH),
            height: (max.y - min.y).max(NODE_HEIGHT),
        },
    };
}

fn edge_paths(graph: &CanvasGraph) -> Vec<Vec<GraphPoint>> {
    let layout = graph.layout;
    let block = layout.transpose(GraphPoint {
        x: NODE_WIDTH,
        y: NODE_HEIGHT,
    });
    let mut multiplicity = HashMap::new();
    for edge in &graph.edges {
        *multiplicity.entry((&edge.from, &edge.to)).or_insert(0usize) += 1;
    }
    let cross_boundary = graph
        .positions
        .values()
        .map(|position| layout.transpose(*position).y)
        .fold(PADDING, f32::min);
    let mut ranks = HashMap::new();
    let mut return_lane = 0;
    graph
        .edges
        .iter()
        .map(|edge| {
            let (Some(from), Some(to)) = (
                graph.positions.get(&edge.from),
                graph.positions.get(&edge.to),
            ) else {
                return Vec::new();
            };
            let key = (&edge.from, &edge.to);
            let rank = ranks.entry(key).or_insert(0usize);
            let count = multiplicity.get(&key).copied().unwrap_or(1);
            let lane = (*rank as f32 + 0.5) / count as f32 - 0.5;
            *rank += 1;
            let port = block.y / 2.0 + lane * (block.y - 12.0).min(60.0);
            let point = |x, y| GraphPoint { x, y };
            let from = layout.transpose(*from);
            let to = layout.transpose(*to);
            let start = point(from.x + block.x, from.y + port);
            let end = point(to.x, to.y + port);
            let departure = point(start.x + 16.0 + lane * 12.0, start.y);
            let arrival = point(end.x - 16.0 - lane * 12.0, end.y);
            let mut route = vec![start, departure];
            if from.x >= to.x {
                // Keep return edges outside the occupied ranks on either axis.
                let bypass = cross_boundary - 18.0 - return_lane as f32 * 7.0;
                return_lane += 1;
                route.extend([point(departure.x, bypass), point(arrival.x, bypass)]);
            }
            route.extend([arrival, end]);
            route
                .into_iter()
                .map(|point| layout.transpose(point))
                .collect()
        })
        .collect()
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
    (point.x - start.x - projection * delta_x).hypot(point.y - start.y - projection * delta_y)
}

/// Interactive read-only canvas state: viewport, selection, hover/focus
/// overlays and box selection. Owned by the hosting surface (the diagram
/// editor item); rendered with `Render`.
pub struct CanvasState {
    pub graph: CanvasGraph,
    pub viewport: sim_runtime::graph::GraphViewport,
    pub selection: BTreeSet<String>,
    /// Node ids of the active inspection highlight (selected node, neighbours,
    /// active paths).
    pub inspected: Vec<String>,
    pub inspected_edges: BTreeSet<usize>,
    pub hover: Option<String>,
    hovered_edge: Option<usize>,
    edge_paths: Vec<Vec<GraphPoint>>,
    pub focus: Option<String>,
    node_focus: BTreeMap<String, gpui::FocusHandle>,
    /// Box selection in graph coordinates, if one is being drawn.
    pub box_selection: Option<(GraphPoint, GraphPoint)>,
    drag_anchor: Option<GraphPoint>,
    drag_origin: GraphPoint,
    /// The canvas element's window origin, captured during paint so input
    /// positions can be translated into canvas coordinates.
    pub canvas_origin: std::sync::Arc<std::sync::Mutex<GraphPoint>>,
    /// The canvas element's painted size, captured during paint so fit and
    /// zoom actions can operate without waiting for a paint pass.
    pub canvas_size: std::sync::Arc<std::sync::Mutex<GraphSize>>,
    /// Shared staging for canvas context menus, set by the owning diagram
    /// item together with the menu handle.
    pub context_state:
        Option<std::sync::Arc<crate::dependency_diagram_menu::DependencyDiagramContextState>>,
    pub context_handle: Option<ui::RightClickMenuHandle<ui::ContextMenu>>,
}

impl CanvasState {
    pub fn new(graph: CanvasGraph) -> Self {
        Self {
            edge_paths: edge_paths(&graph),
            graph,
            viewport: sim_runtime::graph::GraphViewport::default(),
            selection: BTreeSet::new(),
            inspected: Vec::new(),
            inspected_edges: BTreeSet::new(),
            hover: None,
            hovered_edge: None,
            focus: None,
            node_focus: BTreeMap::new(),
            box_selection: None,
            drag_anchor: None,
            drag_origin: GraphPoint::ZERO,
            canvas_origin: std::sync::Arc::new(std::sync::Mutex::new(GraphPoint::ZERO)),
            canvas_size: std::sync::Arc::new(std::sync::Mutex::new(GraphSize {
                width: 0.0,
                height: 0.0,
            })),
            context_state: None,
            context_handle: None,
        }
    }

    pub(crate) fn replace_graph(&mut self, mut graph: CanvasGraph) {
        if graph.layout != self.graph.layout {
            graph.layout = self.graph.layout;
            layout_canvas_graph(&mut graph);
        }
        let contains = |id: &str| graph.positions.contains_key(id);
        self.selection.retain(|id| contains(id));
        self.focus.take_if(|id| !contains(id));
        self.node_focus.retain(|id, _| contains(id));
        self.hover = None;
        self.hovered_edge = None;
        self.inspected.clear();
        self.inspected_edges.clear();
        self.box_selection = None;
        self.end_pan();
        self.edge_paths = edge_paths(&graph);
        self.graph = graph;
    }

    pub(crate) fn set_layout(&mut self, layout: CanvasLayout) -> bool {
        if self.graph.layout == layout {
            return false;
        }
        self.graph.layout = layout;
        layout_canvas_graph(&mut self.graph);
        self.edge_paths = edge_paths(&self.graph);
        self.hover = None;
        self.hovered_edge = None;
        self.box_selection = None;
        self.end_pan();
        let available = *self.canvas_size.lock().expect("canvas size");
        self.fit(available);
        true
    }

    /// Zooms by `factor` around a canvas-relative anchor point.
    pub fn zoom(&mut self, factor: f32, anchor: GraphPoint) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let before = self.viewport.screen_to_graph(anchor);
        self.viewport.scale =
            (self.viewport.scale * factor).clamp(MIN_VIEWPORT_SCALE, MAX_VIEWPORT_SCALE);
        self.viewport.offset = GraphPoint {
            x: anchor.x - before.x * self.viewport.scale,
            y: anchor.y - before.y * self.viewport.scale,
        };
    }

    pub fn pan(&mut self, delta: GraphPoint) {
        self.viewport.offset = self.viewport.offset.translated(delta);
    }

    /// Fits the graph bounds into the available canvas size with padding.
    pub fn fit(&mut self, available: GraphSize) {
        if available.width <= 0.0 || available.height <= 0.0 {
            return;
        }
        let bounds = self.graph.bounds;
        if bounds.size.width <= 0.0 || bounds.size.height <= 0.0 {
            return;
        }
        let scale = ((available.width - 2.0 * PADDING) / bounds.size.width)
            .min((available.height - 2.0 * PADDING) / bounds.size.height)
            .clamp(MIN_VIEWPORT_SCALE, MAX_VIEWPORT_SCALE);
        self.viewport.scale = scale;
        self.viewport.offset = GraphPoint {
            x: PADDING - bounds.origin.x * scale,
            y: PADDING - bounds.origin.y * scale,
        };
    }

    /// Returns the node id under a canvas-relative screen point, if any.
    pub fn node_at(&self, point: GraphPoint) -> Option<String> {
        let point = self.viewport.screen_to_graph(point);
        let size = self.graph.node_size();
        for node in &self.graph.nodes {
            let Some(origin) = self.graph.positions.get(&node.id).copied() else {
                continue;
            };
            let rect = GraphRect { origin, size };
            if rect.contains(point) {
                return Some(node.id.clone());
            }
        }
        None
    }

    fn edge_at(&self, point: GraphPoint) -> Option<usize> {
        let point = self.viewport.screen_to_graph(point);
        let radius = 6.0 / self.viewport.scale;
        self.edge_paths
            .iter()
            .enumerate()
            .filter_map(|(index, path)| {
                let distance = path
                    .iter()
                    .zip(path.iter().skip(1))
                    .map(|(start, end)| point_segment_distance(point, *start, *end))
                    .fold(f32::INFINITY, f32::min);
                (distance <= radius).then_some((index, distance))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
    }

    /// Toggles a node in the selection; without modifiers the selection is
    /// replaced. Returns whether the selection changed.
    pub fn select_node(&mut self, id: &str, additive: bool, toggle: bool) -> bool {
        let changed = if additive && toggle {
            if self.selection.contains(id) {
                self.selection.remove(id);
                true
            } else {
                self.selection.insert(id.to_owned());
                true
            }
        } else if additive {
            self.selection.insert(id.to_owned())
        } else {
            let changed = self.selection.len() != 1 || !self.selection.contains(id);
            if changed {
                self.selection.clear();
                self.selection.insert(id.to_owned());
            }
            changed
        };
        if changed {
            self.focus = Some(id.to_owned());
        }
        changed
    }

    /// Selects the node and pans the viewport so the node sits at the canvas
    /// centre; returns whether the node exists.
    pub fn reveal_node(&mut self, id: &str) -> bool {
        let Some(position) = self.graph.positions.get(id).copied() else {
            return false;
        };
        self.selection.clear();
        self.selection.insert(id.to_owned());
        self.focus = Some(id.to_owned());
        let size = *self.canvas_size.lock().expect("canvas size");
        if size.width > 0.0 && size.height > 0.0 {
            self.viewport.offset = GraphPoint {
                x: size.width / 2.0 - position.x * self.viewport.scale,
                y: size.height / 2.0 - position.y * self.viewport.scale,
            };
        }
        true
    }

    /// Extends the selection to every node inside the current box selection.
    pub fn apply_box_selection(&mut self) -> bool {
        let Some((start, end)) = self.box_selection.take() else {
            return false;
        };
        let rect = GraphRect {
            origin: GraphPoint {
                x: start.x.min(end.x),
                y: start.y.min(end.y),
            },
            size: GraphSize {
                width: (start.x - end.x).abs(),
                height: (start.y - end.y).abs(),
            },
        };
        if start == end {
            return false;
        }
        let size = self.graph.node_size();
        let mut changed = false;
        for node in &self.graph.nodes {
            let Some(origin) = self.graph.positions.get(&node.id).copied() else {
                continue;
            };
            if rect.intersects(GraphRect { origin, size }) && self.selection.insert(node.id.clone())
            {
                changed = true;
            }
        }
        changed
    }

    fn canvas_position(&self, position: gpui::Point<gpui::Pixels>) -> GraphPoint {
        let origin = *self.canvas_origin.lock().expect("canvas origin");
        GraphPoint {
            x: (position.x - px(origin.x)).into(),
            y: (position.y - px(origin.y)).into(),
        }
    }

    fn update_box_selection(&mut self, position: GraphPoint) -> bool {
        if let Some((_, end)) = &mut self.box_selection {
            *end = self.viewport.screen_to_graph(position);
            true
        } else {
            false
        }
    }

    fn finish_pointer_selection(
        &mut self,
        event: &gpui::MouseUpEvent,
        _: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) {
        if self.update_box_selection(self.canvas_position(event.position)) {
            self.apply_box_selection();
            cx.notify();
        }
        self.end_pan();
    }

    /// Begins a viewport pan at a canvas-relative anchor point.
    pub fn begin_pan(&mut self, anchor: GraphPoint) {
        self.drag_anchor = Some(anchor);
        self.drag_origin = self.viewport.offset;
    }

    /// Continues an active pan; returns when no pan is in progress.
    pub fn update_pan(&mut self, position: GraphPoint) {
        let Some(anchor) = self.drag_anchor else {
            return;
        };
        self.viewport.offset = GraphPoint {
            x: self.drag_origin.x + (position.x - anchor.x),
            y: self.drag_origin.y + (position.y - anchor.y),
        };
    }

    pub fn end_pan(&mut self) {
        self.drag_anchor = None;
    }

    pub(crate) fn focused_node(&self, window: &gpui::Window) -> Option<&str> {
        self.node_focus
            .iter()
            .find_map(|(node, handle)| handle.is_focused(window).then_some(node.as_str()))
    }

    #[cfg(test)]
    pub(crate) fn focus_handle_for_node(&self, node: &str) -> Option<gpui::FocusHandle> {
        self.node_focus.get(node).cloned()
    }

    pub fn node_screen_position(&self, id: &str) -> Option<GraphPoint> {
        let position = self.graph.positions.get(id).copied()?;
        Some(self.viewport.graph_to_screen(position))
    }
}

const MIN_VIEWPORT_SCALE: f32 = 0.1;
const MAX_VIEWPORT_SCALE: f32 = 8.0;

impl gpui::Render for CanvasState {
    fn render(
        &mut self,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        use gpui::{
            InteractiveElement as _, ParentElement as _, PathBuilder, Rgba, ScrollWheelEvent,
            StatefulInteractiveElement as _, Styled as _, canvas, point,
        };
        use ui::prelude::*;

        let theme = cx.theme().colors().clone();
        for node in &self.graph.nodes {
            self.node_focus
                .entry(node.id.clone())
                .or_insert_with(|| cx.focus_handle());
        }
        let focused = self.focused_node(window).map(str::to_owned);

        let viewport = self.viewport.clone();
        let node_size = self.graph.node_size();
        let scale = viewport.scale;
        let overlay_node = self.hover.as_ref().or(focused.as_ref());
        let selection = self.selection.clone();
        let inspected: HashSet<String> = self.inspected.iter().cloned().collect();
        let box_selection = self.box_selection;
        let canvas_origin = self.canvas_origin.clone();
        let canvas_size = self.canvas_size.clone();

        // Snapshot the painted geometry so the edge canvas can stroke it.
        let edges: Vec<_> = self
            .graph
            .edges
            .iter()
            .zip(&self.edge_paths)
            .enumerate()
            .map(|(index, (edge, path))| {
                (
                    path.iter()
                        .map(|point| viewport.graph_to_screen(*point))
                        .collect::<Vec<_>>(),
                    edge.variant,
                    self.hovered_edge == Some(index)
                        || self.inspected_edges.contains(&edge.provider_index)
                        || (self.inspected.is_empty()
                            && (selection.contains(&edge.from) || selection.contains(&edge.to))),
                )
            })
            .collect();
        let edge_color = Rgba::from(theme.border);
        let edge_inspected_color = Rgba::from(theme.border_selected);
        let edge_muted_color = Rgba::from(theme.text_muted);
        let nodes: Vec<(
            GraphPoint,
            String,
            String,
            Option<String>,
            CanvasNodeKind,
            bool,
            bool,
            Option<gpui::FocusHandle>,
        )> = self
            .graph
            .nodes
            .iter()
            .map(|node| {
                let position = self
                    .graph
                    .positions
                    .get(&node.id)
                    .copied()
                    .unwrap_or(GraphPoint::ZERO);
                (
                    viewport.graph_to_screen(position),
                    node.id.clone(),
                    node.label.clone(),
                    node.version.clone(),
                    node.kind,
                    selection.contains(&node.id),
                    inspected.contains(&node.id),
                    self.node_focus.get(&node.id).cloned(),
                )
            })
            .collect();
        let box_screen = box_selection.map(|(start, end)| {
            (
                viewport.graph_to_screen(start),
                viewport.graph_to_screen(end),
            )
        });

        let hover_label = self
            .hovered_edge
            .and_then(|index| self.graph.edges.get(index))
            .map(|edge| {
                let source = self
                    .graph
                    .node(&edge.from)
                    .map_or(&edge.from, |node| &node.label);
                let target = self
                    .graph
                    .node(&edge.to)
                    .map_or(&edge.to, |node| &node.label);
                format!(
                    "{} from {source} to {target} ({})",
                    edge.variant.label(),
                    edge.label
                )
            })
            .or_else(|| {
                overlay_node.and_then(|id| self.graph.node(id)).map(|node| {
                    let mut label = node.label.clone();
                    if let Some(version) = &node.version {
                        label.push_str(&format!(" {version}"));
                    }
                    if node.kind == CanvasNodeKind::Unresolved {
                        label.push_str(" (unresolved)");
                    }
                    label
                })
            });

        div()
            .id(gpui::ElementId::Name(
                "groovy-dependency-diagram-canvas".into(),
            ))
            .relative()
            .size_full()
            .flex_1()
            .min_h_0()
            .bg(theme.background)
            .border_1()
            .border_color(theme.border)
            .rounded_b_md()
            .overflow_hidden()
            .aria_label("Dependency diagram")
            .role(gpui::Role::Application)
            .on_scroll_wheel(cx.listener(
                move |state, event: &ScrollWheelEvent, _, cx: &mut gpui::Context<Self>| {
                    let delta = event.delta.pixel_delta(px(20.0));
                    let anchor = state.canvas_position(event.position);
                    if event.modifiers.control || event.modifiers.alt {
                        let vertical: f32 = delta.y.into();
                        let factor = if vertical < 0.0 { 1.1 } else { 1.0 / 1.1 };
                        state.zoom(factor, anchor);
                    } else {
                        state.pan(GraphPoint {
                            x: delta.x.into(),
                            y: delta.y.into(),
                        });
                    }
                    cx.notify();
                },
            ))
            .on_mouse_down(
                gpui::MouseButton::Left,
                cx.listener(
                    move |state, event: &gpui::MouseDownEvent, _, cx: &mut gpui::Context<Self>| {
                        let position = state.canvas_position(event.position);
                        if let Some(node) = state.node_at(position) {
                            let toggle = event.modifiers.platform || event.modifiers.control;
                            state.select_node(&node, event.modifiers.shift || toggle, toggle);
                        } else if event.modifiers.shift {
                            let start = state.viewport.screen_to_graph(position);
                            state.box_selection = Some((start, start));
                        } else {
                            state.begin_pan(position);
                        }
                        cx.notify();
                    },
                ),
            )
            .on_mouse_move(cx.listener(
                move |state, event: &gpui::MouseMoveEvent, _, cx: &mut gpui::Context<Self>| {
                    let position = state.canvas_position(event.position);
                    let dragging = event.pressed_button == Some(gpui::MouseButton::Left);
                    if dragging {
                        if !state.update_box_selection(position) {
                            state.update_pan(position);
                        }
                    }
                    let hover = state.node_at(position);
                    let edge = if hover.is_none() && !dragging {
                        state.edge_at(position)
                    } else {
                        None
                    };
                    if dragging || hover != state.hover || edge != state.hovered_edge {
                        state.hover = hover;
                        state.hovered_edge = edge;
                        cx.notify();
                    }
                },
            ))
            .on_mouse_exit(cx.listener(|state, _, _, cx| {
                let hover = state.hover.take();
                let edge = state.hovered_edge.take();
                if hover.is_some() || edge.is_some() {
                    cx.notify();
                }
            }))
            .on_mouse_up(
                gpui::MouseButton::Left,
                cx.listener(Self::finish_pointer_selection),
            )
            .on_mouse_up_out(
                gpui::MouseButton::Left,
                cx.listener(Self::finish_pointer_selection),
            )
            .on_mouse_down(
                gpui::MouseButton::Right,
                cx.listener(
                    // Stage the context target from the secondary click: an
                    // unselected node targets it, an in-selection click
                    // preserves the selection, blank canvas acts on the view.
                    move |state, event: &gpui::MouseDownEvent, window, cx| {
                        let position = state.canvas_position(event.position);
                        let hit = state.node_at(position);
                        state.focus = hit.clone();
                        let target =
                            crate::dependency_diagram_menu::stage_dependency_diagram_target(
                                hit.as_deref(),
                                &state.selection,
                            );
                        if let Some(context_state) = state.context_state.clone() {
                            context_state.stage(
                                crate::dependency_diagram_menu::DependencyDiagramContextInvocation {
                                    target,
                                    screen_position: GraphPoint {
                                        x: event.position.x.into(),
                                        y: event.position.y.into(),
                                    },
                                },
                            );
                            if let Some(handle) = state.context_handle.clone() {
                                let position = event.position;
                                window.defer(cx, move |window, cx| {
                                    handle.show_at(position, window, cx);
                                });
                            }
                        }
                        cx.notify();
                    },
                ),
            )
            .child(
                canvas(
                    move |_, _, _| {},
                    move |bounds, _, window, _| {
                        *canvas_origin.lock().expect("canvas origin") = GraphPoint {
                            x: bounds.origin.x.into(),
                            y: bounds.origin.y.into(),
                        };
                        *canvas_size.lock().expect("canvas size") = GraphSize {
                            width: bounds.size.width.into(),
                            height: bounds.size.height.into(),
                        };
                        let position = |graph_point: GraphPoint| {
                            point(
                                bounds.origin.x + px(graph_point.x),
                                bounds.origin.y + px(graph_point.y),
                            )
                        };
                        for (points, variant, selected) in &edges {
                            let [from, rest @ .., before_tip, tip] = points.as_slice() else {
                                continue;
                            };
                            let stroke_color = if *selected {
                                edge_inspected_color
                            } else {
                                match variant {
                                    CanvasEdgeVariant::Dependency
                                    | CanvasEdgeVariant::Constraint => edge_color,
                                    CanvasEdgeVariant::Unresolved | CanvasEdgeVariant::Omitted => {
                                        edge_muted_color
                                    }
                                }
                            };
                            let width = if *selected || *variant != CanvasEdgeVariant::Dependency {
                                2.0
                            } else {
                                1.5
                            };
                            let mut builder = PathBuilder::stroke(px(width));
                            if let Some((dash, gap)) = variant.dash_pattern() {
                                builder = builder.dash_array(&[px(dash), px(gap)]);
                            }
                            builder.move_to(position(*from));
                            for next in rest.iter().chain([before_tip, tip]) {
                                builder.line_to(position(*next));
                            }
                            if let Ok(path) = builder.build() {
                                window.paint_path(path, stroke_color);
                            }
                            let direction = GraphPoint {
                                x: tip.x - before_tip.x,
                                y: tip.y - before_tip.y,
                            };
                            let length = direction.x.hypot(direction.y);
                            if length <= f32::EPSILON {
                                continue;
                            }
                            let scale = length.min(10.0) / length;
                            let offset_x = direction.x * scale;
                            let offset_y = direction.y * scale;
                            let base = GraphPoint {
                                x: tip.x - offset_x,
                                y: tip.y - offset_y,
                            };
                            let mut builder = PathBuilder::fill();
                            builder.move_to(position(*tip));
                            builder.line_to(position(GraphPoint {
                                x: base.x - offset_y * 0.4,
                                y: base.y + offset_x * 0.4,
                            }));
                            builder.line_to(position(GraphPoint {
                                x: base.x + offset_y * 0.4,
                                y: base.y - offset_x * 0.4,
                            }));
                            builder.close();
                            if let Ok(path) = builder.build() {
                                window.paint_path(path, stroke_color);
                            }
                        }
                        if let Some((start, end)) = box_screen {
                            let rect = GraphRect {
                                origin: GraphPoint {
                                    x: start.x.min(end.x),
                                    y: start.y.min(end.y),
                                },
                                size: GraphSize {
                                    width: (start.x - end.x).abs(),
                                    height: (start.y - end.y).abs(),
                                },
                            };
                            let mut builder = PathBuilder::stroke(px(1.0));
                            builder.move_to(point(
                                bounds.origin.x + px(rect.origin.x),
                                bounds.origin.y + px(rect.origin.y),
                            ));
                            builder.line_to(point(
                                bounds.origin.x + px(rect.origin.x + rect.size.width),
                                bounds.origin.y + px(rect.origin.y),
                            ));
                            builder.line_to(point(
                                bounds.origin.x + px(rect.origin.x + rect.size.width),
                                bounds.origin.y + px(rect.origin.y + rect.size.height),
                            ));
                            builder.line_to(point(
                                bounds.origin.x + px(rect.origin.x),
                                bounds.origin.y + px(rect.origin.y + rect.size.height),
                            ));
                            builder.line_to(point(
                                bounds.origin.x + px(rect.origin.x),
                                bounds.origin.y + px(rect.origin.y),
                            ));
                            if let Ok(path) = builder.build() {
                                window.paint_path(path, edge_color);
                            }
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
            .children(nodes.into_iter().map(
                |(screen, id, label, version, kind, selected, inspected, focus_handle)| {
                    let width = node_size.width * scale;
                    let height = node_size.height * scale;
                    let background = if selected {
                        theme.element_selected
                    } else if inspected {
                        theme.element_hover
                    } else if kind == CanvasNodeKind::Root {
                        theme.elevated_surface_background
                    } else {
                        theme.element_background
                    };
                    let text_color = if kind == CanvasNodeKind::Unresolved {
                        theme.text_muted
                    } else {
                        theme.text
                    };
                    let border = if selected || inspected || focused.as_ref() == Some(&id) {
                        theme.border_selected
                    } else {
                        theme.border
                    };
                    let title = match &version {
                        Some(version) => format!("{label} {version}"),
                        None => label,
                    };
                    let accessible = format!(
                        "Dependency {}{}",
                        title,
                        if kind == CanvasNodeKind::Unresolved {
                            " (unresolved)".to_owned()
                        } else {
                            String::new()
                        }
                    );
                    div()
                        .id(gpui::ElementId::Name(
                            format!("groovy-diagram-node-{id}").into(),
                        ))
                        .absolute()
                        .left(px(screen.x))
                        .top(px(screen.y))
                        .w(px(width))
                        .h(px(height))
                        .bg(background)
                        .border_1()
                        .border_color(border)
                        .rounded_md()
                        .px_2()
                        .flex()
                        .items_center()
                        .overflow_hidden()
                        .role(gpui::Role::Button)
                        .aria_label(accessible)
                        .when_some(focus_handle, |node, focus| {
                            node.track_focus(&focus).tab_index(0_isize)
                        })
                        .on_click(cx.listener(move |state, event: &gpui::ClickEvent, _, cx| {
                            if matches!(event, gpui::ClickEvent::Keyboard(_))
                                && state.select_node(&id, false, false)
                            {
                                cx.notify();
                            }
                        }))
                        .child(
                            div()
                                .text_size(px(11.0 * scale.max(0.75)))
                                .text_color(text_color)
                                .truncate()
                                .child(title),
                        )
                },
            ))
            .when_some(hover_label, |canvas, label| {
                canvas.child(
                    div()
                        .id("groovy-dependency-details")
                        .absolute()
                        .bottom_2()
                        .left_2()
                        .px_2()
                        .py_1()
                        .rounded_md()
                        .bg(theme.panel_overlay_background)
                        .border_1()
                        .border_color(theme.border)
                        .text_size(px(12.0))
                        .text_color(theme.text)
                        .role(gpui::Role::Status)
                        .aria_label("Dependency details")
                        .aria_value(label.clone())
                        .child(label),
                )
            })
    }
}

pub(crate) fn svg_escape(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '&' => "&amp;".to_owned(),
            '<' => "&lt;".to_owned(),
            '>' => "&gt;".to_owned(),
            '"' => "&quot;".to_owned(),
            other => other.to_string(),
        })
        .collect()
}

/// Renders a deterministic SVG from the same node/edge tuples the canvas
/// paints. This is the Export Image path that does not depend on a
/// rasterizing surface being available at implementation time; it is distinct
/// from the Mermaid Copy Diagram Source payload.
pub fn canvas_graph_svg(graph: &CanvasGraph) -> String {
    let node_size = graph.node_size();
    let bounds = graph.bounds;
    let padding = node_size.width / 2.0;
    let width = (bounds.size.width + 2.0 * padding).max(node_size.width);
    let height = (bounds.size.height + 2.0 * padding).max(node_size.height);
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.0}\" height=\"{height:.0}\" viewBox=\"0 0 {width:.0} {height:.0}\">\n"
    );
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#1a1a1f\"/>\n");
    svg.push_str("<defs><marker id=\"dependency-arrow\" viewBox=\"0 0 10 8\" refX=\"10\" refY=\"4\" markerWidth=\"8\" markerHeight=\"8\" orient=\"auto\"><path d=\"M 0 0 L 10 4 L 0 8 Z\" fill=\"#7d7d87\"/></marker></defs>\n");
    let origin = GraphPoint {
        x: bounds.origin.x - padding,
        y: bounds.origin.y - padding,
    };
    for (edge, path) in graph.edges.iter().zip(edge_paths(graph)) {
        if path.is_empty() {
            continue;
        }
        let points = path
            .iter()
            .map(|point| format!("{:.1},{:.1}", point.x - origin.x, point.y - origin.y))
            .collect::<Vec<_>>()
            .join(" ");
        let dash = edge
            .variant
            .dash_pattern()
            .map_or_else(String::new, |(dash, gap)| {
                format!(" stroke-dasharray=\"{dash} {gap}\"")
            });
        svg.push_str(&format!(
            "<polyline points=\"{points}\" fill=\"none\" stroke=\"#7d7d87\" stroke-width=\"1.5\" marker-end=\"url(#dependency-arrow)\"{dash}><title>{title}</title></polyline>\n",
            title = svg_escape(&format!("{}: {}", edge.variant.label(), edge.label)),
        ));
    }
    for node in &graph.nodes {
        let Some(origin_node) = graph.positions.get(&node.id).copied() else {
            continue;
        };
        let fill = match node.kind {
            CanvasNodeKind::Root => "#2f8f5b",
            CanvasNodeKind::Module | CanvasNodeKind::Project => "#3b6ea5",
            CanvasNodeKind::Unresolved => "#a5573b",
        };
        let label = match &node.version {
            Some(version) => format!("{} {version}", node.label),
            None => node.label.clone(),
        };
        svg.push_str(&format!(
            "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{width:.1}\" height=\"{height:.1}\" rx=\"4\" fill=\"{fill}\"/>\n<text x=\"{text_x:.1}\" y=\"{text_y:.1}\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#e8e8ee\">{label}</text>\n",
            x = origin_node.x - origin.x,
            y = origin_node.y - origin.y,
            width = node_size.width,
            height = node_size.height,
            text_x = origin_node.x - origin.x + 6.0,
            text_y = origin_node.y - origin.y + node_size.height - 6.0,
            label = svg_escape(&label),
        ));
    }
    svg.push_str("</svg>\n");
    svg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradle_component_json(
        id: &str,
        module: &str,
    ) -> project::jvm_project::GradleDependencyComponent {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "kind": "module",
            "group": "org.example",
            "module": module,
            "version": "1.0",
            "buildTreePath": null,
            "projectPath": null,
            "conflictResolution": false,
            "forced": false,
            "selectedByRule": false,
            "compositeSubstitution": false,
            "constrained": false,
            "selectionCauses": ["REQUESTED"]
        }))
        .expect("component fixture")
    }

    fn gradle_edge_json(
        from: &str,
        module: &str,
        selected: &str,
    ) -> project::jvm_project::GradleDependencyEdge {
        serde_json::from_value(
            serde_json::json!({
                "from": from,
                "requested": {"kind": "module", "group": "org.example", "module": module, "version": "1.0", "requiredVersion": "1.0", "preferredVersion": "1.0", "strictVersion": "", "rejectedVersions": []},
                "constraint": false,
                "originKnown": true,
                "resolved": true,
                "selected": selected,
                "selectedVariant": null,
                "failureCategory": null
            }),
        )
        .expect("edge fixture")
    }

    fn gradle_fixture() -> GradleResolutionGraph {
        serde_json::from_str(
            r#"{
                "configuration": ":core",
                "rootComponent": "root",
                "structuralCoverageKnown": true,
                "resolutionComplete": true,
                "originsKnown": true,
                "artifactsKnown": true,
                "components": [
                    {"id": "root", "kind": "project", "group": null, "module": null, "version": null, "buildTreePath": ":", "projectPath": ":", "conflictResolution": false, "forced": false, "selectedByRule": false, "compositeSubstitution": false, "constrained": false, "selectionCauses": ["ROOT"]},
                    {"id": "c1", "kind": "module", "group": "org.example", "module": "left", "version": "1.0", "buildTreePath": null, "projectPath": null, "conflictResolution": false, "forced": false, "selectedByRule": false, "compositeSubstitution": false, "constrained": false, "selectionCauses": ["REQUESTED"]},
                    {"id": "c2", "kind": "module", "group": "org.example", "module": "right", "version": "2.0", "buildTreePath": null, "projectPath": null, "conflictResolution": false, "forced": false, "selectedByRule": false, "compositeSubstitution": false, "constrained": false, "selectionCauses": ["REQUESTED"]},
                    {"id": "c3", "kind": "module", "group": "org.example", "module": "shared", "version": "3.0", "buildTreePath": null, "projectPath": null, "conflictResolution": false, "forced": false, "selectedByRule": false, "compositeSubstitution": false, "constrained": false, "selectionCauses": ["REQUESTED"]}
                ],
                "edges": [
                    {"from": "root", "requested": {"kind": "module", "group": "org.example", "module": "left", "version": "1.0", "requiredVersion": "1.0", "preferredVersion": "1.0", "strictVersion": "", "rejectedVersions": []}, "constraint": false, "originKnown": true, "resolved": true, "selected": "c1", "selectedVariant": null, "failureCategory": null},
                    {"from": "root", "requested": {"kind": "module", "group": "org.example", "module": "right", "version": "2.0", "requiredVersion": "2.0", "preferredVersion": "2.0", "strictVersion": "", "rejectedVersions": []}, "constraint": false, "originKnown": true, "resolved": true, "selected": "c2", "selectedVariant": null, "failureCategory": null},
                    {"from": "c1", "requested": {"kind": "module", "group": "org.example", "module": "shared", "version": "3.0", "requiredVersion": "3.0", "preferredVersion": "3.0", "strictVersion": "", "rejectedVersions": []}, "constraint": false, "originKnown": true, "resolved": true, "selected": "c3", "selectedVariant": null, "failureCategory": null},
                    {"from": "c2", "requested": {"kind": "module", "group": "org.example", "module": "shared", "version": "3.0", "requiredVersion": "3.0", "preferredVersion": "3.0", "strictVersion": "", "rejectedVersions": []}, "constraint": false, "originKnown": true, "resolved": true, "selected": "c3", "selectedVariant": null, "failureCategory": null},
                    {"from": "c3", "requested": {"kind": "module", "group": "org.example", "module": "missing", "version": "9.0", "requiredVersion": "9.0", "preferredVersion": "9.0", "strictVersion": "", "rejectedVersions": []}, "constraint": false, "originKnown": false, "resolved": false, "selected": null, "selectedVariant": null, "failureCategory": "missing"}
                ]
            }"#,
        )
        .expect("gradle fixture")
    }

    #[test]
    fn gradle_adapter_preserves_shared_descendants_and_unresolved_targets() {
        let graph = canvas_graph_from_gradle(&gradle_fixture());
        assert_eq!(graph.nodes.len(), 5, "4 components + 1 unresolved target");
        let shared = graph
            .nodes
            .iter()
            .find(|node| node.label.ends_with("shared"));
        assert_eq!(
            shared.map(|node| node.version.as_deref()),
            Some(Some("3.0"))
        );
        let unresolved = graph
            .nodes
            .iter()
            .find(|node| node.kind == CanvasNodeKind::Unresolved)
            .expect("unresolved node");
        assert_eq!(unresolved.label, "org.example:missing");
        assert_eq!(unresolved.version.as_deref(), Some("9.0"));
        assert_eq!(
            graph.edges.iter().filter(|edge| edge.to == "c3").count(),
            2,
            "shared descendant reached by two parents"
        );
        assert!(
            graph
                .edges
                .iter()
                .any(|edge| edge.variant == CanvasEdgeVariant::Unresolved)
        );
        assert!(
            graph
                .nodes
                .iter()
                .find(|node| node.kind == CanvasNodeKind::Root)
                .is_some()
        );
    }

    #[test]
    fn cycle_and_parallel_edges_keep_their_own_routes_and_provider_identity() {
        let mut fixture = gradle_fixture();
        let first = fixture.edges.first().expect("root edge").clone();
        let mut constraint = first.clone();
        constraint.constraint = true;
        fixture.edges.push(constraint);
        let mut back = first;
        back.from = "c3".into();
        back.selected = Some("root".into());
        fixture.edges.push(back.clone());
        back.selected = Some("c3".into());
        fixture.edges.push(back);

        let graph = canvas_graph_from_gradle(&fixture);
        let state = CanvasState::new(graph);
        let paths = &state.edge_paths;
        assert_eq!(*paths, edge_paths(&canvas_graph_from_gradle(&fixture)));
        let root = state.graph.positions["root"];
        let left = state.graph.positions["c1"];
        let shared = state.graph.positions["c3"];
        assert!(root.x < left.x && left.x < shared.x);

        let edge_index = |variant, from: &str, to: &str| {
            state
                .graph
                .edges
                .iter()
                .position(|edge| edge.variant == variant && edge.from == from && edge.to == to)
                .expect("routed fixture edge")
        };
        let dependency = edge_index(CanvasEdgeVariant::Dependency, "root", "c1");
        let constraint = edge_index(CanvasEdgeVariant::Constraint, "root", "c1");
        assert_ne!(
            state.graph.edges[dependency].id,
            state.graph.edges[constraint].id
        );
        for index in [dependency, constraint] {
            let [start, _, _, end] = paths[index].as_slice() else {
                panic!("forward route")
            };
            assert_eq!(start.x, root.x + NODE_WIDTH);
            assert_eq!(end.x, left.x);
            assert_eq!(
                state.edge_at(GraphPoint {
                    x: start.x + 8.0,
                    y: start.y
                }),
                Some(index)
            );
        }
        for (target, position) in [("root", root), ("c3", shared)] {
            let index = edge_index(CanvasEdgeVariant::Dependency, "c3", target);
            let [_, _, upper_right, upper_left, _, end] = paths[index].as_slice() else {
                panic!("return route")
            };
            assert_eq!(end.x, position.x);
            assert!(upper_right.y < root.y);
            assert_eq!(upper_left.y, upper_right.y);
            assert!(
                paths[index]
                    .iter()
                    .all(|point| state.graph.bounds.contains(*point))
            );
        }

        let svg = canvas_graph_svg(&state.graph);
        assert_eq!(svg.matches("<polyline ").count(), state.graph.edges.len());
        assert!(svg.contains("stroke-dasharray=\"3 3\""));
        assert!(svg.contains("stroke-dasharray=\"7 4\""));
        assert!(svg.contains("marker-end=\"url(#dependency-arrow)\""));
        assert!(svg.contains("org.example:missing 9.0"));
    }

    #[test]
    fn layout_is_deterministic_and_layered() {
        let first = canvas_graph_from_gradle(&gradle_fixture());
        let mut second = canvas_graph_from_gradle(&gradle_fixture());
        layout_canvas_graph(&mut second);
        assert_eq!(first.positions, second.positions);
        let root_position = first.positions["root"];
        let left_position = first.positions["c1"];
        let shared_position = first.positions["c3"];
        assert!(
            shared_position.x > left_position.x && left_position.x > root_position.x,
            "dependencies lay out left-to-right: {root_position:?} {left_position:?} {shared_position:?}"
        );
    }

    #[test]
    fn vertical_layout_preserves_selection_and_survives_graph_replacement() {
        let mut state = canvas_state_fixture();
        assert!(state.select_node("c1", false, false));
        state.inspected_edges.insert(0);
        let horizontal = state.graph.positions.clone();
        assert!(state.set_layout(CanvasLayout::TopToBottom));
        let vertical = state.graph.positions.clone();
        let root = vertical["root"];
        let left = vertical["c1"];
        assert!(root.y < left.y);
        let route = state.edge_paths.first().expect("root route");
        assert_eq!(route.first().expect("departure").y, root.y + NODE_HEIGHT);
        assert_eq!(route.last().expect("arrival").y, left.y);
        assert!(state.selection.contains("c1"));
        assert!(state.inspected_edges.contains(&0));
        assert!(!state.set_layout(CanvasLayout::TopToBottom));

        state.replace_graph(canvas_graph_from_gradle(&gradle_fixture()));
        assert_eq!(state.graph.positions, vertical);
        assert!(state.selection.contains("c1"));
        assert!(state.set_layout(CanvasLayout::LeftToRight));
        assert_eq!(state.graph.positions, horizontal);
    }

    #[test]
    fn budgets_truncate_deterministically_and_mark_truncation() {
        let mut fixture = gradle_fixture();
        for index in 0..80 {
            let id = format!("c{index}");
            let module = format!("m{index}");
            fixture.components.push(gradle_component_json(&id, &module));
            fixture.edges.push(gradle_edge_json("root", &module, &id));
        }
        let graph = canvas_graph_from_gradle(&fixture);
        assert!(graph.truncated);
        assert!(graph.nodes.len() <= MAX_CANVAS_NODES);
        assert!(graph.edges.len() <= MAX_CANVAS_EDGES);
    }

    #[test]
    fn maven_adapter_preserves_occurrence_identity_and_omitted_edges() {
        let graph: MavenDependencyGraph = serde_json::from_str(
            r#"{
                "version": 2,
                "occurrencesKnown": true,
                "rootNode": "n0",
                "rootComponent": "c0",
                "nodes": [
                    {"id": "n0", "component": "c0"},
                    {"id": "n1", "component": "c1"},
                    {"id": "n2", "component": "c1"},
                    {"id": "n3", "component": "c2"}
                ],
                "components": [
                    {"id": "c0", "groupId": "org.example", "artifactId": "app", "extension": "jar", "classifier": "", "version": "1.0"},
                    {"id": "c1", "groupId": "org.example", "artifactId": "lib", "extension": "jar", "classifier": "", "version": "2.0"},
                    {"id": "c2", "groupId": "org.example", "artifactId": "util", "extension": "jar", "classifier": "", "version": "3.0"}
                ],
                "edges": [
                    {"from": "c0", "fromNode": "n0", "targetNode": "n1", "selectedNode": "n1", "position": 0, "component": "c1", "selected": "c1", "omitted": false, "direct": true, "versionConstraint": null, "premanagedVersion": null, "premanagedScope": null, "premanagedOptional": null, "managedBits": 0, "originalScope": "compile", "originalOptional": null, "scope": "compile", "optional": false, "exclusions": [], "premanagedExclusions": null},
                    {"from": "c0", "fromNode": "n0", "targetNode": "n2", "selectedNode": "n2", "position": 1, "component": "c1", "selected": "c1", "omitted": true, "direct": false, "versionConstraint": "2.0", "premanagedVersion": null, "premanagedScope": null, "premanagedOptional": null, "managedBits": 0, "originalScope": "runtime", "originalOptional": null, "scope": "runtime", "optional": false, "exclusions": [], "premanagedExclusions": null},
                    {"from": "c1", "fromNode": "n1", "targetNode": "n3", "selectedNode": "n3", "position": 0, "component": "c2", "selected": "c2", "omitted": false, "direct": false, "versionConstraint": null, "premanagedVersion": null, "premanagedScope": null, "premanagedOptional": null, "managedBits": 0, "originalScope": "compile", "originalOptional": null, "scope": "compile", "optional": true, "exclusions": [], "premanagedExclusions": null}
                ],
                "unavailableDescriptors": [],
                "collectionComplete": true,
                "structuralCoverageKnown": true,
                "problem": null,
                "originsKnown": true,
                "artifactsKnown": true,
                "offline": true,
                "cachedOnly": false,
                "cycleCount": 0,
                "cycles": [],
                "cycleOccurrencesKnown": true
            }"#,
        )
        .expect("maven fixture");
        let canvas = canvas_graph_from_maven(&graph, |_| true);
        assert_eq!(canvas.nodes.len(), 4, "one node per occurrence");
        let occurrences = canvas
            .nodes
            .iter()
            .filter(|node| node.component_id == "c1")
            .count();
        assert_eq!(
            occurrences, 2,
            "repeated artifacts stay distinct occurrences"
        );
        assert!(
            canvas
                .edges
                .iter()
                .any(|edge| edge.variant == CanvasEdgeVariant::Omitted)
        );
        let root = canvas
            .nodes
            .iter()
            .find(|node| node.kind == CanvasNodeKind::Root);
        assert_eq!(root.map(|node| node.id.as_str()), Some("n0"));
        assert!(
            canvas
                .edges
                .iter()
                .any(|edge| edge.label.contains("optional")),
            "optional scope is preserved on the edge label"
        );
    }

    fn canvas_state_fixture() -> CanvasState {
        CanvasState::new(canvas_graph_from_gradle(&gradle_fixture()))
    }

    #[test]
    fn zoom_keeps_the_anchor_fixed_and_respects_scale_bounds() {
        let mut state = canvas_state_fixture();
        let anchor = GraphPoint { x: 400.0, y: 300.0 };
        state.viewport.offset = GraphPoint { x: 100.0, y: 50.0 };
        let before = state.viewport.screen_to_graph(anchor);
        state.zoom(2.0, anchor);
        let after = state.viewport.screen_to_graph(anchor);
        assert!(
            (before.x - after.x).abs() < 0.01 && (before.y - after.y).abs() < 0.01,
            "the anchor stays on the same graph point: {before:?} vs {after:?}"
        );
        assert_eq!(state.viewport.scale, 2.0);
        state.zoom(100.0, anchor);
        assert_eq!(state.viewport.scale, MAX_VIEWPORT_SCALE);
        state.zoom(0.00001, anchor);
        assert_eq!(state.viewport.scale, MIN_VIEWPORT_SCALE);
    }

    #[test]
    fn fit_centers_the_graph_bounds_with_padding() {
        let mut state = canvas_state_fixture();
        state.fit(GraphSize {
            width: 800.0,
            height: 600.0,
        });
        let top_left = state.viewport.graph_to_screen(state.graph.bounds.origin);
        let bottom_right = state.viewport.graph_to_screen(GraphPoint {
            x: state.graph.bounds.origin.x + state.graph.bounds.size.width,
            y: state.graph.bounds.origin.y + state.graph.bounds.size.height,
        });
        assert!(top_left.x > 0.0 && top_left.y > 0.0);
        assert!(bottom_right.x < 800.0 && bottom_right.y < 600.0);
        assert!(top_left.x <= PADDING + 0.01);
    }

    #[test]
    fn node_at_respects_screen_to_graph_translation() {
        let mut state = canvas_state_fixture();
        state.viewport.offset = GraphPoint { x: 40.0, y: 60.0 };
        state.viewport.scale = 1.5;
        let root_graph = state.graph.positions["root"];
        let screen = state.viewport.graph_to_screen(root_graph);
        let hit = state.node_at(screen.translated(GraphPoint { x: 5.0, y: 5.0 }));
        assert_eq!(hit.as_deref(), Some("root"));
        let miss = state.node_at(screen.translated(GraphPoint {
            x: -1000.0,
            y: -1000.0,
        }));
        assert!(miss.is_none());
    }

    #[test]
    fn selection_replaces_without_modifiers_and_extends_or_toggles_with_them() {
        let mut state = canvas_state_fixture();
        assert!(state.select_node("root", false, false));
        assert_eq!(state.selection.len(), 1);
        assert!(!state.select_node("root", false, false));
        assert!(state.selection.contains("root"));
        assert!(state.select_node("c1", true, false));
        assert_eq!(state.selection.len(), 2, "shift extends the selection");
        assert!(state.select_node("c1", true, true));
        assert_eq!(state.selection.len(), 1, "cmd toggles a node off");
        assert!(state.selection.contains("root"));
        assert!(state.select_node("c2", false, false));
        assert_eq!(
            state.selection.len(),
            1,
            "plain click replaces the selection"
        );
    }

    #[test]
    fn box_selection_extends_the_selection_by_graph_rect() {
        let mut state = canvas_state_fixture();
        let first = state.graph.positions["root"];
        let second = state.graph.positions["c1"];
        state.box_selection = Some((
            GraphPoint {
                x: first.x - 1.0,
                y: first.y - 1.0,
            },
            GraphPoint {
                x: second.x + 10.0,
                y: second.y + 10.0,
            },
        ));
        assert!(state.apply_box_selection());
        assert!(state.selection.contains("root"));
        assert!(state.selection.contains("c1"));
        assert!(state.box_selection.is_none());
    }

    #[gpui::test]
    fn canvas_pointer_selection_keeps_the_viewport_and_toggles_exact_nodes(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let (canvas, cx) = cx.add_window_view(|_, _| canvas_state_fixture());
        cx.update(|window, cx| window.draw(cx).clear(cx));
        let (first, second, start, end) = canvas.read_with(cx, |state, _| {
            let origin = *state.canvas_origin.lock().expect("canvas origin");
            let point = |node: &str, horizontal: f32, vertical: f32| {
                let node = state.node_screen_position(node).expect("fixture node");
                gpui::point(
                    px(origin.x + node.x + horizontal),
                    px(origin.y + node.y + vertical),
                )
            };
            (
                point("root", 20.0, 10.0),
                point("c1", 20.0, 10.0),
                point("c1", NODE_WIDTH + 10.0, 10.0),
                point("c1", NODE_WIDTH - 10.0, 20.0),
            )
        });
        cx.simulate_click(first, gpui::Modifiers::default());
        cx.simulate_click(first, gpui::Modifiers::default());
        let toggle = gpui::Modifiers {
            platform: true,
            ..Default::default()
        };
        cx.simulate_click(second, toggle);
        assert_eq!(
            canvas.read_with(cx, |state, _| state.selection.clone()),
            BTreeSet::from(["root".into(), "c1".into()])
        );
        cx.simulate_click(second, toggle);
        let extend = gpui::Modifiers {
            shift: true,
            ..Default::default()
        };
        cx.simulate_mouse_down(start, gpui::MouseButton::Left, extend);
        cx.simulate_mouse_move(end, gpui::MouseButton::Left, extend);
        assert!(canvas.read_with(cx, |state, _| state.box_selection.is_some()));
        cx.simulate_mouse_up(end, gpui::MouseButton::Left, extend);
        canvas.read_with(cx, |state, _| {
            assert_eq!(
                state.selection,
                BTreeSet::from(["root".into(), "c1".into()])
            );
            assert_eq!(state.viewport.offset, GraphPoint::ZERO);
            assert!(state.box_selection.is_none());
        });
    }
}
