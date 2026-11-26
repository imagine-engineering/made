//! The dependency analyzer editor item: a searchable list/tree over admitted
//! Groovy provider dependency facts with a Usages pane. Rows are derived from
//! the Dependencies panel's admitted snapshot at open (and scope-change) time
//! and never query a provider; filters, group IDs, presentation, selection and
//! the Usages pane live on the item so they survive tab switches.

use crate::dependencies_panel::{
    DependencyGraphReceipt, GraphQueryTarget, GraphSelection, MavenGraphScope,
    ShowDependencyRowMenu, current_dependency_graph_receipt, derive_dependency_analyzer,
};
use crate::dependency_diagram_item::{
    DependencyDiagramProvider, DependencyDiagramScope, DependencyDiagramTarget, DependencySearch,
    DependencySearchEntry,
};
use gpui::{
    App, AppContext as _, Context, EventEmitter, FocusHandle, Focusable, FontWeight, Pixels,
    Render, ScrollStrategy, SharedString, actions, anchored, deferred,
};
use language_tools::language_tool_tree::{
    LanguageToolNode, LanguageToolNodeId, LanguageToolProviderStatus, LanguageToolSnapshot,
    LanguageToolTreeHost, SelectFirst, SelectFirstChild, SelectLast, SelectNext, SelectParent,
    SelectPrevious, ToggleExpanded, language_tool_tree,
};
use project::jvm_project::{
    GradleDependencyComponent, GradleDependencyEdge, GradleDependencySelector,
    JdtlsProjectSnapshot, MavenDependencyComponent, MavenDependencyEdge, MavenDependencyNode,
    MavenUnavailableDescriptor,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use ui::prelude::FluentBuilder as _;
use workspace::Workspace;

actions!(groovy_dependencies_panel, [AnalyzeDependencies]);

/// Applies a scope choice from the analyzer scope picker.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct SelectDependencyAnalyzerScope {
    pub label: String,
}

/// Applies a conflicts/unresolved fact-filter choice.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct SelectDependencyAnalyzerFilter {
    pub filter: DependencyAnalyzerFilter,
}

/// Toggles group IDs on artifact and occurrence labels.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct ToggleDependencyAnalyzerGroupIds;

/// Switches between the tree and flat presentations.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct ToggleDependencyAnalyzerPresentation;

/// Expands every admitted analyzer row.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct ExpandDependencyAnalyzerRows;

/// Collapses every admitted analyzer row.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct CollapseDependencyAnalyzerRows;

/// Shows the analyzer row's dependency in the diagram tab for the analyzer's
/// own module and configuration scope, revealing the row's node when the
/// canvas admits it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct ShowInDependencyDiagram {
    pub node_key: String,
}

/// Navigates to the Gradle build script that declares the row's dependency;
/// the row's admitted reverse edge must report a trustworthy origin.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema, gpui::Action)]
#[action(namespace = groovy_dependencies_panel)]
#[serde(deny_unknown_fields)]
pub struct GoToDependencyDeclaration {
    pub declaring: String,
}

/// Conflicts/unresolved fact filter over the admitted analyzer facts.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DependencyAnalyzerFilter {
    All,
    Conflicts,
    Unresolved,
}

impl DependencyAnalyzerFilter {
    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All dependencies",
            Self::Conflicts => "Conflicts",
            Self::Unresolved => "Unresolved",
        }
    }
}

/// Tree/list presentation switch over the same admitted facts.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DependencyAnalyzerPresentation {
    Tree,
    Flat,
}

/// The admitted facts the analyzer presents. The extraction keeps the item
/// bounded to graph facts (never a second store) and keeps exactly the graph
/// the provider admitted at derivation time.
#[derive(Clone, Debug)]
pub enum DependencyAnalyzerGraph {
    Gradle {
        configuration: String,
        components: Vec<GradleDependencyComponent>,
        edges: Vec<GradleDependencyEdge>,
    },
    Maven {
        scope: MavenGraphScope,
        nodes: Vec<MavenDependencyNode>,
        components: Vec<MavenDependencyComponent>,
        edges: Vec<MavenDependencyEdge>,
        unavailable_descriptors: Vec<MavenUnavailableDescriptor>,
    },
}

impl DependencyAnalyzerGraph {
    fn clear(&mut self) {
        match self {
            Self::Gradle {
                components, edges, ..
            } => {
                *components = Vec::new();
                *edges = Vec::new();
            }
            Self::Maven {
                nodes,
                components,
                edges,
                unavailable_descriptors,
                ..
            } => {
                *nodes = Vec::new();
                *components = Vec::new();
                *edges = Vec::new();
                *unavailable_descriptors = Vec::new();
            }
        }
    }
}

/// One analyzer derivation: admitted facts plus the scope the picker applied.
pub struct DependencyAnalyzerDerivation {
    pub target: DependencyDiagramTarget,
    pub graph: DependencyAnalyzerGraph,
    pub scope: DependencyDiagramScope,
    pub scope_options: Vec<String>,
}

/// Where the analyzer's facts came from, so scope changes re-derive from the
/// same admitted selection without a provider query.
#[derive(Clone)]
pub struct DependencyAnalyzerSource {
    pub project: gpui::Entity<project::Project>,
    pub selection: GraphSelection,
    pub panel: gpui::WeakEntity<crate::dependencies_panel::DependenciesPanel>,
}

pub enum DependencyAnalyzerEvent {}

pub struct DependencyAnalyzerItem {
    host: LanguageToolTreeHost,
    derivation: DependencyAnalyzerDerivation,
    source: Option<DependencyAnalyzerSource>,
    receipt: Option<DependencyGraphReceipt>,
    _panel_subscription: Option<gpui::Subscription>,
    search_handle: ui::PopoverMenuHandle<picker::Picker<DependencySearch>>,
    filter: DependencyAnalyzerFilter,
    presentation: DependencyAnalyzerPresentation,
    show_group_ids: bool,
    selected_key: Option<String>,
    selected_outside_tree: bool,
    usages: Vec<AnalyzerUsage>,
    usages_truncated: bool,
    usages_scroll: gpui::ScrollHandle,
    usage_focus_selection: Option<String>,
    usage_focus_invalidated: bool,
    usage_focus_handles: Vec<FocusHandle>,
    last_focused_usage: Option<usize>,
    notice: Option<String>,
    /// Staged row context menu, anchored at the invoking position until
    /// dismissed; shared by the tree rows and the Usages pane rows.
    row_context_menu: Option<(
        gpui::Entity<ui::ContextMenu>,
        gpui::Point<Pixels>,
        gpui::Subscription,
    )>,
    focus_handle: FocusHandle,
}

impl EventEmitter<DependencyAnalyzerEvent> for DependencyAnalyzerItem {}

impl Focusable for DependencyAnalyzerItem {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

/// Occurrence rows are bounded like the diagram's node budget so an oversized
/// admitted graph cannot grow the item without an explicit boundary.
const MAX_ANALYZER_OCCURRENCES: usize = 256;
/// The Usages pane is bounded; excess usages are counted, never silently dropped.
const MAX_USAGE_ROWS: usize = 64;
pub(crate) fn analyzer_row_id_for_node(key: &str) -> LanguageToolNodeId {
    LanguageToolNodeId(format!("analyzer-node:{key}"))
}

pub(crate) fn analyzer_node_key(id: &LanguageToolNodeId) -> Option<&str> {
    id.0.strip_prefix("analyzer-node:")
}

fn node_row(
    key: &str,
    label: String,
    secondary_label: Option<String>,
    accessibility_label: String,
) -> LanguageToolNode {
    LanguageToolNode {
        id: analyzer_row_id_for_node(key),
        label,
        secondary_label,
        icon: None,
        accessibility_label,
        children: Vec::new(),
        enabled: false,
        activation_label: None,
    }
}

fn wrapper_row(
    id: LanguageToolNodeId,
    label: String,
    accessibility_label: String,
    children: Vec<LanguageToolNode>,
) -> LanguageToolNode {
    LanguageToolNode {
        id,
        label,
        secondary_label: None,
        icon: None,
        accessibility_label,
        children,
        enabled: false,
        activation_label: None,
    }
}

/// Occurrence badge suffix shared by both providers: admitted conflict and
/// unresolved facts, in a stable order.
fn badges_secondary(conflict: bool, unresolved: bool) -> Option<String> {
    let mut badges = Vec::new();
    if conflict {
        badges.push("conflict");
    }
    if unresolved {
        badges.push("unresolved");
    }
    if badges.is_empty() {
        None
    } else {
        Some(badges.join("; "))
    }
}

/// Maven coordinate with the group ID shown or hidden by the user's toggle.
fn maven_coordinate(component: &MavenDependencyComponent, show_group_ids: bool) -> String {
    if show_group_ids {
        format!("{}:{}", component.group_id, component.artifact_id)
    } else {
        component.artifact_id.clone()
    }
}

fn maven_component<'a>(
    components: &'a [MavenDependencyComponent],
    id: &str,
) -> Option<&'a MavenDependencyComponent> {
    components.iter().find(|component| component.id == id)
}

/// Occurrence keys involved in Maven mediation: the loser keeps an omitted
/// incoming edge and the winner is its selected reference.
fn maven_conflict_occurrences(edges: &[MavenDependencyEdge]) -> HashSet<&str> {
    edges
        .iter()
        .filter(|edge| edge.omitted)
        .flat_map(|edge| [edge.target_node.as_str(), edge.selected_node.as_str()])
        .collect()
}

/// Component IDs with unavailable descriptors, the admitted unresolved facts.
fn maven_unavailable_components(
    unavailable_descriptors: &[MavenUnavailableDescriptor],
) -> HashSet<&str> {
    unavailable_descriptors
        .iter()
        .map(|descriptor| descriptor.component.as_str())
        .collect()
}

struct MavenOccurrence<'a> {
    key: &'a str,
    component: &'a MavenDependencyComponent,
    scope: Option<&'a str>,
    conflict: bool,
    unresolved: bool,
}

fn maven_occurrences<'a>(
    nodes: &'a [MavenDependencyNode],
    components: &'a [MavenDependencyComponent],
    edges: &'a [MavenDependencyEdge],
    unavailable_descriptors: &[MavenUnavailableDescriptor],
    graph_scope: MavenGraphScope,
    filter: DependencyAnalyzerFilter,
) -> Vec<MavenOccurrence<'a>> {
    let mut components_by_id = HashMap::new();
    for component in components {
        components_by_id
            .entry(component.id.as_str())
            .or_insert(component);
    }
    let mut scopes_by_node = HashMap::new();
    for edge in edges {
        scopes_by_node
            .entry(edge.target_node.as_str())
            .or_insert(edge.scope.as_str());
    }
    let conflicts = maven_conflict_occurrences(edges);
    let unavailable = maven_unavailable_components(unavailable_descriptors);
    nodes
        .iter()
        .filter_map(|node| {
            let component = components_by_id.get(node.component.as_str()).copied()?;
            let key = node.id.as_str();
            let scope = scopes_by_node.get(key).copied();
            if graph_scope != MavenGraphScope::All && scope.is_none() {
                return None;
            }
            let occurrence = MavenOccurrence {
                key,
                component,
                scope,
                conflict: conflicts.contains(key),
                unresolved: unavailable.contains(node.component.as_str()),
            };
            match filter {
                DependencyAnalyzerFilter::All => Some(occurrence),
                DependencyAnalyzerFilter::Conflicts => occurrence.conflict.then_some(occurrence),
                DependencyAnalyzerFilter::Unresolved => occurrence.unresolved.then_some(occurrence),
            }
        })
        .collect()
}

fn maven_occurrence_label(
    occurrence: &MavenOccurrence,
    flat: bool,
    show_group_ids: bool,
) -> String {
    let coordinate = maven_coordinate(occurrence.component, show_group_ids);
    let version = occurrence.component.version.as_str();
    let subject = if flat {
        format!("{coordinate}:{version}")
    } else {
        version.to_owned()
    };
    match &occurrence.scope {
        Some(scope) => format!("{subject} — {scope}"),
        None => subject,
    }
}

fn maven_occurrence_accessibility(occurrence: &MavenOccurrence, show_group_ids: bool) -> String {
    let coordinate = maven_coordinate(occurrence.component, show_group_ids);
    let mut text = format!("Maven dependency {coordinate}");
    if let Some(scope) = &occurrence.scope {
        text.push_str(&format!(" scope {scope}"));
    }
    if occurrence.conflict {
        text.push_str("; conflict");
    }
    if occurrence.unresolved {
        text.push_str("; unresolved");
    }
    text
}

fn maven_analyzer_rows(
    kept: &[MavenOccurrence<'_>],
    presentation: DependencyAnalyzerPresentation,
    show_group_ids: bool,
) -> Vec<LanguageToolNode> {
    match presentation {
        DependencyAnalyzerPresentation::Flat => kept
            .iter()
            .map(|occurrence| {
                node_row(
                    occurrence.key,
                    maven_occurrence_label(occurrence, true, show_group_ids),
                    badges_secondary(occurrence.conflict, occurrence.unresolved),
                    maven_occurrence_accessibility(occurrence, show_group_ids),
                )
            })
            .collect(),
        DependencyAnalyzerPresentation::Tree => {
            let mut groups: BTreeMap<String, BTreeMap<String, Vec<&MavenOccurrence>>> =
                BTreeMap::new();
            for occurrence in kept {
                groups
                    .entry(occurrence.component.group_id.clone())
                    .or_default()
                    .entry(format!(
                        "{}:{}",
                        occurrence.component.group_id, occurrence.component.artifact_id
                    ))
                    .or_default()
                    .push(occurrence);
            }
            groups
                .into_iter()
                .map(|(group_id, artifacts)| {
                    let artifact_rows = artifacts
                        .into_iter()
                        .map(|(artifact_key, occurrences)| {
                            let occurrence_rows = occurrences
                                .iter()
                                .map(|occurrence| {
                                    node_row(
                                        occurrence.key,
                                        maven_occurrence_label(occurrence, false, show_group_ids),
                                        badges_secondary(
                                            occurrence.conflict,
                                            occurrence.unresolved,
                                        ),
                                        maven_occurrence_accessibility(occurrence, show_group_ids),
                                    )
                                })
                                .collect::<Vec<_>>();
                            let count = occurrence_rows.len();
                            wrapper_row(
                                LanguageToolNodeId(format!("analyzer-artifact:{artifact_key}")),
                                maven_coordinate(occurrences[0].component, show_group_ids),
                                format!("Maven artifact {artifact_key} with {count} occurrences"),
                                occurrence_rows,
                            )
                        })
                        .collect::<Vec<_>>();
                    let artifacts_len = artifact_rows.len();
                    wrapper_row(
                        LanguageToolNodeId(format!("analyzer-group:{group_id}")),
                        group_id.clone(),
                        format!("Maven dependency group {group_id} with {artifacts_len} artifacts"),
                        artifact_rows,
                    )
                })
                .collect()
        }
    }
}

/// Gradle coordinate with the group ID shown or hidden by the user's toggle;
/// project components fall back to their project path or build-tree path.
fn gradle_coordinate(component: &GradleDependencyComponent, show_group_ids: bool) -> String {
    let module = component.module.clone().unwrap_or_else(|| {
        component
            .project_path
            .clone()
            .or_else(|| component.build_tree_path.clone())
            .unwrap_or_else(|| component.id.clone())
    });
    match (&component.group, show_group_ids) {
        (Some(group), true) => format!("{group}:{module}"),
        _ => module,
    }
}

struct GradleOccurrence {
    key: String,
    group: String,
    artifact: String,
    label: String,
    conflict: bool,
    unresolved: bool,
}

fn gradle_occurrence_accessibility(occurrence: &GradleOccurrence) -> String {
    let mut text = format!("Gradle dependency {}", occurrence.label);
    if occurrence.conflict {
        text.push_str("; conflict");
    }
    if occurrence.unresolved {
        text.push_str("; unresolved");
    }
    text
}

fn gradle_occurrences(
    components: &[GradleDependencyComponent],
    edges: &[GradleDependencyEdge],
    filter: DependencyAnalyzerFilter,
) -> Vec<GradleOccurrence> {
    // Dependency structure comes from resolved non-constraint edges; constraint
    // edges introduce no dependency rows (the same rule the inspection walks
    // apply) and unresolved edges become explicit placeholder rows.
    let mut occurrences: Vec<GradleOccurrence> = Vec::new();
    for component in components {
        let artifact = component
            .module
            .clone()
            .or_else(|| {
                component
                    .project_path
                    .clone()
                    .or_else(|| component.build_tree_path.clone())
            })
            .unwrap_or_else(|| component.id.clone());
        let group = component
            .group
            .clone()
            .unwrap_or_else(|| "(project)".into());
        occurrences.push(GradleOccurrence {
            key: component.id.clone(),
            label: component
                .version
                .clone()
                .unwrap_or_else(|| component.id.clone()),
            group,
            artifact,
            conflict: component.conflict_resolution,
            unresolved: false,
        });
    }
    let mut placeholders: BTreeMap<String, GradleOccurrence> = BTreeMap::new();
    for edge in edges {
        if edge.constraint || edge.resolved {
            continue;
        }
        let (key, label) = match &edge.requested {
            GradleDependencySelector::Module {
                group,
                module,
                version,
                ..
            } => (
                format!("unresolved:{group}:{module}:{version}"),
                format!("{group}:{module}:{version}"),
            ),
            GradleDependencySelector::Project { project_path, .. } => (
                format!("unresolved-project:{project_path}"),
                project_path.clone(),
            ),
        };
        let occurrence = GradleOccurrence {
            key: key.clone(),
            label: label.clone(),
            group: "(unresolved)".into(),
            artifact: label,
            conflict: false,
            unresolved: true,
        };
        placeholders.entry(key).or_insert(occurrence);
    }
    occurrences.extend(placeholders.into_values());
    occurrences.retain(|occurrence| match filter {
        DependencyAnalyzerFilter::All => true,
        DependencyAnalyzerFilter::Conflicts => occurrence.conflict,
        DependencyAnalyzerFilter::Unresolved => occurrence.unresolved,
    });
    occurrences
}

fn gradle_analyzer_rows(
    components: &[GradleDependencyComponent],
    edges: &[GradleDependencyEdge],
    filter: DependencyAnalyzerFilter,
    presentation: DependencyAnalyzerPresentation,
    show_group_ids: bool,
) -> Vec<LanguageToolNode> {
    let kept = gradle_occurrences(components, edges, filter);
    match presentation {
        DependencyAnalyzerPresentation::Flat => kept
            .iter()
            .map(|occurrence| {
                node_row(
                    &occurrence.key,
                    gradle_flat_occurrence_label(occurrence, show_group_ids),
                    badges_secondary(occurrence.conflict, occurrence.unresolved),
                    gradle_occurrence_accessibility(occurrence),
                )
            })
            .collect(),
        DependencyAnalyzerPresentation::Tree => {
            let mut groups: BTreeMap<String, BTreeMap<String, Vec<&GradleOccurrence>>> =
                BTreeMap::new();
            for occurrence in &kept {
                let artifact_key = format!("{}:{}", occurrence.group, occurrence.artifact);
                groups
                    .entry(occurrence.group.clone())
                    .or_default()
                    .entry(artifact_key)
                    .or_default()
                    .push(occurrence);
            }
            groups
                .into_iter()
                .map(|(group_id, artifacts)| {
                    let artifact_rows = artifacts
                        .into_iter()
                        .map(|(artifact_key, group_occurrences)| {
                            let occurrence_rows = group_occurrences
                                .iter()
                                .map(|occurrence| {
                                    node_row(
                                        &occurrence.key,
                                        occurrence.label.clone(),
                                        badges_secondary(
                                            occurrence.conflict,
                                            occurrence.unresolved,
                                        ),
                                        gradle_occurrence_accessibility(occurrence),
                                    )
                                })
                                .collect::<Vec<_>>();
                            let count = occurrence_rows.len();
                            wrapper_row(
                                LanguageToolNodeId(format!("analyzer-artifact:{artifact_key}")),
                                artifact_key.clone(),
                                format!("Gradle artifact {artifact_key} with {count} versions"),
                                occurrence_rows,
                            )
                        })
                        .collect::<Vec<_>>();
                    let artifacts_len = artifact_rows.len();
                    wrapper_row(
                        LanguageToolNodeId(format!("analyzer-group:{group_id}")),
                        group_id.clone(),
                        format!(
                            "Gradle dependency group {group_id} with {artifacts_len} artifacts"
                        ),
                        artifact_rows,
                    )
                })
                .collect()
        }
    }
}

/// The Gradle flat label keeps the coordinate stable across the group-ID
/// toggle for project components, whose artifact already carries the path.
fn gradle_coordinate_label(occurrence: &GradleOccurrence) -> String {
    if occurrence.group == "(project)" || occurrence.group == "(unresolved)" {
        occurrence.artifact.clone()
    } else {
        format!("{}:{}", occurrence.group, occurrence.artifact)
    }
}

fn gradle_flat_occurrence_label(occurrence: &GradleOccurrence, show_group_ids: bool) -> String {
    if show_group_ids {
        format!(
            "{}:{}",
            gradle_coordinate_label(occurrence),
            occurrence.label
        )
    } else {
        format!("{}:{}", occurrence.artifact, occurrence.label)
    }
}

/// One Usages pane row: the displayed label plus the trustworthy Gradle
/// declaration target when the underlying reverse edge reports a known origin.
#[derive(Clone, Debug)]
pub(crate) struct AnalyzerUsage {
    pub label: String,
    pub user_node: String,
    pub declaration: Option<String>,
}

/// Reverse-edge usage rows for one occurrence or component key, bounded with
/// an explicit truncation count.
pub(crate) struct AnalyzerUsages {
    pub rows: Vec<AnalyzerUsage>,
    pub truncated: bool,
}

pub(crate) fn analyzer_usages(graph: &DependencyAnalyzerGraph, node_key: &str) -> AnalyzerUsages {
    let mut rows = Vec::new();
    let mut truncated = false;
    match graph {
        DependencyAnalyzerGraph::Gradle {
            components, edges, ..
        } => {
            for edge in edges {
                let Some(target) = edge.selected.as_ref() else {
                    continue;
                };
                if target != node_key || edge.constraint {
                    continue;
                }
                if rows.len() == MAX_USAGE_ROWS {
                    truncated = true;
                    break;
                }
                let user = components
                    .iter()
                    .find(|component| component.id == edge.from)
                    .map(|component| gradle_coordinate(component, true))
                    .unwrap_or_else(|| edge.from.clone());
                let declaration = components
                    .iter()
                    .find(|component| component.id == edge.from)
                    .filter(|_| edge.origin_known)
                    .and_then(|component| component.project_path.clone());
                rows.push(AnalyzerUsage {
                    label: format!("used by {user}"),
                    user_node: edge.from.clone(),
                    declaration,
                });
            }
        }
        DependencyAnalyzerGraph::Maven {
            nodes,
            components,
            edges,
            ..
        } => {
            for edge in edges {
                if edge.target_node != node_key && edge.selected_node != node_key {
                    continue;
                }
                if rows.len() == MAX_USAGE_ROWS {
                    truncated = true;
                    break;
                }
                let user = nodes
                    .iter()
                    .find(|node| node.id == edge.from_node)
                    .and_then(|node| maven_component(components, &node.component))
                    .map(|component| maven_coordinate(component, true))
                    .unwrap_or_else(|| edge.from_node.clone());
                let mut detail = format!("used by {user}");
                if !edge.scope.is_empty() {
                    detail.push_str(&format!("; scope {}", edge.scope));
                }
                if edge.omitted {
                    detail.push_str("; omitted by mediation");
                }
                // Maven facts carry no Gradle declaration location.
                rows.push(AnalyzerUsage {
                    label: detail,
                    user_node: edge.from_node.clone(),
                    declaration: None,
                });
            }
        }
    }
    AnalyzerUsages { rows, truncated }
}

/// The trustworthy Gradle declaration target for an analyzer row: the
/// declaring component's Gradle project path from the row's first admitted
/// reverse edge whose origin the provider reports as known. Maven facts have
/// no Gradle declaration, and unresolved placeholders have no admitted
/// reverse edge, so both yield no target.
pub(crate) fn analyzer_declaration_target(
    graph: &DependencyAnalyzerGraph,
    node_key: &str,
) -> Option<String> {
    let (components, edges) = match graph {
        DependencyAnalyzerGraph::Gradle {
            components, edges, ..
        } => (components, edges),
        DependencyAnalyzerGraph::Maven { .. } => return None,
    };
    for edge in edges {
        if edge.constraint || edge.selected.as_deref() != Some(node_key) {
            continue;
        }
        if !edge.origin_known {
            continue;
        }
        if let Some(declaring) = components
            .iter()
            .find(|component| component.id == edge.from)
            .and_then(|component| component.project_path.clone())
        {
            return Some(declaring);
        }
    }
    None
}

/// Builds the analyzer rows for one admitted graph under the current filters.
pub(crate) fn analyzer_rows(
    graph: &DependencyAnalyzerGraph,
    filter: DependencyAnalyzerFilter,
    presentation: DependencyAnalyzerPresentation,
    show_group_ids: bool,
) -> Vec<LanguageToolNode> {
    match graph {
        DependencyAnalyzerGraph::Gradle {
            components, edges, ..
        } => gradle_analyzer_rows(components, edges, filter, presentation, show_group_ids),
        DependencyAnalyzerGraph::Maven {
            scope,
            nodes,
            components,
            edges,
            unavailable_descriptors,
            ..
        } => maven_analyzer_rows(
            &maven_occurrences(
                nodes.get(..MAX_ANALYZER_OCCURRENCES).unwrap_or(nodes),
                components,
                edges,
                unavailable_descriptors,
                *scope,
                filter,
            ),
            presentation,
            show_group_ids,
        ),
    }
}

/// The analyzer tab title; the provider's configuration stays explicit so
/// Gradle configuration scopes are distinguishable at a glance.
fn analyzer_title(target: &DependencyDiagramTarget) -> String {
    match (&target.provider, &target.configuration) {
        (DependencyDiagramProvider::Gradle, Some(configuration)) => {
            format!(
                "Dependency Analyzer — {module} ({configuration})",
                module = target.module_label
            )
        }
        _ => format!("Dependency Analyzer — {}", target.module_label),
    }
}

impl DependencyAnalyzerItem {
    pub fn new(
        derivation: DependencyAnalyzerDerivation,
        source: Option<DependencyAnalyzerSource>,
        cx: &mut Context<Self>,
    ) -> Self {
        let receipt = source.as_ref().and_then(|source| {
            current_dependency_graph_receipt(source.project.read(cx), &source.selection, cx)
        });
        let panel_subscription = source.as_ref().and_then(|source| {
            let panel = source.panel.upgrade()?;
            Some(cx.observe(&panel, |item, _, cx| item.refresh_receipt(cx)))
        });
        let focus_handle = cx.focus_handle().tab_stop(true);
        let mut item = Self {
            host: LanguageToolTreeHost::with_focus_handle(focus_handle.clone()),
            derivation,
            source,
            receipt,
            _panel_subscription: panel_subscription,
            search_handle: ui::PopoverMenuHandle::default(),
            filter: DependencyAnalyzerFilter::All,
            presentation: DependencyAnalyzerPresentation::Tree,
            show_group_ids: true,
            selected_key: None,
            selected_outside_tree: false,
            usages: Vec::new(),
            usages_truncated: false,
            usages_scroll: gpui::ScrollHandle::new(),
            usage_focus_selection: None,
            usage_focus_invalidated: false,
            usage_focus_handles: Vec::new(),
            last_focused_usage: None,
            notice: None,
            row_context_menu: None,
            focus_handle,
        };
        let roots = analyzer_rows(
            &item.derivation.graph,
            item.filter,
            item.presentation,
            item.show_group_ids,
        );
        item.host.replace_snapshot(LanguageToolSnapshot {
            roots,
            status: LanguageToolProviderStatus::Current,
        });
        item
    }

    /// Opens the analyzer as a focused editor tab in the workspace's active
    /// pane and returns the item handle.
    pub fn open(
        derivation: DependencyAnalyzerDerivation,
        source: Option<DependencyAnalyzerSource>,
        workspace: &mut Workspace,
        window: &mut gpui::Window,
        cx: &mut App,
    ) -> gpui::Entity<Self> {
        let item = cx.new(|cx| Self::new(derivation, source, cx));
        workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
        item
    }

    pub(crate) fn matches_target(&self, selection: &GraphSelection) -> bool {
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
        self.usage_focus_invalidated = true;
        if self.selected_outside_tree {
            self.selected_key = None;
            self.selected_outside_tree = false;
        }
        let derivation = receipt.as_ref().and_then(|receipt| {
            derive_dependency_analyzer(
                project,
                &receipt.selection,
                Some(&self.derivation.scope),
                cx,
            )
        });
        self.search_handle.hide(cx);
        if let Some((menu, _, _)) = self.row_context_menu.as_ref() {
            menu.update(cx, |_, cx| cx.emit(gpui::DismissEvent));
        }

        match (receipt, derivation) {
            (Some(receipt), Some(derivation)) => {
                source.selection = receipt.selection.clone();
                self.receipt = Some(receipt);
                self.derivation = derivation;
                self.notice = None;
            }
            _ => {
                self.receipt = None;
                self.derivation.graph.clear();
                self.derivation.scope_options.clear();
                self.notice = Some("Dependency graph facts are unavailable. Refresh and collect graph facts in the Dependencies panel.".into());
            }
        }
        self.replace_rows(cx);
    }

    pub fn target(&self) -> &DependencyDiagramTarget {
        &self.derivation.target
    }

    pub fn scope(&self) -> &DependencyDiagramScope {
        &self.derivation.scope
    }

    pub fn scope_options(&self) -> &[String] {
        &self.derivation.scope_options
    }

    pub fn filter(&self) -> DependencyAnalyzerFilter {
        self.filter
    }

    pub fn presentation(&self) -> DependencyAnalyzerPresentation {
        self.presentation
    }

    pub fn show_group_ids(&self) -> bool {
        self.show_group_ids
    }

    #[cfg(test)]
    pub(crate) fn usages(&self) -> &[AnalyzerUsage] {
        &self.usages
    }

    #[cfg(test)]
    pub(crate) fn usages_scroll(&self) -> &gpui::ScrollHandle {
        &self.usages_scroll
    }

    /// Whether a row context menu is staged; tests read this to assert the
    /// menu's lifecycle.
    #[cfg(test)]
    pub(crate) fn row_context_menu_is_deployed(&self) -> bool {
        self.row_context_menu.is_some()
    }

    pub fn usages_truncated(&self) -> bool {
        self.usages_truncated
    }

    pub fn notice(&self) -> Option<&str> {
        self.notice.as_deref()
    }

    pub fn selected_key(&self) -> Option<&str> {
        self.selected_key.as_deref()
    }

    pub(crate) fn search_entries(&self, cx: &App) -> Vec<DependencySearchEntry> {
        if !self.receipt_is_current(cx) {
            return Vec::new();
        }
        match &self.derivation.graph {
            DependencyAnalyzerGraph::Gradle {
                components, edges, ..
            } => gradle_occurrences(components, edges, self.filter)
                .iter()
                .map(|occurrence| {
                    let coordinate = if occurrence.unresolved {
                        occurrence.artifact.clone()
                    } else {
                        gradle_flat_occurrence_label(occurrence, true)
                    };
                    DependencySearchEntry::new(
                        analyzer_row_id_for_node(&occurrence.key).0,
                        gradle_flat_occurrence_label(occurrence, self.show_group_ids),
                        Some(coordinate),
                    )
                })
                .collect(),
            DependencyAnalyzerGraph::Maven {
                scope,
                nodes,
                components,
                edges,
                unavailable_descriptors,
            } => maven_occurrences(
                nodes,
                components,
                edges,
                unavailable_descriptors,
                *scope,
                self.filter,
            )
            .iter()
            .map(|occurrence| {
                let coordinate = format!(
                    "{}:{}",
                    maven_coordinate(occurrence.component, true),
                    occurrence.component.version
                );
                DependencySearchEntry::new(
                    analyzer_row_id_for_node(occurrence.key).0,
                    maven_occurrence_label(occurrence, true, self.show_group_ids),
                    Some(coordinate),
                )
            })
            .collect(),
        }
    }

    pub(crate) fn reveal_row(&mut self, id: LanguageToolNodeId, cx: &mut Context<Self>) {
        if !self.receipt_is_current(cx) {
            return;
        }
        fn find_ancestors(
            rows: &[LanguageToolNode],
            id: &LanguageToolNodeId,
            ancestors: &mut Vec<LanguageToolNodeId>,
        ) -> bool {
            for row in rows {
                if &row.id == id {
                    return true;
                }
                ancestors.push(row.id.clone());
                if find_ancestors(&row.children, id, ancestors) {
                    return true;
                }
                ancestors.pop();
            }
            false
        }

        let rows = analyzer_rows(
            &self.derivation.graph,
            self.filter,
            self.presentation,
            self.show_group_ids,
        );
        let mut ancestors = Vec::new();
        if find_ancestors(&rows, &id, &mut ancestors) {
            self.host.expand_nodes(ancestors);
            self.host.select(id);
            self.host.reveal_selection(ScrollStrategy::Center);
            self.notice = None;
            self.refresh_selection(cx);
        } else if let Some(key) = analyzer_node_key(&id)
            && self.node_label(key).is_some()
        {
            self.host.clear_selection();
            self.selected_key = Some(key.to_owned());
            self.selected_outside_tree = true;
            self.notice = None;
            self.refresh_usages();
            cx.notify();
        } else {
            self.notice = Some(
                "The dependency is no longer part of the admitted facts; refresh the analyzer and retry".into(),
            );
            cx.notify();
        }
    }

    /// Re-derives the rows from admitted facts under the chosen scope without
    /// any provider query.
    pub fn apply_scope(&mut self, label: String, cx: &mut Context<Self>) {
        let Some(source) = self.source.clone() else {
            return;
        };
        let scope = match &self.derivation.scope {
            DependencyDiagramScope::GradleConfiguration(_) => {
                DependencyDiagramScope::GradleConfiguration(label)
            }
            DependencyDiagramScope::MavenScope(_) => DependencyDiagramScope::MavenScope(label),
        };
        let project = source.project.read(cx);
        match derive_dependency_analyzer(project, &source.selection, Some(&scope), cx) {
            Some(derivation) => {
                self.derivation = derivation;
                self.usage_focus_invalidated = true;
                self.selected_key = None;
                self.selected_outside_tree = false;
                self.usages.clear();
                self.usages_truncated = false;
                self.notice = None;
                self.replace_rows(cx);
            }
            None => {
                self.notice = Some(
                    "No admitted graph facts for the chosen scope; refresh provider graph facts and retry"
                        .into(),
                );
                cx.notify();
            }
        }
    }

    /// Stages the context menu for a tree row: Show in Diagram targets the
    /// row's dependency node and Go to Gradle Declaration requires the row's
    /// admitted reverse edge to report a trustworthy origin. Wrapper rows
    /// (group and artifact) carry no node target, so both entries stay
    /// disabled for them.
    pub fn show_row_context_menu(
        &mut self,
        id: LanguageToolNodeId,
        position: gpui::Point<Pixels>,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        let node_key = analyzer_node_key(&id).map(|key| key.to_owned());
        self.select_node_row(id, cx);
        let declaration = node_key
            .as_ref()
            .and_then(|key| analyzer_declaration_target(&self.derivation.graph, key));
        self.stage_row_context_menu(node_key, declaration, position, window, cx);
    }

    /// Stages the context menu for a Usages pane row; the menu targets the
    /// specific usage occurrence the row displays, so Go to Gradle
    /// Declaration uses that usage's own edge origin.
    pub fn show_usage_context_menu(
        &mut self,
        index: usize,
        position: gpui::Point<Pixels>,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        if !self.usage_focus_is_current() {
            return;
        }
        let Some(usage) = self.usages.get(index) else {
            return;
        };
        self.stage_row_context_menu(
            Some(usage.user_node.clone()),
            usage.declaration.clone(),
            position,
            window,
            cx,
        );
    }

    fn usage_focus_is_current(&self) -> bool {
        !self.usage_focus_invalidated && self.usage_focus_selection == self.selected_key
    }

    fn focused_usage(&self, window: &gpui::Window) -> Option<usize> {
        if !self.usage_focus_is_current() {
            return None;
        }
        self.usage_focus_handles
            .iter()
            .take(self.usages.len())
            .position(|focus| focus.is_focused(window))
    }

    fn usage_child_index(&self, index: usize) -> usize {
        1 + usize::from(self.selected_outside_tree && self.selected_key.is_some()) + index
    }

    fn show_focused_row_context_menu(&mut self, window: &mut gpui::Window, cx: &mut Context<Self>) {
        if let Some(index) = self.focused_usage(window) {
            let Some(bounds) = self
                .usages_scroll
                .bounds_for_item(self.usage_child_index(index))
            else {
                return;
            };
            let viewport = self.usages_scroll.bounds();
            let offset = self.usages_scroll.offset();
            let position = gpui::point(
                (bounds.left() + offset.x + gpui::px(12.)).clamp(viewport.left(), viewport.right()),
                (bounds.center().y + offset.y).clamp(viewport.top(), viewport.bottom()),
            );
            self.show_usage_context_menu(index, position, window, cx);
        } else if self.focus_handle.is_focused(window)
            && let Some(id) = self.host.selected().cloned()
            && let Some(position) = crate::selected_dependency_row_menu_position(&self.host)
        {
            self.show_row_context_menu(id, position, window, cx);
        }
    }

    fn stage_row_context_menu(
        &mut self,
        node_key: Option<String>,
        declaration: Option<String>,
        position: gpui::Point<Pixels>,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        if !self.receipt_is_current(cx) {
            return;
        }
        let show_in_diagram = node_key.is_some() && self.source.is_some();
        let focus_handle = self.focus_handle.clone();
        let menu = ui::ContextMenu::build(window, cx, |menu, _, _| {
            menu.context(focus_handle).map(|menu| {
                let node_key = node_key.clone().unwrap_or_default();
                let declaring = declaration.clone().unwrap_or_default();
                menu.action_disabled_when(
                    !show_in_diagram,
                    "Show in Dependency Diagram",
                    Box::new(ShowInDependencyDiagram { node_key }),
                )
                .separator()
                .action_disabled_when(
                    declaration.is_none(),
                    "Go to Gradle Declaration",
                    Box::new(GoToDependencyDeclaration { declaring }),
                )
            })
        });
        let menu_focus = menu.focus_handle(cx);
        let return_focus = window
            .focused(cx)
            .filter(|_| self.focus_handle.contains_focused(window, cx))
            .unwrap_or_else(|| self.focus_handle.clone());
        let returning_to_usage = self.usage_focus_handles.contains(&return_focus);
        window.focus(&menu_focus, cx);
        let subscription = cx.subscribe_in(
            &menu,
            window,
            move |this, _, _: &gpui::DismissEvent, window, cx| {
                this.row_context_menu.take();
                if menu_focus.contains_focused(window, cx) {
                    if returning_to_usage
                        && (!this.usage_focus_is_current()
                            || !this.usage_focus_handles.contains(&return_focus))
                    {
                        this.focus_handle.focus(window, cx);
                    } else {
                        return_focus.focus(window, cx);
                    }
                }
                cx.notify();
            },
        );
        self.row_context_menu = Some((menu, position, subscription));
        cx.notify();
    }

    /// Routes Show in Diagram through the opening panel so the diagram tab
    /// shares one derivation and open/focus path with the panel's own
    /// diagram surfaces; the row's node key lets the diagram emphasize the
    /// targeted dependency when the canvas admits it.
    pub fn show_in_dependency_diagram(&mut self, node_key: &str, cx: &mut Context<Self>) {
        if !self.receipt_is_current(cx) {
            return;
        }
        let Some(source) = self.source.clone() else {
            self.notice = Some(
                "The analyzer's opening panel is unavailable; reopen the analyzer from the Dependencies panel, then retry Show in Diagram"
                    .into(),
            );
            cx.notify();
            return;
        };
        let node_key = node_key.to_owned();
        let scope = self.derivation.scope.clone();
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
                panel.show_dependency_diagram_for_analyzer(&source.selection, &node_key, &scope, cx)
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

    /// Routes Go to Gradle Declaration through the opening panel, which owns
    /// the snapshot-to-build-script mapping and the workspace navigation.
    pub fn go_to_dependency_declaration(
        &mut self,
        declaring: &str,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) {
        if !self.receipt_is_current(cx) {
            return;
        }
        let Some(source) = self.source.clone() else {
            self.notice = Some(
                "The analyzer's opening panel is unavailable; reopen the analyzer from the Dependencies panel, then retry Go to Gradle Declaration"
                    .into(),
            );
            cx.notify();
            return;
        };
        let has_origin = self.selected_key.as_ref().is_some_and(|node| {
            analyzer_declaration_target(&self.derivation.graph, node).as_deref() == Some(declaring)
        }) || self
            .usages
            .iter()
            .any(|usage| usage.declaration.as_deref() == Some(declaring));
        if !has_origin {
            self.notice = Some(
                "The provider did not report a current Gradle declaration for this usage.".into(),
            );
            cx.notify();
            return;
        }
        let selection = source.selection.clone();
        let declaring = declaring.to_owned();
        let result = source.panel.update(cx, |panel, cx| {
            panel.open_gradle_declaration(&selection, &declaring, window, cx)
        });
        self.notice = match result {
            Ok(Ok(())) => None,
            Ok(Err(error)) | Err(error) => Some(error.to_string()),
        };
        cx.notify();
    }

    /// Applies the conflicts/unresolved fact filter; the selection is
    /// re-resolved against the filtered rows.
    pub fn apply_filter(&mut self, filter: DependencyAnalyzerFilter, cx: &mut Context<Self>) {
        self.filter = filter;
        self.replace_rows(cx);
    }

    pub fn toggle_group_ids(&mut self, cx: &mut Context<Self>) {
        self.show_group_ids = !self.show_group_ids;
        self.replace_rows(cx);
    }

    pub fn toggle_presentation(&mut self, cx: &mut Context<Self>) {
        self.presentation = match self.presentation {
            DependencyAnalyzerPresentation::Tree => DependencyAnalyzerPresentation::Flat,
            DependencyAnalyzerPresentation::Flat => DependencyAnalyzerPresentation::Tree,
        };
        self.replace_rows(cx);
    }

    pub fn expand_all_rows(&mut self, cx: &mut Context<Self>) {
        self.host.expand_all();
        cx.notify();
    }

    pub fn collapse_all_rows(&mut self, cx: &mut Context<Self>) {
        self.host.collapse_all();
        cx.notify();
    }

    pub fn preselect_node(&mut self, key: &str, cx: &mut Context<Self>) {
        self.reveal_row(analyzer_row_id_for_node(key), cx);
    }

    pub fn select_node_row(&mut self, id: LanguageToolNodeId, cx: &mut Context<Self>) {
        self.host.select(id);
        self.refresh_selection(cx);
    }

    fn navigate_rows(&mut self, select: fn(&mut LanguageToolTreeHost), cx: &mut Context<Self>) {
        select(&mut self.host);
        self.host.reveal_selection(ScrollStrategy::Center);
        self.refresh_selection(cx);
    }

    /// The disclosure path: the panel selects before toggling so a collapsed
    /// branch keeps its selection.
    pub fn toggle_row(&mut self, id: LanguageToolNodeId, cx: &mut Context<Self>) {
        self.host.select(id.clone());
        self.host.toggle(&id);
        self.refresh_selection(cx);
    }

    fn refresh_selection(&mut self, cx: &mut Context<Self>) {
        self.selected_outside_tree = false;
        self.selected_key = self
            .host
            .selected()
            .and_then(|id| analyzer_node_key(id))
            .map(|key| key.to_owned());
        self.refresh_usages();
        cx.notify();
    }

    fn refresh_usages(&mut self) {
        self.usages.clear();
        self.usages_truncated = false;
        if let Some(key) = &self.selected_key {
            let usages = analyzer_usages(&self.derivation.graph, key);
            self.usages = usages.rows;
            self.usages_truncated = usages.truncated;
        }
    }

    fn node_label(&self, key: &str) -> Option<String> {
        match &self.derivation.graph {
            DependencyAnalyzerGraph::Gradle { components, .. } => {
                let component = components.iter().find(|component| component.id == key)?;
                let coordinate = gradle_coordinate(component, self.show_group_ids);
                Some(match &component.version {
                    Some(version) => format!("{coordinate}:{version}"),
                    None => coordinate,
                })
            }
            DependencyAnalyzerGraph::Maven {
                nodes, components, ..
            } => {
                let node = nodes.iter().find(|node| node.id == key)?;
                let component = maven_component(components, &node.component)?;
                let coordinate = maven_coordinate(component, self.show_group_ids);
                Some(format!("{coordinate}:{}", component.version))
            }
        }
    }

    fn replace_rows(&mut self, cx: &mut Context<Self>) {
        let roots = analyzer_rows(
            &self.derivation.graph,
            self.filter,
            self.presentation,
            self.show_group_ids,
        );
        self.host.replace_snapshot(LanguageToolSnapshot {
            roots,
            status: LanguageToolProviderStatus::Current,
        });
        // Re-resolve the selected occurrence against the new rows so a
        // filtered or re-derived tree never keeps a selection its rows cannot
        // represent.
        let retained = self.selected_key.as_ref().and_then(|key| {
            let id = analyzer_row_id_for_node(key);
            self.host
                .visible_rows()
                .iter()
                .find(|row| row.id == id)
                .map(|_| id)
        });
        if let Some(id) = retained {
            self.host.select(id);
            self.selected_outside_tree = false;
        } else if !self.selected_outside_tree {
            self.selected_key = None;
        }
        self.refresh_usages();
        cx.notify();
    }
}

impl Render for DependencyAnalyzerItem {
    fn render(
        &mut self,
        window: &mut gpui::Window,
        cx: &mut Context<Self>,
    ) -> impl gpui::IntoElement {
        use ui::prelude::*;

        if !self.usage_focus_is_current() {
            if self
                .usage_focus_handles
                .iter()
                .any(|focus| focus.is_focused(window))
            {
                self.focus_handle.focus(window, cx);
            }
            self.usage_focus_handles.clear();
            self.last_focused_usage = None;
            self.usage_focus_selection.clone_from(&self.selected_key);
            self.usage_focus_invalidated = false;
        }
        self.usage_focus_handles
            .resize_with(self.usages.len(), || cx.focus_handle().tab_stop(true));
        let focused_usage = self.focused_usage(window);
        // Scroll on focus entry so a later manual scroll stays in place.
        if focused_usage != self.last_focused_usage
            && let Some(index) = focused_usage
        {
            self.usages_scroll
                .scroll_to_item(self.usage_child_index(index));
        }
        self.last_focused_usage = focused_usage;

        let theme = cx.theme().colors().clone();
        let title = analyzer_title(&self.derivation.target);
        let scope_label = self.derivation.scope.label();
        let scope_options = self.derivation.scope_options.clone();
        let filter = self.filter;
        let filter_label = filter.label();
        let presentation = self.presentation;
        let show_group_ids = self.show_group_ids;
        let search_item = cx.weak_entity();
        let usages_rows = self.usages.clone();
        let usage_item = cx.weak_entity();
        // Right-click handlers are built per row up front because the render
        // children closure cannot borrow the render context.
        let usage_row_handlers: Vec<_> = (0..usages_rows.len())
            .map(|index| {
                cx.listener(move |item, event: &gpui::MouseDownEvent, window, cx| {
                    item.show_usage_context_menu(index, event.position, window, cx);
                })
            })
            .collect();
        let usages_truncated = self.usages_truncated;
        let selected_label = self
            .selected_key
            .as_deref()
            .and_then(|key| self.node_label(key));
        let hidden_key = self
            .selected_key
            .clone()
            .filter(|_| self.selected_outside_tree);
        let notice = self.notice.clone();
        let rows = self.host.visible_rows_snapshot();
        let selected = self.host.selected().cloned();
        let scroll_handle = self.host.scroll_handle().clone();
        let click_item = cx.weak_entity();
        let toggle_item = cx.weak_entity();
        let menu_item = cx.weak_entity();

        let toolbar = div()
            .key_context("DependencyAnalyzerControls")
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
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .child(title),
            )
            .child(
                ui::PopoverMenu::new("dependency-analyzer-search")
                    .with_handle(self.search_handle.clone())
                    .trigger(
                        ui::IconButton::new(
                            "dependency-analyzer-search-icon",
                            ui::IconName::MagnifyingGlass,
                        )
                        .shape(ui::IconButtonShape::Square)
                        .tab_index(0_isize)
                        .aria_label("Search dependency analyzer rows")
                        .tooltip(ui::Tooltip::text("Search dependencies")),
                    )
                    .menu(move |window, cx| {
                        let item = search_item.upgrade()?;
                        let entries = item.read(cx).search_entries(cx);
                        let item = search_item.clone();
                        Some(
                            DependencySearch::new(entries, move |id, cx| {
                                if let Some(item) = item.upgrade() {
                                    item.update(cx, |item, cx| {
                                        item.reveal_row(LanguageToolNodeId(id.into()), cx);
                                    });
                                }
                            })
                            .popover(window, cx),
                        )
                    }),
            )
            .child(
                ui::PopoverMenu::new("dependency-analyzer-scope")
                    .trigger(
                        ui::IconButton::new("dependency-analyzer-scope-icon", ui::IconName::Filter)
                            .shape(ui::IconButtonShape::Square)
                            .tab_index(0_isize)
                            .aria_label(format!(
                                "Dependency analyzer scope; current scope {scope_label}"
                            ))
                            .tooltip(ui::Tooltip::text(format!("Scope: {scope_label}"))),
                    )
                    .menu(move |window, cx| {
                        Some(ui::ContextMenu::build(window, cx, |menu, _, _| {
                            let mut menu = menu;
                            for option in &scope_options {
                                menu = menu.action(
                                    option.clone(),
                                    Box::new(SelectDependencyAnalyzerScope {
                                        label: option.clone(),
                                    }),
                                );
                            }
                            menu
                        }))
                    }),
            )
            .child({
                let menu_entries: Vec<(&'static str, DependencyAnalyzerFilter)> = vec![
                    (
                        DependencyAnalyzerFilter::All.label(),
                        DependencyAnalyzerFilter::All,
                    ),
                    (
                        DependencyAnalyzerFilter::Conflicts.label(),
                        DependencyAnalyzerFilter::Conflicts,
                    ),
                    (
                        DependencyAnalyzerFilter::Unresolved.label(),
                        DependencyAnalyzerFilter::Unresolved,
                    ),
                ];
                ui::PopoverMenu::new("dependency-analyzer-filter")
                    .trigger(
                        ui::IconButton::new(
                            "dependency-analyzer-filter-icon",
                            ui::IconName::Filter,
                        )
                        .shape(ui::IconButtonShape::Square)
                        .tab_index(0_isize)
                        .aria_label(format!(
                            "Dependency analyzer fact filter; current filter {filter_label}"
                        ))
                        .tooltip(ui::Tooltip::text(format!("Filter: {filter_label}"))),
                    )
                    .menu(move |window, cx| {
                        Some(ui::ContextMenu::build(window, cx, |menu, _, _| {
                            let mut menu = menu;
                            for (label, entry_filter) in &menu_entries {
                                menu = menu.action(
                                    label.to_string(),
                                    Box::new(SelectDependencyAnalyzerFilter {
                                        filter: *entry_filter,
                                    }),
                                );
                            }
                            menu
                        }))
                    })
            })
            .child(
                ui::IconButton::new("dependency-analyzer-group-ids-icon", ui::IconName::Hash)
                    .shape(ui::IconButtonShape::Square)
                    .tab_index(0_isize)
                    .aria_label(if show_group_ids {
                        "Hide dependency analyzer group IDs"
                    } else {
                        "Show dependency analyzer group IDs"
                    })
                    .tooltip(ui::Tooltip::text("Group IDs"))
                    .on_click(cx.listener(|item, _, _, cx| item.toggle_group_ids(cx))),
            )
            .child(
                ui::IconButton::new(
                    "dependency-analyzer-presentation-icon",
                    ui::IconName::ListCollapse,
                )
                .shape(ui::IconButtonShape::Square)
                .tab_index(0_isize)
                .aria_label(match presentation {
                    DependencyAnalyzerPresentation::Tree => {
                        "Switch the dependency analyzer to the flat list"
                    }
                    DependencyAnalyzerPresentation::Flat => {
                        "Switch the dependency analyzer to the tree"
                    }
                })
                .tooltip(ui::Tooltip::text(match presentation {
                    DependencyAnalyzerPresentation::Tree => "Flat list",
                    DependencyAnalyzerPresentation::Flat => "Tree",
                }))
                .on_click(cx.listener(|item, _, _, cx| item.toggle_presentation(cx))),
            )
            .child(
                ui::IconButton::new(
                    "dependency-analyzer-expand-icon",
                    ui::IconName::ExpandVertical,
                )
                .shape(ui::IconButtonShape::Square)
                .tab_index(0_isize)
                .aria_label("Expand all dependency analyzer rows")
                .tooltip(ui::Tooltip::text("Expand all"))
                .on_click(cx.listener(|item, _, _, cx| item.expand_all_rows(cx))),
            )
            .child(
                ui::IconButton::new(
                    "dependency-analyzer-collapse-icon",
                    ui::IconName::ChevronDownUp,
                )
                .shape(ui::IconButtonShape::Square)
                .tab_index(0_isize)
                .aria_label("Collapse all dependency analyzer rows")
                .tooltip(ui::Tooltip::text("Collapse all"))
                .on_click(cx.listener(|item, _, _, cx| item.collapse_all_rows(cx))),
            )
            .child(crate::DependenciesPanel::ordered_operations_button());

        div()
            .id("groovy-dependency-analyzer-item")
            .key_context("DependencyAnalyzer")
            .track_focus(&self.focus_handle)
            .tab_group()
            .size_full()
            .flex()
            .flex_col()
            .text_color(theme.text)
            .on_action(|_: &crate::FocusNextGroovyControl, window, cx| window.focus_next(cx))
            .on_action(|_: &crate::FocusPreviousGroovyControl, window, cx| window.focus_prev(cx))
            .on_action(cx.listener(|item, _: &SelectNext, _, cx| {
                item.navigate_rows(LanguageToolTreeHost::select_next, cx)
            }))
            .on_action(cx.listener(|item, _: &SelectPrevious, _, cx| {
                item.navigate_rows(LanguageToolTreeHost::select_previous, cx)
            }))
            .on_action(cx.listener(|item, _: &SelectFirst, _, cx| {
                item.navigate_rows(LanguageToolTreeHost::select_first, cx)
            }))
            .on_action(cx.listener(|item, _: &SelectLast, _, cx| {
                item.navigate_rows(LanguageToolTreeHost::select_last, cx)
            }))
            .on_action(cx.listener(|item, _: &SelectParent, _, cx| {
                item.navigate_rows(LanguageToolTreeHost::select_parent, cx)
            }))
            .on_action(cx.listener(|item, _: &SelectFirstChild, _, cx| {
                item.navigate_rows(LanguageToolTreeHost::select_first_child, cx)
            }))
            .on_action(cx.listener(|item, _: &ToggleExpanded, _, cx| {
                if let Some(id) = item.host.selected().cloned() {
                    item.toggle_row(id, cx);
                }
            }))
            .on_action(cx.listener(|item, _: &ShowDependencyRowMenu, window, cx| {
                item.show_focused_row_context_menu(window, cx);
            }))
            .on_action(cx.listener(
                |item, event: &SelectDependencyAnalyzerScope, _, cx| {
                    item.apply_scope(event.label.clone(), cx)
                },
            ))
            .on_action(cx.listener(
                |item, event: &SelectDependencyAnalyzerFilter, _, cx| {
                    item.apply_filter(event.filter, cx)
                },
            ))
            .on_action(cx.listener(|item, _: &ToggleDependencyAnalyzerGroupIds, _, cx| {
                item.toggle_group_ids(cx)
            }))
            .on_action(cx.listener(
                |item, _: &ToggleDependencyAnalyzerPresentation, _, cx| {
                    item.toggle_presentation(cx)
                },
            ))
            .on_action(cx.listener(|item, _: &ExpandDependencyAnalyzerRows, _, cx| {
                item.expand_all_rows(cx)
            }))
            .on_action(cx.listener(|item, _: &CollapseDependencyAnalyzerRows, _, cx| {
                item.collapse_all_rows(cx)
            }))
            .on_action(cx.listener(
                |item, event: &ShowInDependencyDiagram, _, cx| {
                    item.show_in_dependency_diagram(&event.node_key, cx)
                },
            ))
            .on_action(cx.listener(
                |item, event: &GoToDependencyDeclaration, window, cx| {
                    item.go_to_dependency_declaration(&event.declaring, window, cx)
                },
            ))
            .children(self.row_context_menu.as_ref().map(|(menu, position, _)| {
                deferred(
                    anchored()
                        .position(*position)
                        .anchor(gpui::Anchor::TopLeft)
                        .snap_to_window()
                        .child(menu.clone()),
                )
                .with_priority(3)
            }))
            .child(toolbar)
            .when_some(notice, |this, notice| {
                this.child(
                    div()
                        .id("groovy-dependency-analyzer-notice")
                        .role(gpui::Role::Note)
                        .aria_label("Dependency analyzer notice")
                        .px_2()
                        .py_1()
                        .text_size(px(12.0))
                        .text_color(theme.text_muted)
                        .child(notice),
                )
            })
            .child(
                div()
                    .flex_1()
                    .border_1()
                    .border_color(theme.border)
                    .child(language_tool_tree(
                        rows,
                        selected,
                        None,
                        scroll_handle,
                        move |id, _, window, cx| {
                            if let Err(error) = click_item.update(cx, |item, cx| {
                                item.focus_handle.focus(window, cx);
                                item.select_node_row(id, cx);
                            }) {
                                log::debug!("Dependency analyzer closed during click: {error:#}");
                            }
                        },
                        move |id, _, cx| {
                            if let Err(error) = toggle_item.update(cx, |item, cx| {
                                item.toggle_row(id, cx);
                            }) {
                                log::debug!(
                                    "Dependency analyzer closed during toggle: {error:#}"
                                );
                            }
                        },
                        move |id, position, window, cx| {
                            if let Err(error) = menu_item.update(cx, |item, cx| {
                                item.show_row_context_menu(id, position, window, cx);
                            }) {
                                log::debug!(
                                    "Dependency analyzer closed during context menu: {error:#}"
                                );
                            }
                        },
                    )),
            )
            .child(
                div()
                    .id("groovy-dependency-analyzer-usages")
                    .role(gpui::Role::Group)
                    .aria_label("Dependency analyzer usages")
                    .aria_description(
                        "Usage occurrences for the selected artifact: every admitted reverse edge that reaches it, bounded with an explicit count.",
                    )
                    .border_1()
                    .border_color(theme.border)
                    .px_2()
                    .py_1()
                    .max_h(px(240.0))
                    .overflow_y_scroll()
                    .track_scroll(&self.usages_scroll)
                    .child(
                        div()
                            .text_size(px(12.0))
                            .font_weight(FontWeight::MEDIUM)
                            .child(if self.selected_key.is_none() {
                                "Usages — select an artifact to see its usages".to_owned()
                            } else {
                                let count = usages_rows.len();
                                let occurrences = if count == 1 { "occurrence" } else { "occurrences" };
                                let mut header = match selected_label {
                                    Some(label) => format!("Usages for {label} — {count} {occurrences}"),
                                    None => format!("Usages — {count} {occurrences}"),
                                };
                                if usages_truncated {
                                    header.push_str(". More usages are available than the pane can show.");
                                }
                                header
                            }),
                    )
                    .when_some(hidden_key, |this, key| {
                        this.child(
                            h_flex().gap_2().items_center()
                                .key_context("DependencyAnalyzerControls")
                                .child(Label::new("This dependency is outside the current list.").size(LabelSize::Small))
                                .child(
                                    Button::new("show-hidden-dependency-in-diagram", "Show in Diagram")
                                        .tab_index(0_isize)
                                        .aria_label("Show selected dependency in diagram")
                                        .on_click(cx.listener(move |item, _, _, cx| {
                                            item.show_in_dependency_diagram(&key, cx)
                                        })),
                                )
                        )
                    })
                    .children(
                        usages_rows
                            .iter()
                            .zip(self.usage_focus_handles.iter())
                            .enumerate()
                            .zip(usage_row_handlers)
                            .map(|((index, (usage, focus)), handler)| {
                                let item = usage_item.clone();
                                div()
                                    .key_context("DependencyAnalyzerControls DependencyAnalyzerUsage")
                                    .on_mouse_down(gpui::MouseButton::Right, handler)
                                    .child(
                                        Button::new(format!("dependency-usage-{index}"), usage.label.clone())
                                            .label_size(LabelSize::Small)
                                            .track_focus(focus)
                                            .on_click(move |event, window, cx| {
                                                if let Err(error) = item.update(cx, |item, cx| {
                                                    item.show_usage_context_menu(index, event.position(), window, cx);
                                                }) {
                                                    log::debug!("Dependency analyzer closed during usage menu: {error:#}");
                                                }
                                            }),
                                    )
                            }),
                    ),
            )
    }
}

/// Analyzer tabs are session-scoped: admitted facts come from provider
/// collections and are not persisted, so restored sessions drop the tab
/// instead of reopening a stale or empty analyzer.
impl workspace::SerializableItem for DependencyAnalyzerItem {
    fn serialized_item_kind() -> &'static str {
        "DependencyAnalyzerItem"
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
            "Dependency analyzer tabs are session-scoped and are not restored"
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

impl workspace::Item for DependencyAnalyzerItem {
    type Event = DependencyAnalyzerEvent;

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        analyzer_title(&self.derivation.target).into()
    }

    fn tab_icon(&self, _window: &gpui::Window, _cx: &App) -> Option<ui::Icon> {
        Some(ui::Icon::new(match self.derivation.target.provider {
            DependencyDiagramProvider::Gradle => ui::IconName::Gradle,
            DependencyDiagramProvider::Maven => ui::IconName::Maven,
        }))
    }
}

/// Derives analyzer facts for one admitted selection from the panel's
/// snapshot: matches the selection target and applies the chosen scope.
/// Read-only; never queries a provider.
pub(crate) fn dependency_analyzer_for_selection(
    snapshot: &JdtlsProjectSnapshot,
    selection: &GraphSelection,
    scope: Option<&DependencyDiagramScope>,
) -> Option<DependencyAnalyzerDerivation> {
    for module in &snapshot.projects {
        match selection.target() {
            GraphQueryTarget::Maven(pom) => {
                let Some(maven) = module.maven.as_ref().filter(|maven| maven.pom == *pom) else {
                    continue;
                };
                let Some(graph) = &maven.dependency_graph else {
                    continue;
                };
                let chosen_scope = match scope {
                    Some(DependencyDiagramScope::MavenScope(label)) => {
                        crate::dependencies_panel::maven_graph_scope_from_label(label)?
                    }
                    Some(_) => continue,
                    None => MavenGraphScope::default(),
                };
                return Some(DependencyAnalyzerDerivation {
                    graph: DependencyAnalyzerGraph::Maven {
                        scope: chosen_scope,
                        // Usages can refer to sources whose incoming edges are in another scope.
                        nodes: graph.nodes.clone(),
                        components: graph.components.clone(),
                        edges: graph
                            .edges
                            .iter()
                            .filter(|edge| chosen_scope.matches(&edge.scope))
                            .cloned()
                            .collect(),
                        unavailable_descriptors: graph.unavailable_descriptors.clone(),
                    },
                    target: DependencyDiagramTarget {
                        module_label: crate::dependencies_panel::dependency_module_label(
                            &module.module_root,
                        ),
                        provider: DependencyDiagramProvider::Maven,
                        configuration: None,
                    },
                    scope: DependencyDiagramScope::MavenScope(chosen_scope.label().to_owned()),
                    scope_options: crate::dependencies_panel::maven_graph_scope_labels(),
                });
            }
            GraphQueryTarget::Gradle {
                module: module_root,
                configuration,
            } => {
                if module.module_root != *module_root
                    || scope.is_some_and(|scope| {
                        !matches!(scope, DependencyDiagramScope::GradleConfiguration(name) if name == configuration)
                    })
                {
                    continue;
                }
                let Some(gradle) = &module.gradle else {
                    continue;
                };
                // The scope picker offers the configurations the provider
                // reports for the module; the chosen one admits its matching
                // graph only.
                let mut scope_options: Vec<String> = gradle
                    .configurations
                    .iter()
                    .filter(|configuration| gradle.configurations_known && configuration.resolvable)
                    .map(|configuration| configuration.name.clone())
                    .collect();
                if !scope_options.iter().any(|option| option == configuration) {
                    scope_options.push(configuration.clone());
                    scope_options.sort();
                }
                let Some(graph) = gradle
                    .dependency_graph
                    .as_ref()
                    .filter(|graph| graph.resolution.configuration == *configuration)
                else {
                    continue;
                };
                return Some(DependencyAnalyzerDerivation {
                    graph: DependencyAnalyzerGraph::Gradle {
                        configuration: configuration.clone(),
                        components: graph.resolution.components.clone(),
                        edges: graph.resolution.edges.clone(),
                    },
                    target: DependencyDiagramTarget {
                        module_label: crate::dependencies_panel::dependency_module_label(
                            &module.module_root,
                        ),
                        provider: DependencyDiagramProvider::Gradle,
                        configuration: Some(configuration.clone()),
                    },
                    scope: DependencyDiagramScope::GradleConfiguration(configuration.clone()),
                    scope_options,
                });
            }
        }
    }
    None
}
