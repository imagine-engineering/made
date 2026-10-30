use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use db::kvp::KeyValueStore;
use editor::{Editor, EditorEvent};
use serde::{Deserialize, Serialize};

use gpui::{
    Action, App, AsyncWindowContext, Context, Entity, EventEmitter, FocusHandle, Focusable,
    IntoElement, Pixels, Render, Role, ScrollStrategy, Subscription, WeakEntity, Window, actions,
    div, px,
};
use language_tools::language_tool_tree::{
    self, LanguageToolNode, LanguageToolNodeId, LanguageToolProviderStatus, LanguageToolSnapshot,
    LanguageToolTreeHost, language_tool_tree, status_message,
};
use project::{
    Project, ProjectPath,
    jvm_project::{
        GradleAnnotationProcessorOutput, GradleProject, JdtlsProjectMembership,
        JdtlsProjectSnapshot, JvmBuildOperation, JvmBuildOperationReceipt, KotlinBuildSystem,
        KotlinWorkspaceDependency, KotlinWorkspaceSnapshot,
    },
    trusted_worktrees::TrustedWorktrees,
};
use ui::{Tooltip, prelude::*};
use util::rel_path::RelPath;
use workspace::{
    Panel, Workspace, dock::DockPosition, dock::PanelEvent, workspace_scoped_state_key,
};

actions!(
    groovy_dependencies_panel,
    [
        ToggleDependenciesPanel,
        ShowGradle,
        ShowMaven,
        CollectSelectedGraph,
        CancelGraphCollection,
        CycleGraphFactFilter,
        CycleMavenGraphScope,
        ClearGraphFactSearch,
        InspectGraphNeighbours,
        InspectGraphRootPaths,
        ClearGraphInspection,
        ApplySavedMavenSelection,
        ImportGradleSelection,
        CancelSelectedImport,
        ShowDependencyDiagram,
        CloseDependencyDiagram,
        RunSelectedOfflineOperation,
        AddSelectedOrderedOperation,
        RemoveLastOrderedOperation,
        ClearOrderedOperations,
        RunOrderedOfflineOperations
    ]
);

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum BuildToolView {
    #[default]
    Dependencies,
    Gradle,
    Maven,
}

impl BuildToolView {
    fn label(self) -> &'static str {
        match self {
            Self::Dependencies => "Dependencies",
            Self::Gradle => "Gradle",
            Self::Maven => "Maven",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum GraphFactFilter {
    #[default]
    All,
    Direct,
    Conflicts,
    Unavailable,
}

impl GraphFactFilter {
    fn label(self) -> &'static str {
        match self {
            Self::All => "All facts",
            Self::Direct => "Direct dependencies",
            Self::Conflicts => "Conflict resolution",
            Self::Unavailable => "Unavailable metadata",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::All => Self::Direct,
            Self::Direct => Self::Conflicts,
            Self::Conflicts => Self::Unavailable,
            Self::Unavailable => Self::All,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum MavenGraphScope {
    #[default]
    All,
    Compile,
    Runtime,
    Test,
    Provided,
    System,
    Import,
    Other,
}

impl MavenGraphScope {
    fn label(self) -> &'static str {
        match self {
            Self::All => "All scopes",
            Self::Compile => "compile",
            Self::Runtime => "runtime",
            Self::Test => "test",
            Self::Provided => "provided",
            Self::System => "system",
            Self::Import => "import",
            Self::Other => "Other or unspecified",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::All => Self::Compile,
            Self::Compile => Self::Runtime,
            Self::Runtime => Self::Test,
            Self::Test => Self::Provided,
            Self::Provided => Self::System,
            Self::System => Self::Import,
            Self::Import => Self::Other,
            Self::Other => Self::All,
        }
    }

    fn matches(self, scope: &str) -> bool {
        match self {
            Self::All => true,
            Self::Other => !matches!(
                scope,
                "compile" | "runtime" | "test" | "provided" | "system" | "import"
            ),
            _ => scope == self.label(),
        }
    }
}

const PANEL_KEY: &str = "GroovyDependenciesPanel";
const VIEW_STATE_KEY: &str = "GroovyBuildViewState";
const VIEW_STATE_VERSION: u32 = 6;
const MAX_VIEW_STATE_BYTES: usize = 4096;
const MAX_GRAPH_SEARCH_BYTES: usize = 512;
const BASIC_MAVEN_PHASES: &[&str] = &[
    "clean", "validate", "compile", "test", "package", "verify", "install", "deploy", "site",
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BuildViewPreferences {
    version: u32,
    dock_position: settings::DockPosition,
    view: BuildToolView,
    group_gradle_tasks: bool,
    group_gradle_modules: bool,
    group_maven_modules: bool,
    show_maven_artifact_id: bool,
    show_maven_version: bool,
    show_gradle_toolbar: bool,
    show_maven_toolbar: bool,
    show_basic_maven_phases: bool,
    graph_fact_filter: GraphFactFilter,
    maven_graph_scope: MavenGraphScope,
}

impl Default for BuildViewPreferences {
    fn default() -> Self {
        Self {
            version: VIEW_STATE_VERSION,
            dock_position: settings::DockPosition::Right,
            view: BuildToolView::Dependencies,
            group_gradle_tasks: true,
            group_gradle_modules: false,
            group_maven_modules: false,
            show_maven_artifact_id: true,
            show_maven_version: true,
            show_gradle_toolbar: true,
            show_maven_toolbar: true,
            show_basic_maven_phases: true,
            graph_fact_filter: GraphFactFilter::All,
            maven_graph_scope: MavenGraphScope::All,
        }
    }
}

impl BuildViewPreferences {
    fn restore(serialized: Option<&str>) -> anyhow::Result<Self> {
        let Some(serialized) = serialized else {
            return Ok(Self::default());
        };
        anyhow::ensure!(
            serialized.len() <= MAX_VIEW_STATE_BYTES,
            "Oversized build view preferences"
        );
        #[derive(Deserialize)]
        struct VersionHeader {
            version: u32,
        }
        let header: VersionHeader = serde_json::from_str(serialized)?;
        match header.version {
            1 => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct PreviousPreferences {
                    version: u32,
                    view: BuildToolView,
                    group_gradle_tasks: bool,
                    show_maven_artifact_id: bool,
                    show_maven_version: bool,
                    show_gradle_toolbar: bool,
                    show_maven_toolbar: bool,
                }
                let previous: PreviousPreferences = serde_json::from_str(serialized)?;
                anyhow::ensure!(
                    previous.version == 1,
                    "Unsupported previous preferences version"
                );
                Ok(Self {
                    view: previous.view,
                    group_gradle_tasks: previous.group_gradle_tasks,
                    show_maven_artifact_id: previous.show_maven_artifact_id,
                    show_maven_version: previous.show_maven_version,
                    show_gradle_toolbar: previous.show_gradle_toolbar,
                    show_maven_toolbar: previous.show_maven_toolbar,
                    show_basic_maven_phases: false,
                    ..Default::default()
                })
            }
            2 => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct PreviousPreferences {
                    version: u32,
                    view: BuildToolView,
                    group_gradle_tasks: bool,
                    group_gradle_modules: bool,
                    group_maven_modules: bool,
                    show_maven_artifact_id: bool,
                    show_maven_version: bool,
                    show_gradle_toolbar: bool,
                    show_maven_toolbar: bool,
                }
                let previous: PreviousPreferences = serde_json::from_str(serialized)?;
                anyhow::ensure!(
                    previous.version == 2,
                    "Unsupported previous preferences version"
                );
                Ok(Self {
                    view: previous.view,
                    group_gradle_tasks: previous.group_gradle_tasks,
                    group_gradle_modules: previous.group_gradle_modules,
                    group_maven_modules: previous.group_maven_modules,
                    show_maven_artifact_id: previous.show_maven_artifact_id,
                    show_maven_version: previous.show_maven_version,
                    show_gradle_toolbar: previous.show_gradle_toolbar,
                    show_maven_toolbar: previous.show_maven_toolbar,
                    show_basic_maven_phases: false,
                    ..Default::default()
                })
            }
            3 => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct PreviousPreferences {
                    version: u32,
                    view: BuildToolView,
                    group_gradle_tasks: bool,
                    group_gradle_modules: bool,
                    group_maven_modules: bool,
                    show_maven_artifact_id: bool,
                    show_maven_version: bool,
                    show_gradle_toolbar: bool,
                    show_maven_toolbar: bool,
                    show_basic_maven_phases: bool,
                }
                let previous: PreviousPreferences = serde_json::from_str(serialized)?;
                anyhow::ensure!(
                    previous.version == 3,
                    "Unsupported previous preferences version"
                );
                Ok(Self {
                    view: previous.view,
                    group_gradle_tasks: previous.group_gradle_tasks,
                    group_gradle_modules: previous.group_gradle_modules,
                    group_maven_modules: previous.group_maven_modules,
                    show_maven_artifact_id: previous.show_maven_artifact_id,
                    show_maven_version: previous.show_maven_version,
                    show_gradle_toolbar: previous.show_gradle_toolbar,
                    show_maven_toolbar: previous.show_maven_toolbar,
                    show_basic_maven_phases: previous.show_basic_maven_phases,
                    ..Default::default()
                })
            }
            4 => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct PreviousPreferences {
                    version: u32,
                    dock_position: settings::DockPosition,
                    view: BuildToolView,
                    group_gradle_tasks: bool,
                    group_gradle_modules: bool,
                    group_maven_modules: bool,
                    show_maven_artifact_id: bool,
                    show_maven_version: bool,
                    show_gradle_toolbar: bool,
                    show_maven_toolbar: bool,
                    show_basic_maven_phases: bool,
                }
                let previous: PreviousPreferences = serde_json::from_str(serialized)?;
                anyhow::ensure!(
                    previous.version == 4,
                    "Unsupported previous preferences version"
                );
                Ok(Self {
                    dock_position: previous.dock_position,
                    view: previous.view,
                    group_gradle_tasks: previous.group_gradle_tasks,
                    group_gradle_modules: previous.group_gradle_modules,
                    group_maven_modules: previous.group_maven_modules,
                    show_maven_artifact_id: previous.show_maven_artifact_id,
                    show_maven_version: previous.show_maven_version,
                    show_gradle_toolbar: previous.show_gradle_toolbar,
                    show_maven_toolbar: previous.show_maven_toolbar,
                    show_basic_maven_phases: previous.show_basic_maven_phases,
                    ..Default::default()
                })
            }
            5 => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct PreviousPreferences {
                    version: u32,
                    dock_position: settings::DockPosition,
                    view: BuildToolView,
                    group_gradle_tasks: bool,
                    group_gradle_modules: bool,
                    group_maven_modules: bool,
                    show_maven_artifact_id: bool,
                    show_maven_version: bool,
                    show_gradle_toolbar: bool,
                    show_maven_toolbar: bool,
                    show_basic_maven_phases: bool,
                    graph_fact_filter: GraphFactFilter,
                }
                let previous: PreviousPreferences = serde_json::from_str(serialized)?;
                anyhow::ensure!(
                    previous.version == 5,
                    "Unsupported previous preferences version"
                );
                Ok(Self {
                    dock_position: previous.dock_position,
                    view: previous.view,
                    group_gradle_tasks: previous.group_gradle_tasks,
                    group_gradle_modules: previous.group_gradle_modules,
                    group_maven_modules: previous.group_maven_modules,
                    show_maven_artifact_id: previous.show_maven_artifact_id,
                    show_maven_version: previous.show_maven_version,
                    show_gradle_toolbar: previous.show_gradle_toolbar,
                    show_maven_toolbar: previous.show_maven_toolbar,
                    show_basic_maven_phases: previous.show_basic_maven_phases,
                    graph_fact_filter: previous.graph_fact_filter,
                    ..Default::default()
                })
            }
            VIEW_STATE_VERSION => Ok(serde_json::from_str(serialized)?),
            _ => anyhow::bail!("Unsupported build view preferences version"),
        }
    }

    fn group_modules(&self, view: BuildToolView) -> bool {
        match view {
            BuildToolView::Dependencies => false,
            BuildToolView::Gradle => self.group_gradle_modules,
            BuildToolView::Maven => self.group_maven_modules,
        }
    }

    fn show_toolbar(&self, view: BuildToolView) -> bool {
        match view {
            BuildToolView::Dependencies => true,
            BuildToolView::Gradle => self.show_gradle_toolbar,
            BuildToolView::Maven => self.show_maven_toolbar,
        }
    }
}

#[derive(Clone, Copy)]
enum BuildViewOption {
    GroupModules,
    ShowBasicMavenPhases,
    GroupGradleTasks,
    ShowMavenArtifactId,
    ShowMavenVersion,
    ShowToolbar,
}

impl BuildViewOption {
    fn label(self) -> &'static str {
        match self {
            Self::GroupModules => "Group Modules",
            Self::ShowBasicMavenPhases => "Show Basic Phases Only",
            Self::GroupGradleTasks => "Group Tasks",
            Self::ShowMavenArtifactId => "Always Show ArtifactId",
            Self::ShowMavenVersion => "Show Version",
            Self::ShowToolbar => "Show Toolbar",
        }
    }

    fn enabled(self, state: &BuildViewPreferences, view: BuildToolView) -> bool {
        match self {
            Self::GroupModules => state.group_modules(view),
            Self::ShowBasicMavenPhases => state.show_basic_maven_phases,
            Self::GroupGradleTasks => state.group_gradle_tasks,
            Self::ShowMavenArtifactId => state.show_maven_artifact_id,
            Self::ShowMavenVersion => state.show_maven_version,
            Self::ShowToolbar => state.show_toolbar(view),
        }
    }
}

pub struct DependenciesPanel {
    project: Entity<Project>,
    workspace: WeakEntity<Workspace>,
    focus_handle: FocusHandle,
    host: LanguageToolTreeHost,
    provider_servers: HashSet<(lsp::LanguageServerId, settings::WorktreeId)>,
    navigation: HashMap<LanguageToolNodeId, DependencyNavigation>,
    graph_selections: HashMap<LanguageToolNodeId, GraphSelection>,
    operation_selections: HashMap<LanguageToolNodeId, JvmBuildOperationReceipt>,
    operation_notice: Option<String>,
    ordered_operations: Vec<JvmBuildOperationReceipt>,
    graph_query_token: Option<project::jvm_project::ProviderRefreshToken>,
    graph_query: Option<gpui::Task<()>>,
    graph_notice: Option<String>,
    build_phase_token: Option<project::jvm_project::ProviderRefreshToken>,
    build_phase_waiter: Option<gpui::Task<()>>,
    build_phase_notice: Option<String>,
    graph_search: Entity<Editor>,
    graph_inspection: Option<GraphInspection>,
    graph_diagram_selection: Option<GraphSelection>,
    graph_diagram: Option<Entity<markdown::Markdown>>,
    position: DockPosition,
    active: bool,
    view: BuildToolView,
    preferences: BuildViewPreferences,
    preferences_key: Option<String>,
    preferences_notice: Option<String>,
    preferences_write: Option<gpui::Task<()>>,
    preferences_generation: u64,
    show_options: bool,
    _subscriptions: Vec<Subscription>,
}

struct DependencyNavigation {
    path: ProjectPath,
    server_id: lsp::LanguageServerId,
    generation: u64,
    provider: DependencyProvider,
    kind: DependencyNavigationKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DependencyNavigationKind {
    File,
    Directory,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DependencyNavigationTarget {
    ModuleManifest {
        root: ProjectPath,
        build_system: Option<KotlinBuildSystem>,
    },
    SourceDirectory {
        path: ProjectPath,
        navigable: bool,
    },
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SourceDirectoryStatus {
    Available,
    Missing,
    NotDirectory,
    RefreshRequired,
    Stale,
    Hidden,
}

impl SourceDirectoryStatus {
    fn message(self) -> &'static str {
        match self {
            Self::Available => {
                "Present; activate to reveal in Project panel; generation unverified"
            }
            Self::Missing => "Directory not present; generate sources and refresh",
            Self::NotDirectory => "Configured path is not a directory; check the build settings",
            Self::RefreshRequired => "Navigation unavailable; refresh provider metadata",
            Self::Stale => "Saved declaration; refresh before navigating",
            Self::Hidden => "Output details unavailable",
        }
    }
}

fn source_directory_status(
    worktree: &project::Worktree,
    path: &ProjectPath,
    navigable: bool,
    current: bool,
) -> SourceDirectoryStatus {
    if path.worktree_id != worktree.id()
        || path.path.ancestors().any(|ancestor| {
            worktree
                .entry_for_path(ancestor)
                .is_some_and(|entry| entry.is_private || entry.is_external || entry.is_fifo)
        })
    {
        return SourceDirectoryStatus::Hidden;
    }
    if !current {
        return SourceDirectoryStatus::Stale;
    }
    match worktree.entry_for_path(path.path.as_ref()) {
        None => SourceDirectoryStatus::Missing,
        Some(entry) if !entry.is_dir() => SourceDirectoryStatus::NotDirectory,
        Some(_) if !navigable => SourceDirectoryStatus::RefreshRequired,
        Some(_) => SourceDirectoryStatus::Available,
    }
}

fn apply_source_directory_status(
    rows: &mut [LanguageToolNode],
    statuses: &HashMap<LanguageToolNodeId, SourceDirectoryStatus>,
) {
    for row in rows {
        if let Some(status) = statuses.get(&row.id).copied() {
            row.enabled = status == SourceDirectoryStatus::Available;
            row.activation_label = row
                .enabled
                .then(|| "Reveal generated sources in Project panel".into());
            row.secondary_label = Some(status.message().into());
            if status == SourceDirectoryStatus::Hidden {
                row.label = "Annotation processor output".into();
            }
            row.accessibility_label = format!(
                "Gradle annotation-processor output {}; {}; other generated-source coverage unknown",
                row.label,
                status.message()
            );
        }
        apply_source_directory_status(&mut row.children, statuses);
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum DependencyProvider {
    Jdtls,
    Kotlin,
}

#[derive(Clone)]
enum GraphQueryTarget {
    Maven(ProjectPath),
    Gradle {
        module: ProjectPath,
        configuration: String,
    },
}

#[derive(Clone)]
struct GraphSelection {
    server_id: lsp::LanguageServerId,
    generation: u64,
    target: GraphQueryTarget,
}

impl GraphSelection {
    fn worktree_id(&self) -> settings::WorktreeId {
        match &self.target {
            GraphQueryTarget::Maven(pom) => pom.worktree_id,
            GraphQueryTarget::Gradle { module, .. } => module.worktree_id,
        }
    }

    fn matches(&self, snapshot: &JdtlsProjectSnapshot, current: bool) -> bool {
        current
            && snapshot.server_id == self.server_id
            && snapshot.generation == self.generation
            && snapshot.worktree_id == self.worktree_id()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphInspectionMode {
    Neighbours,
    ShortestRootPaths,
}

impl GraphInspectionMode {
    fn label(self) -> &'static str {
        match self {
            Self::Neighbours => "Selected dependency neighbours",
            Self::ShortestRootPaths => "Shortest root paths",
        }
    }
}

#[derive(Clone)]
struct GraphInspection {
    selection: GraphSelection,
    node: String,
    mode: GraphInspectionMode,
}

fn dependency_diagram_markdown(
    nodes: &[(String, String)],
    edges: &[(String, String, String, bool)],
    coverage: &str,
) -> anyhow::Result<String> {
    use std::fmt::Write as _;
    anyhow::ensure!(
        !nodes.is_empty() && nodes.len() <= 64 && edges.len() <= 128,
        "Dependency diagram exceeds presentation limits; use current tree facts"
    );
    let mut identities = HashMap::new();
    let mut source = String::from("flowchart LR\n");
    let label = |text: &str| -> anyhow::Result<String> {
        anyhow::ensure!(
            text.chars().count() <= 256
                && !text
                    .chars()
                    .any(|character| character.is_control() || matches!(character, '<' | '>')),
            "Dependency label is unsupported by the bounded diagram; use current tree facts"
        );
        let mut encoded = String::new();
        for character in text.chars() {
            write!(encoded, "#{};", u32::from(character))?;
        }
        Ok(encoded)
    };
    for (index, (identity, text)) in nodes.iter().enumerate() {
        anyhow::ensure!(
            identities.insert(identity.as_str(), index).is_none(),
            "Dependency diagram contains duplicate provider identities; refresh provider data"
        );
        writeln!(source, "n{index}[\"{}\"]", label(text)?)?;
    }
    for (from, to, text, informational) in edges {
        let from = identities.get(from.as_str()).ok_or_else(|| {
            anyhow::anyhow!("Dependency diagram has an unavailable source; refresh provider data")
        })?;
        let to = identities.get(to.as_str()).ok_or_else(|| {
            anyhow::anyhow!("Dependency diagram has an unavailable target; refresh provider data")
        })?;
        writeln!(
            source,
            "n{from} {}|\"{}\"| n{to}",
            if *informational { "-.->" } else { "-->" },
            label(text)?
        )?;
    }
    let source = format!(
        "{coverage}\n\nRead-only full admitted graph. Dotted edges are constraints or mediated Maven occurrences. Unresolved Gradle targets are placeholders. Tree facts retain keyboard selection and source navigation. Fact/scope/search filters apply to the tree only. Full diagram parity remains unverified.\n\n```mermaid\n{source}```\n"
    );
    anyhow::ensure!(
        source.len() <= 128 * 1024,
        "Dependency diagram exceeds source budget; use current tree facts"
    );
    Ok(source)
}

fn dependency_diagram_source(
    snapshot: &JdtlsProjectSnapshot,
    current: bool,
    view: BuildToolView,
    selection: &GraphSelection,
) -> anyhow::Result<Option<String>> {
    if !selection.matches(snapshot, current) {
        return Ok(None);
    }
    for module in &snapshot.projects {
        match &selection.target {
            GraphQueryTarget::Maven(pom) if view != BuildToolView::Gradle => {
                let Some(maven) = module.maven.as_ref().filter(|maven| maven.pom == *pom) else {
                    continue;
                };
                let Some(graph) = &maven.dependency_graph else {
                    return Ok(None);
                };
                anyhow::ensure!(
                    graph.nodes.len() <= 64 && graph.edges.len() <= 128,
                    "Dependency diagram exceeds presentation limits; use current tree facts"
                );
                let coordinates = graph
                    .components
                    .iter()
                    .map(|component| {
                        (
                            component.id.as_str(),
                            format!(
                                "{}:{}:{}:{}:{}",
                                component.group_id,
                                component.artifact_id,
                                component.extension,
                                component.classifier,
                                component.version
                            ),
                        )
                    })
                    .collect::<HashMap<_, _>>();
                let nodes = graph
                    .nodes
                    .iter()
                    .map(|node| {
                        (
                            node.id.clone(),
                            coordinates
                                .get(node.component.as_str())
                                .cloned()
                                .unwrap_or_else(|| "Unavailable component".into()),
                        )
                    })
                    .collect::<Vec<_>>();
                let edges = graph
                    .edges
                    .iter()
                    .map(|edge| {
                        (
                            edge.from_node.clone(),
                            edge.target_node.clone(),
                            format!(
                                "scope {}; mediated {}; optional {}",
                                edge.scope, edge.omitted, edge.optional
                            ),
                            edge.omitted,
                        )
                    })
                    .collect::<Vec<_>>();
                return dependency_diagram_markdown(&nodes, &edges, &format!("Maven occurrence ancestry; structural coverage {}; collection complete {}; exact cycle occurrences {}; origins and artifacts unavailable.", graph.structural_coverage_known, graph.collection_complete, graph.cycle_occurrences_known)).map(Some);
            }
            GraphQueryTarget::Gradle {
                module: selected_module,
                configuration,
            } if view != BuildToolView::Maven && module.module_root == *selected_module => {
                let Some(graph) = module
                    .gradle
                    .as_ref()
                    .and_then(|gradle| gradle.dependency_graph.as_ref())
                    .filter(|graph| graph.resolution.configuration == *configuration)
                else {
                    return Ok(None);
                };
                anyhow::ensure!(
                    graph.resolution.components.len() <= 64 && graph.resolution.edges.len() <= 128,
                    "Dependency diagram exceeds presentation limits; use current tree facts"
                );
                let mut nodes = graph
                    .resolution
                    .components
                    .iter()
                    .map(|component| (component.id.clone(), gradle_component_label(component)))
                    .collect::<Vec<_>>();
                let mut edges = Vec::new();
                for (index, edge) in graph.resolution.edges.iter().enumerate() {
                    let target = if let Some(selected) = &edge.selected {
                        selected.clone()
                    } else {
                        let identity = format!("diagram-unresolved-{index}");
                        anyhow::ensure!(
                            !nodes.iter().any(|(id, _)| id == &identity),
                            "Unresolved placeholder conflicts with provider identity; use current tree facts"
                        );
                        nodes.push((
                            identity.clone(),
                            format!(
                                "Unresolved; failure {}",
                                edge.failure_category.as_deref().unwrap_or("unavailable")
                            ),
                        ));
                        identity
                    };
                    edges.push((
                        edge.from.clone(),
                        target,
                        if edge.constraint {
                            "constraint (not dependency inclusion)"
                        } else if edge.resolved {
                            "dependency"
                        } else {
                            "unresolved dependency"
                        }
                        .into(),
                        edge.constraint,
                    ));
                }
                return dependency_diagram_markdown(&nodes, &edges, &format!("Gradle configuration graph; structural coverage {}; resolution complete {}; origins and artifacts unavailable.", graph.resolution.structural_coverage_known, graph.resolution.resolution_complete)).map(Some);
            }
            _ => {}
        }
    }
    Ok(None)
}

fn graph_component_row_id(id: LanguageToolNodeId) -> LanguageToolNodeId {
    LanguageToolNodeId(format!("{}:occurrence:0", id.0))
}

fn graph_inspection_edges<'a>(
    root: &'a str,
    selected: &'a str,
    edges: &[(&'a str, Option<&'a str>)],
    mode: GraphInspectionMode,
) -> (Vec<usize>, Option<usize>) {
    if mode == GraphInspectionMode::Neighbours {
        return (
            edges
                .iter()
                .enumerate()
                .filter_map(|(index, (from, target))| {
                    (*from == selected || *target == Some(selected)).then_some(index)
                })
                .collect(),
            None,
        );
    }
    let distances = |start: &'a str, reverse: bool| {
        let mut adjacency = HashMap::<&str, Vec<&str>>::new();
        for (from, target) in edges {
            if let Some(target) = target {
                let (from, target) = if reverse {
                    (*target, *from)
                } else {
                    (*from, *target)
                };
                adjacency.entry(from).or_default().push(target);
            }
        }
        let mut distances = HashMap::from([(start, 0_usize)]);
        let mut pending = VecDeque::from([start]);
        while let Some(node) = pending.pop_front() {
            let Some(distance) = distances.get(node).copied() else {
                continue;
            };
            for target in adjacency.get(node).into_iter().flatten() {
                if !distances.contains_key(target) {
                    distances.insert(*target, distance + 1);
                    pending.push_back(*target);
                }
            }
        }
        distances
    };
    let forward = distances(root, false);
    let backward = distances(selected, true);
    let shortest = forward.get(selected).copied();
    let retained = edges
        .iter()
        .enumerate()
        .filter_map(|(index, (from, target))| {
            let length = forward
                .get(from)?
                .checked_add(1)?
                .checked_add(*backward.get(target.as_ref()?)?)?;
            (Some(length) == shortest).then_some(index)
        })
        .collect();
    (retained, shortest)
}

fn gradle_inspection_relations(
    graph: &project::jvm_project::GradleDependencyGraph,
    mode: GraphInspectionMode,
) -> Vec<(&str, Option<&str>)> {
    graph
        .resolution
        .edges
        .iter()
        .map(|edge| {
            // A constraint selects a version but does not introduce a dependency.
            let target = if mode == GraphInspectionMode::ShortestRootPaths && edge.constraint {
                None
            } else {
                edge.selected.as_deref()
            };
            (edge.from.as_str(), target)
        })
        .collect()
}

fn graph_inspection_row(
    snapshot: &JdtlsProjectSnapshot,
    view: BuildToolView,
    inspection: &GraphInspection,
) -> Option<(LanguageToolNodeId, LanguageToolNode)> {
    use project::jvm_project::GradleDependencySelector;
    for module in &snapshot.projects {
        let module_id = jdtls_module_node_id(snapshot, &module.module_root, view);
        let (graph_id, root, relations, labels, edge_rows, coverage) = match &inspection
            .selection
            .target
        {
            GraphQueryTarget::Maven(pom) if view != BuildToolView::Gradle => {
                let Some(maven) = module.maven.as_ref().filter(|maven| maven.pom == *pom) else {
                    continue;
                };
                let graph = maven.dependency_graph.as_ref()?;
                if !graph.nodes.iter().any(|node| node.id == inspection.node) {
                    return None;
                }
                let coordinates = graph
                    .components
                    .iter()
                    .map(|component| {
                        (
                            component.id.as_str(),
                            format!(
                                "{}:{}:{}:{}:{}",
                                component.group_id,
                                component.artifact_id,
                                component.extension,
                                component.classifier,
                                component.version
                            ),
                        )
                    })
                    .collect::<HashMap<_, _>>();
                let labels = graph
                    .nodes
                    .iter()
                    .map(|node| {
                        (
                            node.id.as_str(),
                            coordinates
                                .get(node.component.as_str())
                                .cloned()
                                .unwrap_or_else(|| "Unavailable component".into()),
                        )
                    })
                    .collect::<HashMap<_, _>>();
                let relations = graph
                    .edges
                    .iter()
                    .map(|edge| (edge.from_node.as_str(), Some(edge.target_node.as_str())))
                    .collect::<Vec<_>>();
                let edge_rows = graph.edges.iter().map(|edge| (
                    graph_fact_id(&module_id, "inspection-maven-edge", &[&edge.from_node, &edge.target_node, &edge.selected_node, &edge.position.to_string()]),
                    format!("{} → {}", labels.get(edge.from_node.as_str()).map(String::as_str).unwrap_or("Unavailable occurrence"), labels.get(edge.target_node.as_str()).map(String::as_str).unwrap_or("Unavailable occurrence")),
                    format!("From occurrence {}; requested occurrence {}; selected occurrence {}; scope {}; omitted by mediation {}; optional {}; source origin unavailable", edge.from_node, edge.target_node, edge.selected_node, edge.scope, edge.omitted, edge.optional),
                )).collect::<Vec<_>>();
                (
                    graph_fact_id(&module_id, "maven", &[]),
                    graph.root_node.as_str(),
                    relations,
                    labels,
                    edge_rows,
                    format!(
                        "Maven occurrence ancestry; structural coverage {}; collection complete {}; exact cycle occurrences {}; source origins and artifacts unavailable",
                        graph.structural_coverage_known,
                        graph.collection_complete,
                        graph.cycle_occurrences_known
                    ),
                )
            }
            GraphQueryTarget::Gradle {
                module: selected_module,
                configuration,
            } if view != BuildToolView::Maven && *selected_module == module.module_root => {
                let graph = module.gradle.as_ref()?.dependency_graph.as_ref()?;
                let resolution = &graph.resolution;
                if resolution.configuration != *configuration
                    || !resolution
                        .components
                        .iter()
                        .any(|component| component.id == inspection.node)
                {
                    return None;
                }
                let labels = resolution
                    .components
                    .iter()
                    .map(|component| (component.id.as_str(), gradle_component_label(component)))
                    .collect::<HashMap<_, _>>();
                let relations = gradle_inspection_relations(graph, inspection.mode);
                let edge_rows = resolution.edges.iter().map(|edge| {
                    let requested = match &edge.requested {
                        GradleDependencySelector::Module { group, module, version, .. } => format!("{group}:{module}:{version}"),
                        GradleDependencySelector::Project { build_path, project_path } => format!("Project {project_path}; build {build_path}"),
                    };
                    (graph_fact_id(&module_id, "inspection-gradle-edge", &[configuration, &edge.from, &format!("{:?}", edge.requested), edge.selected.as_deref().unwrap_or(""), if edge.constraint { "constraint" } else { "dependency" }]),
                        format!("{} → {}", labels.get(edge.from.as_str()).map(String::as_str).unwrap_or("Unavailable component"), edge.selected.as_deref().and_then(|id| labels.get(id)).map(String::as_str).unwrap_or("Unresolved")),
                        format!("Requested {requested}; constraint {}; resolved {}; variant {}; failure {}; source origin unavailable", edge.constraint, edge.resolved, edge.selected_variant.as_deref().unwrap_or("unavailable"), edge.failure_category.as_deref().unwrap_or("none")))
                }).collect::<Vec<_>>();
                (
                    graph_fact_id(&module_id, "gradle", &[]),
                    resolution.root_component.as_str(),
                    relations,
                    labels,
                    edge_rows,
                    format!(
                        "Gradle selected-component relations; constraints are neighbours but not dependency inclusion paths; structural coverage {}; resolution complete {}; source origins and artifacts unavailable",
                        resolution.structural_coverage_known, resolution.resolution_complete
                    ),
                )
            }
            _ => continue,
        };
        let (indices, distance) =
            graph_inspection_edges(root, &inspection.node, &relations, inspection.mode);
        let mut nodes = HashSet::from([inspection.node.as_str()]);
        if inspection.mode == GraphInspectionMode::ShortestRootPaths {
            nodes.insert(root);
        }
        let mut rows = Vec::new();
        for index in indices {
            if let Some((from, target)) = relations.get(index) {
                nodes.insert(*from);
                if let Some(target) = target {
                    nodes.insert(*target);
                }
            }
            if let Some((id, label, detail)) = edge_rows.get(index) {
                rows.push(graph_fact_row(
                    id.clone(),
                    label.clone(),
                    detail.clone(),
                    Vec::new(),
                ));
            }
        }
        let count = rows.len();
        let components = nodes
            .into_iter()
            .map(|node| {
                graph_fact_row(
                    graph_fact_id(&module_id, "inspection-node", &[&inspection.node, node]),
                    labels
                        .get(node)
                        .cloned()
                        .unwrap_or_else(|| "Unavailable component".into()),
                    format!(
                        "Provider identity {node}; {}",
                        if node == root {
                            "Root"
                        } else if node == inspection.node {
                            "Selected"
                        } else {
                            "Path or neighbour"
                        }
                    ),
                    Vec::new(),
                )
            })
            .collect();
        let path_status = match (inspection.mode, distance) {
            (GraphInspectionMode::Neighbours, _) => "Incident edges only; unresolved outgoing edges retained".to_string(),
            (GraphInspectionMode::ShortestRootPaths, Some(distance)) => format!("Shortest distance {distance}; all equal-shortest paths retained without recursive expansion"),
            (GraphInspectionMode::ShortestRootPaths, None) => "No root path in admitted facts; incomplete coverage cannot prove absence; repair provider/cache settings and recollect".into(),
        };
        return Some((
            graph_id,
            graph_fact_row(
                graph_fact_id(&module_id, "inspection", &[&inspection.node]),
                inspection.mode.label().into(),
                format!(
                    "Selection {}; {count} retained edges; {path_status}; {coverage}; uses full admitted graph independently of ordinary fact/scope facets; literal search still applies",
                    labels
                        .get(inspection.node.as_str())
                        .map(String::as_str)
                        .unwrap_or("Unavailable selection")
                ),
                vec![
                    graph_fact_row(
                        graph_fact_id(&module_id, "inspection-nodes", &[&inspection.node]),
                        "Inspected identities".into(),
                        "Provider-specific components or occurrences; no coordinate deduplication"
                            .into(),
                        components,
                    ),
                    graph_fact_row(
                        graph_fact_id(&module_id, "inspection-edges", &[&inspection.node]),
                        "Inspected edges".into(),
                        "Fixed-depth provider facts; no script execution or dependency resolution"
                            .into(),
                        rows,
                    ),
                ],
            ),
        ));
    }
    None
}

fn append_graph_inspection(
    rows: &mut [LanguageToolNode],
    graph: &LanguageToolNodeId,
    inspection: &LanguageToolNode,
) -> bool {
    for row in rows {
        if row.id == *graph {
            row.children.push(inspection.clone());
            return true;
        }
        if append_graph_inspection(&mut row.children, graph, inspection) {
            return true;
        }
    }
    false
}

fn jdtls_module_node_id(
    snapshot: &JdtlsProjectSnapshot,
    module: &ProjectPath,
    view: BuildToolView,
) -> LanguageToolNodeId {
    let id = format!(
        "jdtls:{}:{}",
        snapshot.worktree_id,
        module.path.as_unix_str()
    );
    LanguageToolNodeId(if view == BuildToolView::Dependencies {
        id
    } else {
        format!("{id}:{}", view.label())
    })
}

fn gradle_configuration_node_id(
    module: &LanguageToolNodeId,
    configuration: &str,
) -> LanguageToolNodeId {
    LanguageToolNodeId(format!(
        "gradle-configuration:{}:{}:{}:{}",
        module.0.len(),
        module.0,
        configuration.len(),
        configuration
    ))
}

fn graph_selections_for_snapshot(
    snapshot: &JdtlsProjectSnapshot,
    current: bool,
    view: BuildToolView,
) -> HashMap<LanguageToolNodeId, GraphSelection> {
    let mut selections = HashMap::new();
    if !current {
        return selections;
    }
    for module in &snapshot.projects {
        let id = jdtls_module_node_id(snapshot, &module.module_root, view);
        if view != BuildToolView::Gradle
            && let Some(maven) = &module.maven
        {
            let selection = GraphSelection {
                server_id: snapshot.server_id,
                generation: snapshot.generation,
                target: GraphQueryTarget::Maven(maven.pom.clone()),
            };
            selections.insert(id.clone(), selection.clone());
            selections.insert(graph_fact_id(&id, "maven", &[]), selection);
        }
        if view != BuildToolView::Maven
            && let Some(gradle) = &module.gradle
        {
            for configuration in gradle
                .configurations
                .iter()
                .filter(|configuration| gradle.configurations_known && configuration.resolvable)
            {
                let selection = GraphSelection {
                    server_id: snapshot.server_id,
                    generation: snapshot.generation,
                    target: GraphQueryTarget::Gradle {
                        module: module.module_root.clone(),
                        configuration: configuration.name.clone(),
                    },
                };
                if view == BuildToolView::Gradle {
                    selections.insert(
                        gradle_configuration_node_id(&id, &configuration.name),
                        selection.clone(),
                    );
                }
                if gradle
                    .dependency_graph
                    .as_ref()
                    .is_some_and(|graph| graph.resolution.configuration == configuration.name)
                {
                    selections.insert(graph_fact_id(&id, "gradle", &[]), selection);
                }
            }
        }
    }
    selections
}

struct DependencyProjection {
    provider: DependencyProvider,
    server_id: lsp::LanguageServerId,
    generation: u64,
    partial: bool,
    current: bool,
    rows: Vec<LanguageToolNode>,
    navigation_targets: HashMap<LanguageToolNodeId, DependencyNavigationTarget>,
    graph_selections: HashMap<LanguageToolNodeId, GraphSelection>,
}

fn provider_projections_for_view(
    jdtls_snapshot: Option<(&JdtlsProjectSnapshot, bool)>,
    kotlin_snapshot: Option<(&KotlinWorkspaceSnapshot, bool)>,
    view: BuildToolView,
    preferences: &BuildViewPreferences,
) -> (Vec<DependencyProjection>, bool) {
    let mut projections = Vec::new();
    if let Some((snapshot, current)) = jdtls_snapshot {
        let (rows, navigation_targets) = project_rows_for_view(snapshot, view, preferences);
        projections.push(DependencyProjection {
            provider: DependencyProvider::Jdtls,
            server_id: snapshot.server_id,
            generation: snapshot.generation,
            partial: snapshot.partial,
            current,
            rows,
            navigation_targets,
            graph_selections: graph_selections_for_snapshot(snapshot, current, view),
        });
    }
    if let Some((snapshot, current)) = kotlin_snapshot {
        let (mut rows, navigation_targets) = kotlin_rows_for_view(snapshot, view);
        if preferences.group_modules(view) {
            let groups = navigation_targets
                .iter()
                .filter_map(|(id, target)| {
                    let DependencyNavigationTarget::ModuleManifest { root, .. } = target else {
                        return None;
                    };
                    let directory = root
                        .path
                        .parent()
                        .map(|parent| parent.as_unix_str())
                        .unwrap_or_default();
                    let group = directory_module_group(
                        "kotlin",
                        view,
                        root.worktree_id,
                        directory,
                        if view == BuildToolView::Gradle {
                            "Provider module directory; logical Gradle grouping unavailable"
                        } else {
                            "Provider module directory"
                        },
                    );
                    Some((id.clone(), group))
                })
                .collect();
            rows = group_module_rows(rows, groups);
        }
        projections.push(DependencyProjection {
            provider: DependencyProvider::Kotlin,
            server_id: snapshot.server_id,
            generation: snapshot.generation,
            partial: snapshot.partial || view != BuildToolView::Dependencies,
            current,
            rows,
            navigation_targets,
            graph_selections: HashMap::new(),
        });
    }
    let current_available = projections.iter().any(|projection| projection.current);
    let original_count = projections.len();
    if current_available {
        projections.retain(|projection| projection.current);
    }
    if projections.len() > 1 {
        for projection in &mut projections {
            let provider_name = match projection.provider {
                DependencyProvider::Jdtls => "JDTLS",
                DependencyProvider::Kotlin => "Kotlin LSP",
            };
            for row in &mut projection.rows {
                row.label = format!("{provider_name}: {}", row.label);
            }
        }
    }
    let omitted_stale = current_available && original_count > projections.len();
    (projections, omitted_stale)
}

impl DependencyNavigation {
    fn matches(
        &self,
        server_id: lsp::LanguageServerId,
        worktree_id: settings::WorktreeId,
        generation: u64,
        current: bool,
    ) -> bool {
        current
            && server_id == self.server_id
            && generation == self.generation
            && worktree_id == self.path.worktree_id
    }
}

impl DependenciesPanel {
    #[cfg(feature = "test-support")]
    pub fn snapshot_for_test(
        &self,
    ) -> (
        language_tool_tree::LanguageToolTreeStatus,
        Vec<language_tool_tree::VisibleLanguageToolRow>,
        HashMap<LanguageToolNodeId, ProjectPath>,
    ) {
        (
            self.host.status().clone(),
            self.host.visible_rows().to_vec(),
            self.navigation
                .iter()
                .map(|(id, navigation)| (id.clone(), navigation.path.clone()))
                .collect(),
        )
    }

    pub async fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> anyhow::Result<Entity<Self>> {
        let workspace_handle = workspace.clone();
        workspace.update_in(&mut cx, |workspace, window, cx| {
            let project = workspace.project().clone();
            let preferences_key = workspace_scoped_state_key(
                VIEW_STATE_KEY, workspace.database_id(), workspace.session_id().as_deref(),
            );
            let restored = preferences_key.as_ref().map(|key| KeyValueStore::global(cx).read_kvp(key)).transpose()
                .and_then(|serialized| BuildViewPreferences::restore(serialized.flatten().as_deref()));
            let (preferences, preferences_notice) = match restored {
                Ok(preferences) => (preferences, None),
                Err(error) => {
                    log::warn!("Failed to restore Groovy build view preferences: {error:#}");
                    (BuildViewPreferences::default(), Some("Build view preferences could not be restored. Defaults are shown; change an option to save new preferences.".into()))
                }
            };
            cx.new(|cx: &mut Context<DependenciesPanel>| {
                let lsp_store = project.read(cx).lsp_store();
                let worktree_store = project.read(cx).worktree_store();
                let lsp_subscription = cx.observe(&lsp_store, |panel, _, cx| {
                    panel.on_lsp_store_changed(cx);
                });
                let worktree_subscription = cx.observe(&worktree_store, |panel, _, cx| {
                    panel.invalidate(cx);
                });
                let entry_subscription = cx.subscribe(&project, |panel, _, event, cx| {
                    if matches!(event, project::Event::WorktreeUpdatedEntries(_, _)) {
                        panel.invalidate(cx);
                    }
                });
                let graph_search = cx.new(|cx| {
                    let mut editor = Editor::single_line(window, cx);
                    editor.set_placeholder_text("Search admitted graph facts", window, cx);
                    editor
                });
                let graph_search_subscription = cx.subscribe(&graph_search, |panel, _, event, cx| {
                    if matches!(event, EditorEvent::BufferEdited) {
                        panel.rebuild(cx);
                    }
                });
                let mut subscriptions = vec![lsp_subscription, worktree_subscription, entry_subscription, graph_search_subscription];
                if let Some(remote_client) = project.read(cx).remote_client() {
                    subscriptions.push(cx.subscribe(&remote_client, |panel, _, event, cx| {
                        if let remote::RemoteClientEvent::ConnectionStateChanged(state) = event {
                            if *state == remote::ConnectionState::Connected && panel.active {
                                panel.refresh(cx);
                            } else {
                                panel.invalidate(cx);
                            }
                        }
                    }));
                }
                if let Some(trusted_worktrees) = TrustedWorktrees::try_get_global(cx) {
                    subscriptions.push(cx.subscribe(&trusted_worktrees, |panel, _, _, cx| {
                        panel.invalidate(cx);
                    }));
                }
                let focus_handle = cx.focus_handle();
                let provider_servers = Self::provider_servers_for_project(&project, cx);
                Self {
                    project,
                    workspace: workspace_handle,
                    focus_handle: focus_handle.clone(),
                    host: LanguageToolTreeHost::with_focus_handle(focus_handle),
                    provider_servers,
                    navigation: HashMap::new(),
                    graph_selections: HashMap::new(),
                    operation_selections: HashMap::new(),
                    operation_notice: None,
                    ordered_operations: Vec::new(),
                    graph_query_token: None,
                    graph_query: None,
                    graph_notice: None,
                    build_phase_token: None,
                    build_phase_waiter: None,
                    build_phase_notice: None,
                    graph_search,
                    graph_inspection: None,
                    graph_diagram_selection: None,
                    graph_diagram: None,
                    position: preferences.dock_position.into(),
                    active: false,
                    view: preferences.view,
                    preferences,
                    preferences_key,
                    preferences_notice,
                    preferences_write: None,
                    preferences_generation: 0,
                    show_options: false,
                    _subscriptions: subscriptions,
                }
            })
        })
    }

    pub fn toggle_focus(
        workspace: &mut Workspace,
        _: &ToggleDependenciesPanel,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        workspace.toggle_panel_focus::<Self>(window, cx);
    }

    pub fn show_gradle(
        workspace: &mut Workspace,
        _: &ShowGradle,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        if let Some(panel) = workspace.focus_panel::<Self>(window, cx) {
            panel.update(cx, |panel, cx| {
                panel.set_view(BuildToolView::Gradle, window, cx)
            });
        }
    }

    pub fn show_maven(
        workspace: &mut Workspace,
        _: &ShowMaven,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        if let Some(panel) = workspace.focus_panel::<Self>(window, cx) {
            panel.update(cx, |panel, cx| {
                panel.set_view(BuildToolView::Maven, window, cx)
            });
        }
    }

    fn set_view(&mut self, view: BuildToolView, window: &mut Window, cx: &mut Context<Self>) {
        self.graph_inspection = None;
        self.graph_diagram_selection = None;
        self.graph_diagram = None;
        self.view = view;
        self.preferences.view = view;
        self.persist_preferences(cx);
        self.rebuild(cx);
        window.focus(&self.focus_handle, cx);
    }

    fn selected_graph_inspection(
        &self,
        mode: GraphInspectionMode,
        cx: &App,
    ) -> Option<GraphInspection> {
        if self.host_unavailable(cx) {
            return None;
        }
        let selected = self.host.selected()?;
        let project = self.project.read(cx);
        let worktrees = project.worktree_store();
        let trusted = TrustedWorktrees::try_get_global(cx)?;
        let restricted = trusted
            .read(cx)
            .restricted_worktrees(&worktrees, cx)
            .into_iter()
            .map(|(id, _)| id)
            .collect::<HashSet<_>>();
        let store = project.lsp_store();
        let store = store.read(cx);
        for worktree in worktrees.read(cx).worktrees() {
            let worktree = worktree.read(cx);
            if !worktree.is_visible() || restricted.contains(&worktree.id()) {
                continue;
            }
            let Some((snapshot, true)) = store.jdtls_project_snapshot(worktree.id()) else {
                continue;
            };
            for module in &snapshot.projects {
                let id = jdtls_module_node_id(snapshot, &module.module_root, self.view);
                if self.view != BuildToolView::Gradle
                    && let Some(maven) = &module.maven
                    && let Some(graph) = &maven.dependency_graph
                {
                    for node in &graph.nodes {
                        if *selected
                            == graph_component_row_id(graph_fact_id(&id, "maven-node", &[&node.id]))
                        {
                            return Some(GraphInspection {
                                selection: GraphSelection {
                                    server_id: snapshot.server_id,
                                    generation: snapshot.generation,
                                    target: GraphQueryTarget::Maven(maven.pom.clone()),
                                },
                                node: node.id.clone(),
                                mode,
                            });
                        }
                    }
                }
                if self.view != BuildToolView::Maven
                    && let Some(gradle) = &module.gradle
                    && let Some(graph) = &gradle.dependency_graph
                {
                    for component in &graph.resolution.components {
                        if *selected
                            == graph_component_row_id(graph_fact_id(
                                &id,
                                "gradle-component",
                                &[&graph.resolution.configuration, &component.id],
                            ))
                        {
                            return Some(GraphInspection {
                                selection: GraphSelection {
                                    server_id: snapshot.server_id,
                                    generation: snapshot.generation,
                                    target: GraphQueryTarget::Gradle {
                                        module: module.module_root.clone(),
                                        configuration: graph.resolution.configuration.clone(),
                                    },
                                },
                                node: component.id.clone(),
                                mode,
                            });
                        }
                    }
                }
            }
        }
        None
    }

    fn inspect_selected_graph(&mut self, mode: GraphInspectionMode, cx: &mut Context<Self>) {
        let Some(inspection) = self.selected_graph_inspection(mode, cx) else {
            self.show_graph_error(anyhow::anyhow!("Graph inspection requires a current Gradle component or Maven occurrence; refresh and expand admitted graph facts, then select a dependency row"), cx);
            return;
        };
        self.graph_notice = None;
        self.graph_inspection = Some(inspection);
        self.rebuild(cx);
        self.host.expand_all();
        cx.notify();
    }

    fn clear_graph_inspection(&mut self, cx: &mut Context<Self>) {
        self.graph_inspection = None;
        self.rebuild(cx);
    }

    fn show_dependency_diagram(&mut self, cx: &mut Context<Self>) {
        let selection = self.selected_graph().or_else(|| {
            self.selected_graph_inspection(GraphInspectionMode::Neighbours, cx)
                .map(|inspection| inspection.selection)
        });
        let Some(selection) = selection else {
            self.show_graph_error(anyhow::anyhow!("Select a current Maven occurrence or Gradle configuration/component, refresh provider graph facts and retry Show diagram"), cx);
            return;
        };
        self.graph_notice = None;
        self.graph_diagram_selection = Some(selection);
        self.graph_diagram = None;
        self.rebuild(cx);
        if self.graph_diagram.is_none() && self.graph_notice.is_none() {
            self.show_graph_error(anyhow::anyhow!("A current admitted graph is required. Trust the project, refresh and collect graph facts before retrying Show diagram"), cx);
        }
    }

    fn close_dependency_diagram(&mut self, cx: &mut Context<Self>) {
        self.graph_diagram_selection = None;
        self.graph_diagram = None;
        cx.notify();
    }

    fn clear_graph_fact_search(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.graph_search
            .update(cx, |editor, cx| editor.set_text("", window, cx));
        window.focus(&self.graph_search.read(cx).focus_handle(cx), cx);
    }

    fn cycle_graph_fact_filter(&mut self, cx: &mut Context<Self>) {
        self.preferences.graph_fact_filter = self.preferences.graph_fact_filter.next();
        self.persist_preferences(cx);
        self.rebuild(cx);
    }

    fn cycle_maven_graph_scope(&mut self, cx: &mut Context<Self>) {
        self.preferences.maven_graph_scope = self.preferences.maven_graph_scope.next();
        self.persist_preferences(cx);
        self.rebuild(cx);
    }

    fn toggle_option(&mut self, option: BuildViewOption, cx: &mut Context<Self>) {
        match option {
            BuildViewOption::ShowBasicMavenPhases => {
                self.preferences.show_basic_maven_phases =
                    !self.preferences.show_basic_maven_phases;
            }
            BuildViewOption::GroupModules => match self.view {
                BuildToolView::Gradle => {
                    self.preferences.group_gradle_modules = !self.preferences.group_gradle_modules
                }
                BuildToolView::Maven => {
                    self.preferences.group_maven_modules = !self.preferences.group_maven_modules
                }
                BuildToolView::Dependencies => return,
            },
            BuildViewOption::GroupGradleTasks => {
                self.preferences.group_gradle_tasks = !self.preferences.group_gradle_tasks
            }
            BuildViewOption::ShowMavenArtifactId => {
                self.preferences.show_maven_artifact_id = !self.preferences.show_maven_artifact_id
            }
            BuildViewOption::ShowMavenVersion => {
                self.preferences.show_maven_version = !self.preferences.show_maven_version
            }
            BuildViewOption::ShowToolbar => match self.view {
                BuildToolView::Gradle => {
                    self.preferences.show_gradle_toolbar = !self.preferences.show_gradle_toolbar
                }
                BuildToolView::Maven => {
                    self.preferences.show_maven_toolbar = !self.preferences.show_maven_toolbar
                }
                BuildToolView::Dependencies => return,
            },
        }
        self.persist_preferences(cx);
        self.rebuild(cx);
    }

    fn persist_preferences(&mut self, cx: &mut Context<Self>) {
        let Some(key) = self.preferences_key.clone() else {
            self.preferences_notice = Some("Build view preferences cannot be saved until this workspace has a persistent identity.".into());
            return;
        };
        let serialized = match serde_json::to_string(&self.preferences) {
            Ok(serialized) => serialized,
            Err(error) => {
                log::warn!("Failed to serialize Groovy build view preferences: {error:#}");
                self.preferences_notice = Some(
                    "Build view preferences could not be saved. Change an option to retry.".into(),
                );
                return;
            }
        };
        let previous = self.preferences_write.take();
        let store = KeyValueStore::global(cx);
        self.preferences_generation = self.preferences_generation.wrapping_add(1);
        let generation = self.preferences_generation;
        self.preferences_write = Some(cx.spawn(async move |panel, cx| {
            // Preserve write order when several toggles change while the database is busy.
            if let Some(previous) = previous {
                previous.await;
            }
            let result = store.write_kvp(key, serialized).await;
            let notice = match result {
                Ok(()) => None,
                Err(error) => {
                    log::warn!("Failed to save Groovy build view preferences: {error:#}");
                    Some(
                        "Build view preferences could not be saved. Change an option to retry."
                            .into(),
                    )
                }
            };
            if let Err(error) = panel.update(cx, |panel, cx| {
                if panel.preferences_generation == generation {
                    panel.preferences_notice = notice;
                    cx.notify();
                }
            }) {
                log::debug!("Groovy panel closed after preference write: {error:#}");
            }
        }));
    }

    fn invalidate(&mut self, cx: &mut Context<Self>) {
        if self.active {
            self.rebuild(cx);
        } else {
            self.host.mark_dirty();
        }
    }

    fn provider_servers_for_project(
        project: &Entity<Project>,
        cx: &App,
    ) -> HashSet<(lsp::LanguageServerId, settings::WorktreeId)> {
        let lsp_store = project.read(cx).lsp_store();
        lsp_store
            .read(cx)
            .language_server_statuses()
            .filter_map(|(server_id, status)| {
                let worktree_id = status.worktree?;
                matches!(status.name.as_ref(), "jdtls" | "kotlin-lsp")
                    .then_some((server_id, worktree_id))
            })
            .collect()
    }

    fn on_lsp_store_changed(&mut self, cx: &mut Context<Self>) {
        let provider_servers = Self::provider_servers_for_project(&self.project, cx);
        let server_added = provider_servers
            .iter()
            .any(|server_id| !self.provider_servers.contains(server_id));
        self.provider_servers = provider_servers;
        if server_added && self.active && !self.host_unavailable(cx) {
            self.refresh(cx);
        } else {
            self.invalidate(cx);
        }
    }

    fn host_unavailable(&self, cx: &App) -> bool {
        let project = self.project.read(cx);
        project.is_disconnected(cx)
            || project
                .remote_connection_state(cx)
                .is_some_and(|state| state != remote::ConnectionState::Connected)
    }

    fn rebuild(&mut self, cx: &mut Context<Self>) {
        if self.host_unavailable(cx) {
            self.navigation.clear();
            self.graph_inspection = None;
            self.graph_diagram_selection = None;
            self.graph_diagram = None;
            self.graph_selections.clear();
            self.operation_selections.clear();
            self.ordered_operations.clear();
            self.host.replace_snapshot(LanguageToolSnapshot {
                roots: Vec::new(),
                status: LanguageToolProviderStatus::Disconnected(
                    "The project host connection is unavailable; reconnect to refresh JVM dependencies."
                        .to_string(),
                ),
            });
            cx.notify();
            return;
        }
        let Some(trusted_worktrees) = TrustedWorktrees::try_get_global(cx) else {
            self.navigation.clear();
            self.graph_inspection = None;
            self.graph_diagram_selection = None;
            self.graph_diagram = None;
            self.graph_selections.clear();
            self.operation_selections.clear();
            self.ordered_operations.clear();
            self.host.replace_snapshot(LanguageToolSnapshot {
                roots: Vec::new(),
                status: LanguageToolProviderStatus::Restricted(
                    "Project trust is unavailable; restart the app and reopen the project before loading JVM dependencies."
                        .to_string(),
                ),
            });
            cx.notify();
            return;
        };
        let project = self.project.read(cx);
        let lsp_store = project.lsp_store();
        let worktree_store = project.worktree_store();
        let lsp_store = lsp_store.read(cx);
        let mut roots = Vec::new();
        let mut navigation = HashMap::new();
        let mut graph_selections = HashMap::new();
        let mut operation_selections = HashMap::new();
        let mut inspection_current = false;
        let mut diagram_source = None;
        let mut partial = false;
        let mut stale = false;
        let mut query_failed = false;
        let mut import_failed = false;
        let mut jdtls_import_failed = false;
        let mut query_pending = false;
        let mut local_worktrees = 0;
        let mut remote_worktrees = 0;
        let mut local_without_snapshot = 0;
        let mut remote_without_snapshot = 0;
        let restricted_worktrees = trusted_worktrees
            .read(cx)
            .restricted_worktrees(&worktree_store, cx)
            .into_iter()
            .map(|(worktree_id, _)| worktree_id)
            .collect::<HashSet<_>>();
        for worktree in worktree_store.read(cx).worktrees() {
            let worktree = worktree.read(cx);
            if !worktree.is_visible() {
                continue;
            }
            if worktree.is_local() {
                local_worktrees += 1;
            } else {
                remote_worktrees += 1;
            }
            if restricted_worktrees.contains(&worktree.id()) {
                continue;
            }
            let jdtls_failed = lsp_store.jdtls_project_query_failed(worktree.id());
            jdtls_import_failed |= lsp_store.jdtls_project_import_failed(worktree.id());
            let kotlin_failed = lsp_store.kotlin_workspace_query_failed(worktree.id());
            let kotlin_import_failed = lsp_store.kotlin_workspace_import_failed(worktree.id());
            let jdtls_pending = lsp_store.jdtls_project_query_pending(worktree.id());
            let kotlin_pending = lsp_store.kotlin_workspace_query_pending(worktree.id());
            let kotlin_snapshot = lsp_store.kotlin_workspace_snapshot(worktree.id());
            let jdtls_snapshot = lsp_store.jdtls_project_snapshot(worktree.id());
            let (projections, omitted_stale) = provider_projections_for_view(
                jdtls_snapshot,
                kotlin_snapshot,
                self.view,
                &self.preferences,
            );
            if projections.is_empty() {
                query_failed |= jdtls_failed || kotlin_failed;
                import_failed |= kotlin_import_failed;
                query_pending |= jdtls_pending || kotlin_pending;
                if worktree.is_local() {
                    local_without_snapshot += 1;
                } else {
                    remote_without_snapshot += 1;
                }
                continue;
            }
            partial |= omitted_stale;
            import_failed |= kotlin_import_failed;
            let shown_jdtls = projections
                .iter()
                .any(|projection| projection.provider == DependencyProvider::Jdtls);
            let shown_kotlin = projections
                .iter()
                .any(|projection| projection.provider == DependencyProvider::Kotlin);
            partial |= (!shown_jdtls && (jdtls_failed || jdtls_pending))
                || (!shown_kotlin && (kotlin_failed || kotlin_pending));
            for projection in projections {
                let DependencyProjection {
                    provider,
                    server_id,
                    generation,
                    partial: provider_partial,
                    current,
                    mut rows,
                    navigation_targets,
                    graph_selections: provider_graph_selections,
                } = projection;
                if current && worktree.is_local() {
                    graph_selections.extend(provider_graph_selections);
                }
                if provider == DependencyProvider::Jdtls
                    && current
                    && worktree.is_local()
                    && let Some((snapshot, current)) = jdtls_snapshot
                {
                    let (selections, unavailable) = operation_selections_for_snapshot(
                        snapshot,
                        current,
                        self.view,
                        worktree.path_style(),
                    );
                    annotate_operation_rows(&mut rows, &selections, &unavailable);
                    operation_selections.extend(selections);
                }
                match provider {
                    DependencyProvider::Jdtls => {
                        query_failed |= jdtls_failed;
                        query_pending |= jdtls_pending;
                    }
                    DependencyProvider::Kotlin => {
                        query_failed |= kotlin_failed;
                        query_pending |= kotlin_pending;
                    }
                }
                stale |= !current;
                partial |= provider_partial;
                let mut source_directory_statuses = HashMap::new();
                for (id, target) in navigation_targets {
                    let resolved = match target {
                        DependencyNavigationTarget::ModuleManifest { root, build_system } => {
                            visible_module_manifest(&root, build_system, |candidate| {
                                worktree.entry_for_path(candidate).is_some_and(|entry| {
                                    entry.is_file()
                                        && !entry.is_private
                                        && !entry.is_external
                                        && !entry.is_fifo
                                })
                            })
                            .map(|path| (path, DependencyNavigationKind::File))
                        }
                        DependencyNavigationTarget::SourceDirectory { path, navigable } => {
                            let status =
                                source_directory_status(worktree, &path, navigable, current);
                            source_directory_statuses.insert(id.clone(), status);
                            (status == SourceDirectoryStatus::Available)
                                .then_some((path, DependencyNavigationKind::Directory))
                        }
                    };
                    if let Some((path, kind)) = resolved {
                        navigation.insert(
                            id,
                            DependencyNavigation {
                                path,
                                server_id,
                                generation,
                                provider,
                                kind,
                            },
                        );
                    }
                }
                apply_source_directory_status(&mut rows, &source_directory_statuses);
                if provider == DependencyProvider::Jdtls
                    && let Some(inspection) = &self.graph_inspection
                    && let Some((snapshot, current)) = jdtls_snapshot
                    && inspection.selection.matches(snapshot, current)
                    && let Some((graph, row)) =
                        graph_inspection_row(snapshot, self.view, inspection)
                {
                    inspection_current |= append_graph_inspection(&mut rows, &graph, &row);
                }
                if provider == DependencyProvider::Jdtls
                    && let Some(selection) = &self.graph_diagram_selection
                    && let Some((snapshot, current)) = jdtls_snapshot
                    && selection.matches(snapshot, current)
                {
                    diagram_source = Some(dependency_diagram_source(
                        snapshot, current, self.view, selection,
                    ));
                }
                mark_manifest_navigation(&mut rows, &navigation);
                roots.append(&mut rows);
            }
        }
        if !inspection_current {
            self.graph_inspection = None;
        }
        match diagram_source {
            Some(Ok(Some(source))) => {
                if self
                    .graph_diagram
                    .as_ref()
                    .is_none_or(|diagram| diagram.read(cx).source().as_ref() != source.as_str())
                {
                    self.graph_diagram = Some(cx.new(|cx| {
                        markdown::Markdown::new_with_options(
                            source.into(),
                            None,
                            None,
                            markdown::MarkdownOptions {
                                render_mermaid_diagrams: true,
                                ..Default::default()
                            },
                            cx,
                        )
                    }));
                }
            }
            Some(Err(_)) => {
                self.graph_diagram = None;
                self.graph_diagram_selection = None;
                self.graph_notice = Some("Diagram unavailable: the current graph exceeds presentation limits or contains unsupported labels/references. Use tree facts, review provider data and recollect. Limits: 64 nodes, 128 edges, 256 characters per label, 128 KiB source; no facts were truncated.".into());
            }
            _ => {
                self.graph_diagram = None;
                self.graph_diagram_selection = None;
            }
        }
        let search = graph_search_query(&self.graph_search.read(cx).text(cx));
        match &search {
            Ok(Some(query)) => search_project_graph_rows(&mut roots, query),
            Err(_) => suppress_invalid_graph_search(&mut roots),
            Ok(None) => {}
        }
        let mut graph_search_expansion = Vec::new();
        if matches!(&search, Ok(Some(_))) {
            for root in &roots {
                graph_search_expansion_ids(root, &mut graph_search_expansion);
            }
        }
        let restricted = !restricted_worktrees.is_empty();
        let status = if search.is_err() && !roots.is_empty() {
            LanguageToolProviderStatus::Error("Graph search is invalid; use at most 512 UTF-8 bytes without control characters, or clear the search".into())
        } else if restricted && roots.is_empty() {
            LanguageToolProviderStatus::Restricted(
                "Trust this project to load JVM dependency data.".to_string(),
            )
        } else if query_failed {
            LanguageToolProviderStatus::Error(
                "Could not read JVM provider dependencies. View language server logs and refresh."
                    .to_string(),
            )
        } else if jdtls_import_failed {
            LanguageToolProviderStatus::Error(
                "JVM build import failed or cancelled. Review build selection settings and Language Servers, then restart the Java server."
                    .to_string(),
            )
        } else if import_failed {
            LanguageToolProviderStatus::Error(
                "Kotlin project import failed. Check Language Servers for build or offline-cache errors, then refresh."
                    .to_string(),
            )
        } else if query_pending && roots.is_empty() {
            LanguageToolProviderStatus::Loading
        } else if local_worktrees == 0
            && remote_without_snapshot == remote_worktrees
            && remote_worktrees > 0
        {
            LanguageToolProviderStatus::Unsupported(
                "Remote JVM project data is unavailable; check the project host and refresh."
                    .to_string(),
            )
        } else if local_worktrees == 0 && remote_worktrees == 0 {
            LanguageToolProviderStatus::Empty("Open a Maven or Gradle project.".to_string())
        } else if stale {
            LanguageToolProviderStatus::Stale(
                "JVM project data changed; waiting for current provider results.".to_string(),
            )
        } else if remote_without_snapshot > 0 {
            LanguageToolProviderStatus::Partial(
                "Some remote JVM dependency data is unavailable; visible rows may be incomplete."
                    .to_string(),
            )
        } else if local_without_snapshot > 0 && !roots.is_empty() {
            LanguageToolProviderStatus::Partial(
                "Some project folders have no current JVM dependency data; visible rows may be incomplete."
                    .to_string(),
            )
        } else if restricted {
            LanguageToolProviderStatus::Partial(
                "Some project folders are restricted; their JVM dependencies are hidden."
                    .to_string(),
            )
        } else if roots.is_empty() && self.view == BuildToolView::Gradle {
            LanguageToolProviderStatus::Empty(if local_worktrees > 0 {
                "Gradle metadata unavailable. Open Java source, check Language Servers, then use Synchronize metadata; this executes trusted build configuration scripts.".into()
            } else {
                "Gradle metadata unavailable on the project host. Remote metadata synchronization is not yet supported.".into()
            })
        } else if partial {
            LanguageToolProviderStatus::Partial(
                "The JVM provider returned incomplete dependency or imported-project data. Refresh metadata or check Language Servers for setup errors.".to_string(),
            )
        } else if roots.is_empty() {
            LanguageToolProviderStatus::Empty(
                "No JVM provider dependency data yet. Open a Maven or Gradle project, check Language Servers for setup errors, then refresh."
                    .to_string(),
            )
        } else {
            LanguageToolProviderStatus::Current
        };
        self.navigation = navigation;
        self.graph_selections = graph_selections;
        if self.ordered_operations.iter().any(|receipt| {
            !operation_selections
                .values()
                .any(|current| current == receipt)
        }) {
            self.ordered_operations.clear();
            self.operation_notice = Some(
                "Ordered operations were cleared because their project or provider data changed. Refresh and select them again.".into(),
            );
        }
        if !self.ordered_operations.is_empty() {
            roots.push(ordered_operation_tree(&self.ordered_operations));
        }
        self.operation_selections = operation_selections;
        self.host
            .replace_snapshot(LanguageToolSnapshot { roots, status });
        self.host.expand_nodes(graph_search_expansion);
        cx.notify();
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        if self.host_unavailable(cx) {
            self.rebuild(cx);
            return;
        }
        let Some(trusted_worktrees) = TrustedWorktrees::try_get_global(cx) else {
            self.rebuild(cx);
            return;
        };
        self.host.start_refresh();
        let project = self.project.read(cx);
        let worktree_store = project.worktree_store();
        let restricted_worktrees = trusted_worktrees
            .read(cx)
            .restricted_worktrees(&worktree_store, cx)
            .into_iter()
            .map(|(worktree_id, _)| worktree_id)
            .collect::<HashSet<_>>();
        let worktree_ids = worktree_store
            .read(cx)
            .worktrees()
            .filter_map(|worktree| {
                let worktree = worktree.read(cx);
                (worktree.is_visible() && !restricted_worktrees.contains(&worktree.id()))
                    .then_some(worktree.id())
            })
            .collect::<Vec<_>>();
        project.lsp_store().update(cx, |store, cx| {
            for worktree_id in worktree_ids {
                store.refresh_jdtls_project(worktree_id, cx);
                store.refresh_kotlin_workspace(worktree_id, cx);
            }
        });
        self.rebuild(cx);
    }

    fn synchronize_gradle_metadata(&mut self, cx: &mut Context<Self>) {
        let selected_worktree = self
            .host
            .selected()
            .and_then(|id| self.navigation.get(id))
            .map(|navigation| navigation.path.worktree_id);
        let lsp_store = self.project.read(cx).lsp_store();
        if let Err(error) = lsp_store.update(cx, |store, cx| {
            store.synchronize_local_gradle_project(selected_worktree, cx)
        }) {
            if let Err(closed) = self
                .workspace
                .update(cx, |workspace, cx| workspace.show_error(error, cx))
            {
                log::debug!("Groovy workspace closed before synchronization error: {closed:#}");
            }
        }
        self.rebuild(cx);
    }

    fn run_selected_offline_operation(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let result = (|| -> anyhow::Result<()> {
            anyhow::ensure!(
                !self.project.read(cx).is_remote() && !self.host_unavailable(cx),
                "The project host is unavailable; reconnect and refresh before running operations"
            );
            let receipt = self.host.selected().and_then(|id|self.operation_selections.get(id)).cloned()
                .ok_or_else(||anyhow::anyhow!("Select a current local Maven phase, known plugin goal or Gradle task; refresh provider setup if it is unavailable"))?;
            self.workspace
                .update(cx, |workspace, cx| {
                    workspace.run_jvm_operation(receipt, window, cx)
                })
                .map_err(|_| {
                    anyhow::anyhow!(
                        "The workspace is unavailable; reopen the project before running operations"
                    )
                })??;
            Ok(())
        })();
        self.operation_notice=Some(match result {
            Ok(())=>"Build operation sent to Tasks. Follow terminal output and use terminal controls to cancel.".into(),
            Err(error)=>error.to_string(),
        });
        cx.notify();
    }

    fn add_selected_ordered_operation(&mut self, cx: &mut Context<Self>) {
        let result = (|| -> anyhow::Result<()> {
            anyhow::ensure!(
                !self.project.read(cx).is_remote() && !self.host_unavailable(cx),
                "Ordered execution requires a connected local project host"
            );
            let receipt = self
                .host
                .selected()
                .and_then(|id| self.operation_selections.get(id))
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Select a current Maven phase, known plugin goal or Gradle task to add"
                    )
                })?;
            let mut proposed = self.ordered_operations.clone();
            proposed.push(receipt);
            compose_ordered_operations(&proposed, &self.operation_selections)?;
            self.ordered_operations = proposed;
            Ok(())
        })();
        let error = result.err();
        if error.is_none() {
            self.operation_notice =
                Some("Operation added. Inspect the ordered list before running it.".into());
        }
        self.rebuild(cx);
        if let Some(error) = error {
            self.operation_notice = Some(error.to_string());
        }
        self.host.expand_nodes([ordered_operation_root_id()]);
        cx.notify();
    }

    fn remove_last_ordered_operation(&mut self, cx: &mut Context<Self>) {
        self.ordered_operations.pop();
        self.operation_notice =
            Some("Last ordered operation removed; no execution was started.".into());
        self.rebuild(cx);
    }

    fn clear_ordered_operations(&mut self, cx: &mut Context<Self>) {
        self.ordered_operations.clear();
        self.operation_notice =
            Some("Ordered selection cleared; running Tasks remain under terminal controls.".into());
        self.rebuild(cx);
    }

    fn run_ordered_offline_operations(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let result = (|| -> anyhow::Result<()> {
            anyhow::ensure!(
                !self.project.read(cx).is_remote() && !self.host_unavailable(cx),
                "Ordered execution requires a connected local project host"
            );
            let receipt =
                compose_ordered_operations(&self.ordered_operations, &self.operation_selections)?;
            self.workspace
                .update(cx, |workspace, cx| {
                    workspace.run_jvm_operation(receipt, window, cx)
                })
                .map_err(|_| {
                    anyhow::anyhow!(
                        "The workspace is unavailable; reopen the project before running operations"
                    )
                })??;
            Ok(())
        })();
        self.operation_notice = Some(match result {
            Ok(()) => "Ordered operations sent to Tasks. Follow terminal output and use terminal controls to cancel.".into(),
            Err(error) => error.to_string(),
        });
        cx.notify();
    }

    fn selected_build_worktree(&self, cx: &App) -> anyhow::Result<settings::WorktreeId> {
        anyhow::ensure!(
            !self.project.read(cx).is_remote() && !self.host_unavailable(cx),
            "Build selection requires a connected local project host"
        );
        let selected = self
            .host
            .selected()
            .and_then(|id| self.navigation.get(id))
            .map(|navigation| navigation.path.worktree_id);
        let worktrees = self.project.read(cx).worktree_store();
        let candidates = worktrees
            .read(cx)
            .worktrees()
            .filter_map(|worktree| {
                let worktree = worktree.read(cx);
                (worktree.is_local()
                    && worktree.is_visible()
                    && selected.is_none_or(|selected| selected == worktree.id()))
                .then_some(worktree.id())
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(
            candidates.len() == 1,
            "Select a project folder before applying saved build selection; open a Java source and check Language Servers when its provider is missing"
        );
        candidates
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("The selected local project folder is unavailable"))
    }

    fn build_phase_is_pending(&self, cx: &App) -> bool {
        self.build_phase_token.is_some_and(|token| {
            self.project
                .read(cx)
                .lsp_store()
                .read(cx)
                .jdtls_project_import_token(token.worktree_id())
                == Some(token)
        })
    }

    fn show_build_phase_error(&mut self, cx: &mut Context<Self>) {
        let message = "Could not apply JVM build selection. Trust the project, review jvm_builds settings and Language Servers, then refresh and retry. Remote import is not supported by these controls.";
        self.build_phase_notice = Some(message.into());
        if let Err(error) = self.workspace.update(cx, |workspace, cx| {
            workspace.show_error(anyhow::anyhow!(message), cx)
        }) {
            log::debug!("Groovy workspace closed before build-selection status: {error:#}");
        }
        cx.notify();
    }

    fn apply_saved_build_selection(&mut self, maven: bool, cx: &mut Context<Self>) {
        if self.build_phase_token.is_some() {
            self.build_phase_notice = Some("This panel is waiting for its selected import. Cancel that import or wait before applying another selection.".into());
            cx.notify();
            return;
        }
        let worktree_id = match self.selected_build_worktree(cx) {
            Ok(worktree_id) => worktree_id,
            Err(_) => {
                self.show_build_phase_error(cx);
                return;
            }
        };
        let result = self.project.read(cx).lsp_store().update(cx, |store, cx| {
            let waiter = if maven { store.apply_saved_local_maven_selection(worktree_id, cx)? } else { store.import_selected_local_gradle_project(worktree_id, cx)? };
            let token = store.jdtls_project_import_token(worktree_id).ok_or_else(|| anyhow::anyhow!("Selected import lost its phase receipt; refresh and restart its provider before retrying"))?;
            anyhow::Ok((token, waiter))
        });
        match result {
            Ok((token, waiter)) => {
                self.build_phase_token = Some(token);
                self.build_phase_notice = Some(if maven { "Applying saved Maven roots and exclusions; full synchronization remains unverified." } else { "Importing the effective Gradle selection; full synchronization remains unverified." }.into());
                self.build_phase_waiter = Some(cx.spawn(async move |panel, cx| {
                    let result = waiter.await;
                    if let Err(error) = panel.update(cx, |panel, cx| {
                        if panel.build_phase_token != Some(token) { return; }
                        panel.build_phase_token = None;
                        if result.is_ok() {
                            panel.build_phase_notice = Some("Selected build phase completed. Current provider data can be refreshed; this does not certify full synchronization.".into());
                        } else {
                            panel.show_build_phase_error(cx);
                        }
                        panel.rebuild(cx);
                    }) { log::debug!("Groovy panel closed before selected build-phase completion: {error:#}"); }
                }));
            }
            Err(_) => self.show_build_phase_error(cx),
        }
        self.rebuild(cx);
    }

    fn cancel_selected_import(&mut self, cx: &mut Context<Self>) {
        let Some(token) = self.build_phase_token else {
            return;
        };
        let result = self.project.read(cx).lsp_store().update(cx, |store, cx| {
            store.cancel_selected_project_phase_for_token(token, cx)
        });
        match result {
            Ok(cancelled) => {
                self.build_phase_token = None;
                drop(self.build_phase_waiter.take());
                self.build_phase_notice = Some(if cancelled { "Selected import cancelled. Refresh provider data and review build selection before retrying." } else { "The selected import already completed or changed; no other phase was cancelled. Refresh before retrying." }.into());
            }
            Err(_) => self.show_build_phase_error(cx),
        }
        self.rebuild(cx);
    }

    fn selected_graph(&self) -> Option<GraphSelection> {
        self.host
            .selected()
            .and_then(|id| self.graph_selections.get(id))
            .cloned()
    }

    fn graph_query_is_pending(&self, cx: &App) -> bool {
        self.graph_query_token.is_some_and(|token| {
            self.project
                .read(cx)
                .lsp_store()
                .read(cx)
                .jdtls_project_query_token(token.worktree_id())
                == Some(token)
        })
    }

    fn show_graph_error(&mut self, error: anyhow::Error, cx: &mut Context<Self>) {
        self.graph_notice = Some(error.to_string());
        cx.notify();
        if let Err(closed) = self
            .workspace
            .update(cx, |workspace, cx| workspace.show_error(error, cx))
        {
            log::debug!("Groovy workspace closed before graph error: {closed:#}");
        }
    }

    fn collect_selected_graph(&mut self, cx: &mut Context<Self>) {
        if self.graph_query_token.is_some() {
            return;
        }
        if self.project.read(cx).is_remote() {
            self.show_graph_error(
                anyhow::anyhow!(
                    "Remote graph collection is unavailable; inspect or refresh existing host data"
                ),
                cx,
            );
            return;
        }
        let Some(selection) = self.selected_graph() else {
            self.show_graph_error(anyhow::anyhow!("Select a current Maven project or a resolvable configuration in the Gradle view, then collect its offline graph"), cx);
            return;
        };
        let lsp_store = self.project.read(cx).lsp_store();
        let result = lsp_store.update(cx, |store, cx| -> anyhow::Result<_> {
            anyhow::ensure!(store.jdtls_project_snapshot(selection.worktree_id()).is_some_and(|(snapshot, current)| selection.matches(snapshot, current)),
                "Graph selection changed; refresh and select the current project or configuration before retrying");
            let query = match &selection.target {
                GraphQueryTarget::Maven(pom) => store.query_local_maven_dependency_graph(pom.worktree_id, pom.clone(), cx)?,
                GraphQueryTarget::Gradle { module, configuration } => store.query_local_gradle_dependency_graph(module.worktree_id, module.clone(), configuration.clone(), cx)?,
            };
            let Some(token) = store.jdtls_project_query_token(selection.worktree_id()) else {
                store.cancel_local_jdtls_project_query(selection.worktree_id(), cx)?;
                anyhow::bail!("Graph query lost its host binding; refresh before retrying");
            };
            Ok((token, query))
        });
        match result {
            Ok((token, query)) => {
                self.graph_notice = None;
                self.graph_query_token = Some(token);
                self.graph_query = Some(cx.spawn(async move |panel, cx| {
                    let result = query.await;
                    if let Err(closed) = panel.update(cx, |panel, cx| {
                        if panel.graph_query_token != Some(token) {
                            return;
                        }
                        panel.graph_query_token = None;
                        if let Err(error) = result {
                            panel.show_graph_error(error, cx);
                        }
                        panel.rebuild(cx);
                    }) {
                        log::debug!("Groovy panel closed before graph result: {closed:#}");
                    }
                }));
            }
            Err(error) => self.show_graph_error(error, cx),
        }
        self.rebuild(cx);
    }

    fn cancel_graph_collection(&mut self, cx: &mut Context<Self>) {
        let Some(token) = self.graph_query_token else {
            return;
        };
        let result = self.project.read(cx).lsp_store().update(cx, |store, cx| {
            store.cancel_local_jdtls_project_query_for_token(token, cx)
        });
        match result {
            Ok(cancelled) => {
                if cancelled {
                    self.graph_notice = Some("Graph collection cancelled; refresh current project data before collecting again".into());
                } else {
                    log::debug!(
                        "Groovy graph query already completed or changed before cancellation"
                    );
                }
                self.graph_query_token = None;
                drop(self.graph_query.take());
            }
            Err(error) => self.show_graph_error(error, cx),
        }
        self.rebuild(cx);
    }

    fn activate_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.host.can_activate() {
            return;
        }
        let Some(navigation) = self.host.selected().and_then(|id| self.navigation.get(id)) else {
            return;
        };
        let path = navigation.path.clone();
        let kind = navigation.kind;
        if self.host_unavailable(cx) {
            self.rebuild(cx);
            return;
        }
        let Some(trusted_worktrees) = TrustedWorktrees::try_get_global(cx) else {
            self.rebuild(cx);
            return;
        };
        let project = self.project.read(cx);
        let worktree_store = project.worktree_store();
        if trusted_worktrees
            .read(cx)
            .restricted_worktrees(&worktree_store, cx)
            .into_iter()
            .any(|(worktree_id, _)| worktree_id == path.worktree_id)
        {
            self.rebuild(cx);
            return;
        }
        let lsp_store = project.lsp_store();
        let lsp_store = lsp_store.read(cx);
        let current = match navigation.provider {
            DependencyProvider::Jdtls => lsp_store
                .jdtls_project_snapshot(path.worktree_id)
                .is_some_and(|(snapshot, current)| {
                    navigation.matches(
                        snapshot.server_id,
                        snapshot.worktree_id,
                        snapshot.generation,
                        current,
                    )
                }),
            DependencyProvider::Kotlin => lsp_store
                .kotlin_workspace_snapshot(path.worktree_id)
                .is_some_and(|(snapshot, current)| {
                    navigation.matches(
                        snapshot.server_id,
                        snapshot.worktree_id,
                        snapshot.generation,
                        current,
                    )
                }),
        };
        let entry_id = project
            .worktree_store()
            .read(cx)
            .worktree_for_id(path.worktree_id, cx)
            .and_then(|worktree| {
                let worktree = worktree.read(cx);
                if kind == DependencyNavigationKind::Directory
                    && source_directory_status(worktree, &path, true, current)
                        != SourceDirectoryStatus::Available
                {
                    return None;
                }
                worktree
                    .entry_for_path(path.path.as_ref())
                    .filter(|entry| {
                        !entry.is_private
                            && !entry.is_external
                            && !entry.is_fifo
                            && match kind {
                                DependencyNavigationKind::File => entry.is_file(),
                                DependencyNavigationKind::Directory => entry.is_dir(),
                            }
                    })
                    .map(|entry| entry.id)
            });
        let Some(entry_id) = entry_id.filter(|_| current) else {
            self.rebuild(cx);
            return;
        };
        if kind == DependencyNavigationKind::Directory {
            self.project.update(cx, |_, cx| {
                cx.emit(project::Event::RevealInProjectPanel(entry_id));
            });
            return;
        }
        if let Some(workspace) = self.workspace.upgrade() {
            workspace
                .update(cx, |workspace, cx| {
                    workspace.open_path(path, None, true, window, cx)
                })
                .detach_and_log_err(cx);
        }
    }
}

fn gradle_option_node(
    option_id: LanguageToolNodeId,
    option: &project::jvm_project::GradleTaskOption,
) -> LanguageToolNode {
    let coverage = match &option.enum_constants {
        None => "enum declarations unknown".to_string(),
        Some(constants) if constants.is_empty() => "no declared enum members".to_string(),
        Some(constants) => format!("{} declared enum members", constants.len()),
    };
    let children = option.enum_constants.iter().flatten().map(|constant| LanguageToolNode {
        id: LanguageToolNodeId(format!("{}:enum-member:{constant}", option_id.0)),
        label: constant.clone(),
        secondary_label: Some("Declared enum member; acceptance unknown".to_string()),
        accessibility_label: format!(
            "Declared enum member {constant} for option {}; custom setter acceptance unknown; not an execution argument",
            option.name
        ),
        icon: None,
        children: Vec::new(),
        enabled: false,
        activation_label: None,
    }).collect();
    LanguageToolNode {
        id: option_id,
        label: format!("--{}", option.name),
        secondary_label: Some(format!("{}; {coverage}", option.description)),
        accessibility_label: format!(
            "Declared option {}; {}; Java declaration {}; {coverage}; accepted values and requiredness unknown",
            option.name, option.description, option.declared_type
        ),
        icon: None,
        children,
        enabled: false,
        activation_label: None,
    }
}

fn project_rows_for_view(
    snapshot: &JdtlsProjectSnapshot,
    view: BuildToolView,
    preferences: &BuildViewPreferences,
) -> (
    Vec<LanguageToolNode>,
    HashMap<LanguageToolNodeId, DependencyNavigationTarget>,
) {
    let mut roots = Vec::new();
    let mut navigation = HashMap::new();
    let mut module_groups = HashMap::new();
    for module in &snapshot.projects {
        if (view == BuildToolView::Gradle && module.gradle.is_none())
            || (view == BuildToolView::Maven && module.maven.is_none())
        {
            continue;
        }
        let module_name = if module.module_root.path.is_empty() {
            "Project root".to_string()
        } else {
            module.module_root.path.as_unix_str().to_string()
        };
        let module_id = jdtls_module_node_id(snapshot, &module.module_root, view);
        let mut children = Vec::new();
        let membership = match module.membership {
            JdtlsProjectMembership::Unknown => "imported membership unknown",
            JdtlsProjectMembership::MetadataOnly => "provider metadata only; import not observed",
            JdtlsProjectMembership::ImportedJava => "imported Java project",
            JdtlsProjectMembership::ImportedNonJava => {
                "imported non-Java project; Java semantics unavailable"
            }
        };
        for reference in &module.project_references {
            children.push(LanguageToolNode {
                id: LanguageToolNodeId(format!("{}:project:{reference}", module_id.0)),
                label: reference.clone(),
                secondary_label: Some("Project reference; scope unknown".to_string()),
                icon: None,
                accessibility_label: format!(
                    "Project reference {reference}; dependency scope and version unknown"
                ),
                children: Vec::new(),
                enabled: false,
                activation_label: None,
            });
        }
        if module.library_entries > 0 {
            let artifact_children = module
                .library_artifacts
                .iter()
                .enumerate()
                .map(|(index, artifact)| {
                    let scope = artifact.scope.as_deref().unwrap_or("unknown");
                    LanguageToolNode {
                        id: LanguageToolNodeId(format!(
                            "{}:maven-artifact:{index}",
                            module_id.0
                        )),
                        label: artifact.coordinate.clone(),
                        secondary_label: Some(format!(
                            "Provider Maven scope {scope}; directness unknown"
                        )),
                        icon: None,
                        accessibility_label: format!(
                            "JDTLS Maven artifact {}; scope {scope}; direct or transitive relationship unknown",
                            artifact.coordinate
                        ),
                        children: Vec::new(),
                        enabled: false,
                        activation_label: None,
                    }
                })
                .collect::<Vec<_>>();
            let library_files = module
                .library_files
                .iter()
                .enumerate()
                .map(|(index, file)| LanguageToolNode {
                    id: LanguageToolNodeId(format!(
                        "{}:library:{index}:{}",
                        module_id.0,
                        file.name
                    )),
                    label: file.name.clone(),
                    secondary_label: Some(if file.gradle_source_sets.is_empty() {
                        "Provider classpath file; coordinates and scope unknown".to_string()
                    } else {
                        format!(
                            "Gradle used by source sets {}; coordinates unknown",
                            file.gradle_source_sets.join(", ")
                        )
                    }),
                    icon: None,
                    accessibility_label: if file.gradle_source_sets.is_empty() {
                        format!(
                            "Provider classpath library file {}; direct or transitive relationship, coordinates and scope unknown",
                            file.name
                        )
                    } else {
                        format!(
                            "Provider Gradle classpath library file {}; used by source sets {}; direct or transitive relationship and coordinates unknown",
                            file.name,
                            file.gradle_source_sets.join(", ")
                        )
                    },
                    children: Vec::new(),
                    enabled: false,
                    activation_label: None,
                })
                .collect::<Vec<_>>();
            let unnamed = module
                .library_entries
                .saturating_sub(artifact_children.len() + library_files.len());
            let mut library_children = artifact_children;
            library_children.extend(library_files);
            if unnamed > 0 {
                let noun = if unnamed == 1 { "library" } else { "libraries" };
                library_children.push(LanguageToolNode {
                    id: LanguageToolNodeId(format!("{}:unnamed-libraries", module_id.0)),
                    label: format!("{unnamed} unnamed {noun}"),
                    secondary_label: Some("Provider did not supply safe file names".to_string()),
                    icon: None,
                    accessibility_label: format!(
                        "{unnamed} provider-reported external {noun} without safe coordinates or file names"
                    ),
                    children: Vec::new(),
                    enabled: false,
                    activation_label: None,
                });
            }
            let has_maven_artifacts = !module.library_artifacts.is_empty();
            let has_gradle_source_sets = module
                .library_files
                .iter()
                .any(|file| !file.gradle_source_sets.is_empty());
            let available_facts = match (has_maven_artifacts, has_gradle_source_sets) {
                (true, true) => {
                    "Maven coordinates and scopes plus Gradle source-set use where available"
                }
                (true, false) => "Maven coordinates and scopes where available",
                (false, true) => "Gradle source-set use where available; coordinates unknown",
                (false, false) => "coordinates, scopes, and source-set use unknown",
            };
            children.push(LanguageToolNode {
                id: LanguageToolNodeId(format!("{}:libraries", module_id.0)),
                label: format!("{} external libraries", module.library_entries),
                secondary_label: Some(
                    "Provider coordinates and scopes where available; directness unknown"
                        .to_string(),
                ),
                icon: None,
                accessibility_label: format!(
                    "{} provider-reported external libraries; {available_facts}; direct or transitive relationship unknown",
                    module.library_entries,
                ),
                children: library_children,
                enabled: false,
                activation_label: None,
            });
        }
        if let Some(gradle) = &module.gradle
            && view != BuildToolView::Maven
        {
            children.push(LanguageToolNode {
                id: LanguageToolNodeId(format!("{}:gradle-tasks", module_id.0)),
                label: "Gradle tasks".to_string(),
                secondary_label: Some(
                    "Discovered tasks and declared options; execution pending".to_string(),
                ),
                accessibility_label:
                    "Gradle provider tasks; declared options where available; task execution unavailable"
                        .to_string(),
                icon: None,
                children: gradle
                    .tasks
                    .iter()
                    .map(|task| LanguageToolNode {
                        id: gradle_task_node_id(&module_id,gradle,&task.path),
                        label: task.path.clone(),
                        secondary_label: task.group.clone(),
                        accessibility_label: format!(
                            "Gradle task {}; group {}; {}; option declarations {}; accepted values unknown; execution unavailable",
                            task.path,
                            task.group.as_deref().unwrap_or("ungrouped"),
                            task.description.as_deref().unwrap_or("no description"),
                            if task.options_known { "available" } else { "incomplete" }
                        ),
                        icon: None,
                        children: task.options.iter().map(|option| {
                            let option_id = LanguageToolNodeId(format!(
                                "{}:gradle-option:{}:{}:{}:{}",
                                module_id.0, gradle.build_root.path.as_unix_str(),
                                gradle.project_path, task.path, option.name
                            ));
                            gradle_option_node(option_id, option)
                        }).collect(),
                        enabled: false,
                        activation_label: None,
                    })
                    .collect(),
                enabled: false,
                activation_label: None,
            });
            children.push(LanguageToolNode {
                id: LanguageToolNodeId(format!("{}:gradle-source-sets", module_id.0)),
                label: "Gradle source sets".to_string(),
                secondary_label: Some(if gradle.source_sets_known {
                    "Declared roots; generatedness unknown".to_string()
                } else {
                    "Java source-set model unavailable".to_string()
                }),
                accessibility_label: format!(
                    "Gradle source sets; {}; generatedness unknown",
                    if gradle.source_sets_known {
                        "declared roots available"
                    } else {
                        "model unavailable"
                    }
                ),
                icon: None,
                children: gradle
                    .source_sets
                    .iter()
                    .map(|source| {
                        let mut source_children = source
                            .roots
                            .iter()
                            .map(|root| LanguageToolNode {
                                id: LanguageToolNodeId(format!(
                                    "{}:gradle-source-root:{}:{}",
                                    module_id.0,
                                    source.name,
                                    root.path.path.as_unix_str()
                                )),
                                label: root.path.path.as_unix_str().to_string(),
                                secondary_label: Some(
                                    "Declared source root; generatedness unknown".to_string(),
                                ),
                                accessibility_label: format!(
                                    "Declared Gradle source root {}; generatedness unknown; {}",
                                    root.path.path.as_unix_str(),
                                    if root.navigable {
                                        "present"
                                    } else {
                                        "not present in project snapshot"
                                    }
                                ),
                                icon: None,
                                children: Vec::new(),
                                enabled: false,
                                activation_label: None,
                            })
                            .collect::<Vec<_>>();
                        let output_id = LanguageToolNodeId(format!(
                            "{}:gradle-source-set:{}:annotation-processor-output", module_id.0, source.name
                        ));
                        let (label, description) = match &source.annotation_processor_output {
                            GradleAnnotationProcessorOutput::Unknown => (
                                "Annotation processor output".to_string(), "Output details unavailable"
                            ),
                            GradleAnnotationProcessorOutput::Absent => (
                                "Annotation processor output".to_string(), "No output directory configured"
                            ),
                            GradleAnnotationProcessorOutput::Declared(output) => {
                                navigation.insert(output_id.clone(), DependencyNavigationTarget::SourceDirectory {
                                    path: output.path.clone(), navigable: output.navigable,
                                });
                                (output.path.path.as_unix_str().to_string(), "Declared output; generation unverified")
                            }
                        };
                        source_children.push(LanguageToolNode {
                            id: output_id, label: label.clone(), secondary_label: Some(description.into()),
                            accessibility_label: format!(
                                "Gradle annotation-processor output {label}; {description}; other generated-source coverage unknown"
                            ), icon: None, children: Vec::new(), enabled: false, activation_label: None,
                        });
                        LanguageToolNode {
                            id: LanguageToolNodeId(format!("{}:gradle-source-set:{}", module_id.0, source.name)),
                            label: source.name.clone(),
                            secondary_label: Some(format!("{} declared roots", source.roots.len())),
                            accessibility_label: format!(
                                "Gradle source set {}; {} declared roots; generatedness unknown",
                                source.name, source.roots.len()
                            ), icon: None, children: source_children, enabled: false, activation_label: None,
                        }
                    })
                    .collect(),
                enabled: false,
                activation_label: None,
            });
            children.push(gradle_configurations_node(&module_id, gradle));
        }
        if view == BuildToolView::Maven
            && let Some(maven) = &module.maven
        {
            children.extend(maven_metadata_rows(
                &module_id,
                maven,
                preferences.show_basic_maven_phases,
            ));
            children.push(LanguageToolNode {
                id: LanguageToolNodeId(format!("{}:maven-source-roots", module_id.0)),
                label: "Source roots".into(),
                secondary_label: Some("JDTLS imported roots; generatedness unknown".into()),
                accessibility_label: "Maven imported source roots; generatedness unknown".into(),
                icon: None,
                children: module
                    .source_roots
                    .iter()
                    .map(|root| LanguageToolNode {
                        id: LanguageToolNodeId(format!(
                            "{}:maven-source-root:{}",
                            module_id.0,
                            root.path.path.as_unix_str()
                        )),
                        label: root.path.path.as_unix_str().into(),
                        secondary_label: Some("Imported source root; generatedness unknown".into()),
                        accessibility_label: format!(
                            "Imported Maven source root {}; generatedness unknown; {}",
                            root.path.path.as_unix_str(),
                            if root.navigable {
                                "present"
                            } else {
                                "not present in current project snapshot"
                            }
                        ),
                        icon: None,
                        children: Vec::new(),
                        enabled: false,
                        activation_label: None,
                    })
                    .collect(),
                enabled: false,
                activation_label: None,
            });
        }
        if view != BuildToolView::Maven
            && let Some(gradle) = &module.gradle
        {
            children.push(gradle_graph_node_filtered(
                &module_id,
                gradle.dependency_graph.as_deref(),
                preferences.graph_fact_filter,
            ));
        }
        if view != BuildToolView::Gradle
            && let Some(maven) = &module.maven
        {
            children.push(maven_graph_node_filtered(
                &module_id,
                maven.dependency_graph.as_deref(),
                preferences.graph_fact_filter,
                preferences.maven_graph_scope,
            ));
        }
        if view == BuildToolView::Gradle {
            children.push(LanguageToolNode {
                id: LanguageToolNodeId(format!("{}:gradle-saved-configurations", module_id.0)),
                label: "Saved configurations".into(),
                secondary_label: Some("Not available in this view; use Tasks for configured commands".into()),
                accessibility_label: "Saved configurations; not available in this view; use Tasks for configured commands".into(),
                icon: None,
                children: Vec::new(),
                enabled: false,
                activation_label: None,
            });
        }
        if view == BuildToolView::Gradle && preferences.group_gradle_tasks {
            if let Some(tasks) = children
                .iter_mut()
                .find(|node| node.id.0.ends_with(":gradle-tasks"))
            {
                let mut groups = BTreeMap::<Option<String>, Vec<LanguageToolNode>>::new();
                for task in std::mem::take(&mut tasks.children) {
                    groups
                        .entry(task.secondary_label.clone())
                        .or_default()
                        .push(task);
                }
                tasks.children = groups
                    .into_iter()
                    .map(|(group, children)| {
                        let identity = group
                            .as_ref()
                            .map(|group| format!("some:{}:{group}", group.len()))
                            .unwrap_or_else(|| "none".into());
                        let label = group.unwrap_or_else(|| "Ungrouped".into());
                        LanguageToolNode {
                            id: LanguageToolNodeId(format!("{}:group:{identity}", tasks.id.0)),
                            accessibility_label: format!(
                                "Gradle task group {label}; {} tasks; execution unavailable",
                                children.len()
                            ),
                            label,
                            secondary_label: None,
                            icon: None,
                            children,
                            enabled: false,
                            activation_label: None,
                        }
                    })
                    .collect();
            }
        }
        navigation.insert(
            module_id.clone(),
            DependencyNavigationTarget::ModuleManifest {
                root: module.module_root.clone(),
                build_system: match view {
                    BuildToolView::Gradle => Some(KotlinBuildSystem::Gradle),
                    BuildToolView::Maven => Some(KotlinBuildSystem::Maven),
                    BuildToolView::Dependencies => {
                        module.gradle.as_ref().map(|_| KotlinBuildSystem::Gradle)
                    }
                },
            },
        );
        if preferences.group_modules(view) {
            let group = match view {
                BuildToolView::Gradle => module.gradle.as_ref().map(|gradle| {
                    let parent = gradle
                        .project_path
                        .rsplit_once(':')
                        .map(|(parent, _)| parent)
                        .filter(|parent| !parent.is_empty())
                        .unwrap_or(":");
                    let build = gradle.build_root.path.as_unix_str();
                    ModuleGroup {
                        id: LanguageToolNodeId(format!(
                            "jdtls-gradle-module-group:{}:{}:{build}:{}:{parent}",
                            snapshot.worktree_id,
                            build.len(),
                            parent.len()
                        )),
                        label: format!(
                            "{} · {parent}",
                            if build.is_empty() {
                                "Project root"
                            } else {
                                build
                            }
                        ),
                        description: "Gradle build and logical parent-project path group".into(),
                    }
                }),
                BuildToolView::Maven => Some(directory_module_group(
                    "jdtls",
                    view,
                    snapshot.worktree_id,
                    module
                        .module_root
                        .path
                        .parent()
                        .map(|parent| parent.as_unix_str())
                        .unwrap_or_default(),
                    "Maven module directory",
                )),
                BuildToolView::Dependencies => None,
            };
            if let Some(group) = group {
                module_groups.insert(module_id.clone(), group);
            }
        }
        let source_level = module
            .java_source_level
            .map(|level| format!("; Java source level {level}"))
            .unwrap_or_default();
        let target_level = match module.java_target_level {
            Some(level) => format!("; Java target level {level} reported by JDTLS"),
            None => "; Java target level not reported".to_string(),
        };
        let project_vm = match module.project_vm_java_major {
            Some(major) => format!("; project VM Java {major} reported by JDTLS"),
            None if module.project_vm_known => "; project VM reported, version unknown".into(),
            None => "; project VM not reported".into(),
        };
        let launch_java = match module.server_launch_java_major {
            Some(major) => format!("; JDTLS launch Java {major}"),
            None => "; JDTLS launch Java unknown".into(),
        };
        roots.push(LanguageToolNode {
            id: module_id,
            label: match view {
                BuildToolView::Maven => module.maven.as_ref().map(|maven| {
                    let name = if preferences.show_maven_artifact_id { maven.artifact_id.clone() } else { module_name.clone() };
                    let name = if preferences.show_maven_version { format!("{name}:{}", maven.version) } else { name };
                    if preferences.show_maven_artifact_id { format!("{name} ({module_name})") } else { name }
                }).unwrap_or_else(|| module_name.clone()),
                BuildToolView::Gradle => module.gradle.as_ref().map(|gradle| format!("{} ({})", gradle.project_path, if gradle.build_root.path.is_empty() { "Project root" } else { gradle.build_root.path.as_unix_str() })).unwrap_or_else(|| module_name.clone()),
                BuildToolView::Dependencies => module_name.clone(),
            },
            secondary_label: Some(format!(
                "JDTLS module; {membership}{source_level}{target_level}{project_vm}{launch_java}; build JDK and test runtime unknown"
            )),
            icon: None,
            accessibility_label: format!(
                "JDTLS module {module_name}; {membership}{source_level}{target_level}{project_vm}{launch_java}; build JDK and test runtime unknown"
            ),
            children,
            enabled: true,
            activation_label: Some("Open module manifest".to_string()),
        });
    }
    if preferences.group_modules(view) {
        roots = group_module_rows(roots, module_groups);
    }
    (roots, navigation)
}

struct ModuleGroup {
    id: LanguageToolNodeId,
    label: String,
    description: String,
}

fn directory_module_group(
    provider: &str,
    view: BuildToolView,
    worktree_id: settings::WorktreeId,
    directory: &str,
    description: &str,
) -> ModuleGroup {
    ModuleGroup {
        id: LanguageToolNodeId(format!(
            "{provider}-directory-group:{}:{worktree_id}:{}:{directory}",
            view.label(),
            directory.len()
        )),
        label: format!(
            "Directory: {}",
            if directory.is_empty() {
                "Project root"
            } else {
                directory
            }
        ),
        description: description.into(),
    }
}

fn group_module_rows(
    rows: Vec<LanguageToolNode>,
    mut groups: HashMap<LanguageToolNodeId, ModuleGroup>,
) -> Vec<LanguageToolNode> {
    let mut roots = Vec::new();
    let mut grouped = BTreeMap::<String, (ModuleGroup, Vec<LanguageToolNode>)>::new();
    for row in rows {
        if let Some(group) = groups.remove(&row.id) {
            grouped
                .entry(group.id.0.clone())
                .or_insert_with(|| (group, Vec::new()))
                .1
                .push(row);
        } else {
            roots.push(row);
        }
    }
    roots.extend(
        grouped
            .into_values()
            .map(|(group, children)| LanguageToolNode {
                id: group.id,
                accessibility_label: format!(
                    "{}; {}; {} modules",
                    group.label,
                    group.description,
                    children.len()
                ),
                label: group.label,
                secondary_label: Some(group.description),
                icon: None,
                children,
                enabled: false,
                activation_label: None,
            }),
    );
    roots
}

fn graph_search_query(text: &str) -> anyhow::Result<Option<String>> {
    anyhow::ensure!(
        text.len() <= MAX_GRAPH_SEARCH_BYTES && !text.chars().any(char::is_control),
        "Invalid graph search; clear it or enter a shorter literal query without control characters"
    );
    let query = text.trim();
    Ok((!query.is_empty()).then(|| query.to_lowercase()))
}

fn graph_row_matches_search(row: &LanguageToolNode, query: &str) -> bool {
    row.label.to_lowercase().contains(query)
        || row
            .secondary_label
            .as_ref()
            .is_some_and(|detail| detail.to_lowercase().contains(query))
}

fn retain_graph_search_matches(row: &mut LanguageToolNode, query: &str) -> bool {
    if graph_row_matches_search(row, query) {
        return true;
    }
    row.children
        .retain_mut(|child| retain_graph_search_matches(child, query));
    !row.children.is_empty()
}

fn search_project_graph_rows(rows: &mut [LanguageToolNode], query: &str) {
    for row in rows {
        if row.id.0.starts_with("graph:") {
            let matched = retain_graph_search_matches(row, query);
            let status = if matched {
                "Matching admitted graph facts"
            } else {
                "No matching admitted graph facts; clear search to inspect all facts"
            };
            if let Some(detail) = &mut row.secondary_label {
                detail.push_str("; ");
                detail.push_str(status);
            }
            row.accessibility_label.push_str("; ");
            row.accessibility_label.push_str(status);
        } else {
            search_project_graph_rows(&mut row.children, query);
        }
    }
}

fn graph_search_expansion_ids(row: &LanguageToolNode, ids: &mut Vec<LanguageToolNodeId>) -> bool {
    let mut contains_graph = row.id.0.starts_with("graph:");
    for child in &row.children {
        contains_graph |= graph_search_expansion_ids(child, ids);
    }
    if contains_graph && !row.children.is_empty() {
        ids.push(row.id.clone());
    }
    contains_graph
}

fn suppress_invalid_graph_search(rows: &mut [LanguageToolNode]) {
    for row in rows {
        if row.id.0.starts_with("graph:") {
            row.children.clear();
            row.accessibility_label
                .push_str("; Invalid graph search; clear it to inspect admitted facts");
        } else {
            suppress_invalid_graph_search(&mut row.children);
        }
    }
}

fn graph_fact_id(module: &LanguageToolNodeId, kind: &str, parts: &[&str]) -> LanguageToolNodeId {
    LanguageToolNodeId(format!(
        "graph:{}:{}:{kind}:{}",
        module.0.len(),
        module.0,
        parts
            .iter()
            .map(|part| format!("{}:{part}", part.len()))
            .collect::<Vec<_>>()
            .join(":")
    ))
}

fn graph_fact_row(
    id: LanguageToolNodeId,
    label: String,
    detail: String,
    mut children: Vec<LanguageToolNode>,
) -> LanguageToolNode {
    children.sort_by(|left, right| {
        (&left.id.0, &left.label, &left.secondary_label).cmp(&(
            &right.id.0,
            &right.label,
            &right.secondary_label,
        ))
    });
    let mut occurrences = HashMap::<String, usize>::new();
    for child in &mut children {
        let occurrence = occurrences.entry(child.id.0.clone()).or_default();
        child.id.0 = format!("{}:occurrence:{occurrence}", child.id.0);
        *occurrence += 1;
    }
    LanguageToolNode {
        id,
        accessibility_label: format!("{label}; {detail}"),
        label,
        secondary_label: Some(detail),
        icon: None,
        children,
        enabled: false,
        activation_label: None,
    }
}

fn gradle_component_label(component: &project::jvm_project::GradleDependencyComponent) -> String {
    use project::jvm_project::GradleDependencyKind;
    match component.kind {
        GradleDependencyKind::Module => format!(
            "{}:{}:{}",
            component.group.as_deref().unwrap_or(""),
            component.module.as_deref().unwrap_or(""),
            component.version.as_deref().unwrap_or("")
        ),
        GradleDependencyKind::Project => format!(
            "Project {}; build tree {}",
            component.project_path.as_deref().unwrap_or(""),
            component.build_tree_path.as_deref().unwrap_or("")
        ),
    }
}

#[cfg(test)]
fn gradle_graph_node(
    module: &LanguageToolNodeId,
    graph: Option<&project::jvm_project::GradleDependencyGraph>,
) -> LanguageToolNode {
    gradle_graph_node_filtered(module, graph, GraphFactFilter::All)
}

#[cfg(test)]
fn maven_graph_node(
    module: &LanguageToolNodeId,
    graph: Option<&project::jvm_project::MavenDependencyGraph>,
) -> LanguageToolNode {
    maven_graph_node_filtered(module, graph, GraphFactFilter::All, MavenGraphScope::All)
}

fn gradle_graph_node_filtered(
    module: &LanguageToolNodeId,
    graph: Option<&project::jvm_project::GradleDependencyGraph>,
    filter: GraphFactFilter,
) -> LanguageToolNode {
    use project::jvm_project::GradleDependencySelector;
    let id = graph_fact_id(module, "gradle", &[]);
    let Some(graph) = graph else {
        return graph_fact_row(
            id,
            "Gradle dependency graph".into(),
            "Not collected; configuration declarations do not resolve dependencies".into(),
            Vec::new(),
        );
    };
    let resolution = &graph.resolution;
    let labels = resolution
        .components
        .iter()
        .map(|component| (component.id.as_str(), gradle_component_label(component)))
        .collect::<HashMap<_, _>>();
    let component_label = |id: &str| {
        labels
            .get(id)
            .cloned()
            .unwrap_or_else(|| "Unavailable component".into())
    };
    let conflicts = resolution
        .components
        .iter()
        .filter(|component| component.conflict_resolution)
        .map(|component| component.id.as_str())
        .collect::<HashSet<_>>();
    let retained_edges = resolution
        .edges
        .iter()
        .filter(|edge| match filter {
            GraphFactFilter::All => true,
            GraphFactFilter::Direct => edge.from == resolution.root_component,
            GraphFactFilter::Conflicts => edge
                .selected
                .as_deref()
                .is_some_and(|selected| conflicts.contains(selected)),
            GraphFactFilter::Unavailable => !edge.resolved,
        })
        .collect::<Vec<_>>();
    let retained_components = retained_edges
        .iter()
        .flat_map(|edge| std::iter::once(edge.from.as_str()).chain(edge.selected.as_deref()))
        .collect::<HashSet<_>>();
    let components = resolution.components.iter().filter(|component| filter == GraphFactFilter::All || component.id == resolution.root_component || retained_components.contains(component.id.as_str()) || (filter == GraphFactFilter::Conflicts && conflicts.contains(component.id.as_str()))).map(|component| graph_fact_row(
        graph_fact_id(module, "gradle-component", &[&resolution.configuration, &component.id]),
        gradle_component_label(component),
        format!("{}; conflict resolution {}; forced {}; selected by rule {}; composite substitution {}; constrained {}; selection causes {:?}; origins and artifacts unavailable",
            if component.id == resolution.root_component { "Root component" } else { "Selected component" },
            component.conflict_resolution, component.forced, component.selected_by_rule,
            component.composite_substitution, component.constrained, component.selection_causes), Vec::new(),
    )).collect();
    let edges = retained_edges.iter().map(|edge| {
        let (requested, request_detail) = match &edge.requested {
            GradleDependencySelector::Module { group, module, version, required_version,
                preferred_version, strict_version, rejected_versions } => (
                format!("{group}:{module}:{version}"),
                format!("Required {required_version}; preferred {preferred_version}; strict {strict_version}; rejected {}", rejected_versions.join(", ")),
            ),
            GradleDependencySelector::Project { build_path, project_path } => (
                format!("Project {project_path}; build {build_path}"), "Project selector".into(),
            ),
        };
        let selected = edge.selected.as_deref().map(&component_label).unwrap_or_else(|| "Unresolved".into());
        graph_fact_row(
            graph_fact_id(module, "gradle-edge", &[&resolution.configuration, &edge.from, &format!("{:?}", edge.requested),
                if edge.constraint { "constraint" } else { "dependency" }]),
            format!("{requested} → {selected}"),
            format!("From {}; {}; constraint {}; {request_detail}; variant {}; {}; source origin unavailable",
                component_label(&edge.from), if edge.from == resolution.root_component { "Root edge" } else { "Transitive edge" },
                edge.constraint, edge.selected_variant.as_deref().unwrap_or("unavailable"),
                edge.failure_category.as_deref().unwrap_or("Resolved metadata; artifact readiness unknown")), Vec::new(),
        )
    }).collect();
    graph_fact_row(
        id,
        format!("Gradle dependency graph: {}", resolution.configuration),
        format!(
            "{} components; {} edges; graph facts filter {}; {} retained edges; structural coverage {}; resolution complete {}; source origins and artifacts unavailable",
            resolution.components.len(),
            resolution.edges.len(),
            filter.label(),
            retained_edges.len(),
            resolution.structural_coverage_known,
            resolution.resolution_complete
        ),
        vec![
            graph_fact_row(
                graph_fact_id(module, "gradle-components", &[&resolution.configuration]),
                "Components".into(),
                "Selected provider components; root and selection causes retained".into(),
                components,
            ),
            graph_fact_row(
                graph_fact_id(module, "gradle-edges", &[&resolution.configuration]),
                "Dependency edges".into(),
                "Provider edges; cycles and diamonds retained without recursive expansion".into(),
                edges,
            ),
        ],
    )
}

fn maven_graph_node_filtered(
    module: &LanguageToolNodeId,
    graph: Option<&project::jvm_project::MavenDependencyGraph>,
    filter: GraphFactFilter,
    scope: MavenGraphScope,
) -> LanguageToolNode {
    let id = graph_fact_id(module, "maven", &[]);
    let Some(graph) = graph else {
        return graph_fact_row(
            id,
            "Maven dependency graph".into(),
            "Not collected; imported classpaths are not a dependency graph".into(),
            Vec::new(),
        );
    };
    let labels = graph
        .components
        .iter()
        .map(|component| {
            (
                component.id.as_str(),
                format!(
                    "{}:{}:{}:{}:{}",
                    component.group_id,
                    component.artifact_id,
                    component.extension,
                    component.classifier,
                    component.version
                ),
            )
        })
        .collect::<HashMap<_, _>>();
    let component_label = |id: &str| {
        labels
            .get(id)
            .cloned()
            .unwrap_or_else(|| "Unavailable component".into())
    };
    let unavailable = graph
        .unavailable_descriptors
        .iter()
        .map(|descriptor| descriptor.component.as_str())
        .collect::<HashSet<_>>();
    let retained_edges = graph
        .edges
        .iter()
        .filter(|edge| scope.matches(&edge.scope))
        .filter(|edge| match filter {
            GraphFactFilter::All => true,
            GraphFactFilter::Direct => edge.direct,
            GraphFactFilter::Conflicts => edge.omitted,
            GraphFactFilter::Unavailable => {
                unavailable.contains(edge.component.as_str())
                    || unavailable.contains(edge.selected.as_str())
            }
        })
        .collect::<Vec<_>>();
    let retained_nodes = retained_edges
        .iter()
        .flat_map(|edge| {
            [
                edge.from_node.as_str(),
                edge.target_node.as_str(),
                edge.selected_node.as_str(),
            ]
        })
        .collect::<HashSet<_>>();
    let occurrences = graph
        .nodes
        .iter()
        .filter(|node| {
            (filter == GraphFactFilter::All && scope == MavenGraphScope::All)
                || node.id == graph.root_node
                || retained_nodes.contains(node.id.as_str())
                || (filter == GraphFactFilter::Unavailable
                    && unavailable.contains(node.component.as_str()))
        })
        .map(|node| {
            graph_fact_row(
                graph_fact_id(module, "maven-node", &[&node.id]),
                component_label(&node.component),
                format!(
                    "{}; occurrence {}; source origins and artifacts unavailable",
                    if node.id == graph.root_node {
                        "Root occurrence"
                    } else {
                        "Dependency occurrence"
                    },
                    node.id
                ),
                Vec::new(),
            )
        })
        .collect();
    let exclusions = |values: &[project::jvm_project::MavenDependencyExclusion]| {
        values
            .iter()
            .map(|value| {
                format!(
                    "{}:{}:{}:{}",
                    value.group_id, value.artifact_id, value.extension, value.classifier
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let edges = retained_edges.iter().map(|edge| graph_fact_row(
        graph_fact_id(module, "maven-edge", &[&edge.from_node, &edge.target_node]),
        format!("{} → {}", component_label(&edge.component), component_label(&edge.selected)),
        format!("{}; scope {}; optional {}; omitted by mediation {}; from {}; target occurrence {}; selected occurrence {}; version constraint {}; premanaged version {}; premanaged scope {}; exclusions [{}]; source origin unavailable",
            if edge.direct { "Direct dependency" } else { "Transitive dependency" }, edge.scope, edge.optional, edge.omitted,
            component_label(&edge.from), edge.target_node, edge.selected_node, edge.version_constraint.as_deref().unwrap_or("unknown"),
            edge.premanaged_version.as_deref().unwrap_or("none"), edge.premanaged_scope.as_deref().unwrap_or("none"), exclusions(&edge.exclusions)), Vec::new(),
    )).collect();
    let descriptors = graph
        .unavailable_descriptors
        .iter()
        .filter(|_| matches!(filter, GraphFactFilter::All | GraphFactFilter::Unavailable))
        .map(|descriptor| {
            graph_fact_row(
                graph_fact_id(module, "maven-descriptor", &[&descriptor.component]),
                component_label(&descriptor.component),
                format!("Descriptor unavailable; {}", descriptor.problem),
                Vec::new(),
            )
        })
        .collect();
    graph_fact_row(
        id,
        "Maven dependency graph".into(),
        format!(
            "{} occurrences; {} edges; graph facts filter {}; Maven scope {}; {} retained edges; structural coverage {}; collection complete {}; {} cycle reports; exact cycle occurrences {}; source origins and artifacts unavailable; {}",
            graph.nodes.len(),
            graph.edges.len(),
            filter.label(),
            scope.label(),
            retained_edges.len(),
            graph.structural_coverage_known,
            graph.collection_complete,
            graph.cycle_count,
            if graph.cycle_occurrences_known {
                "known"
            } else {
                "unknown"
            },
            graph
                .problem
                .as_deref()
                .unwrap_or("No collection problem reported")
        ),
        vec![
            graph_fact_row(
                graph_fact_id(module, "maven-nodes", &[]),
                "Occurrences".into(),
                "Distinct dependency occurrences; coordinate duplicates remain separate".into(),
                occurrences,
            ),
            graph_fact_row(
                graph_fact_id(module, "maven-edges", &[]),
                "Dependency edges".into(),
                "Requested and selected occurrences, scopes and mediation retained".into(),
                edges,
            ),
            graph_fact_row(
                graph_fact_id(module, "maven-descriptors", &[]),
                "Unavailable descriptors".into(),
                "Provider collection gaps; repair cache/build settings and recollect".into(),
                descriptors,
            ),
        ],
    )
}

fn gradle_configurations_node(
    module_id: &LanguageToolNodeId,
    gradle: &GradleProject,
) -> LanguageToolNode {
    let coverage = if !gradle.configurations_known {
        "Provider configuration declarations unavailable; refresh with a supported provider"
            .to_string()
    } else if gradle.configurations.is_empty() {
        "No declared configurations; resolved dependencies unavailable".to_string()
    } else {
        format!(
            "{} declarations; {}",
            gradle.configurations.len(),
            if gradle.dependency_graph.is_some() {
                "selected graph shown separately"
            } else {
                "resolved dependencies unavailable"
            }
        )
    };
    LanguageToolNode {
        id: LanguageToolNodeId(format!("{}:gradle-configurations", module_id.0)),
        label: "Configurations".into(),
        secondary_label: Some(coverage.clone()),
        accessibility_label: format!("Configurations; {coverage}"),
        icon: None,
        children: gradle.configurations.iter().map(|configuration| {
            let configuration_id = gradle_configuration_node_id(module_id, &configuration.name);
            let roles = format!("{}; {}",
                if configuration.resolvable { "Resolvable" } else { "Not resolvable" },
                if configuration.consumable { "consumable" } else { "not consumable" }
            );
            LanguageToolNode {
                label: configuration.name.clone(),
                secondary_label: Some(roles.clone()),
                accessibility_label: format!(
                    "Gradle configuration {}; {roles}; {}; {} declared parents; resolved dependencies unavailable",
                    configuration.name, configuration.description.as_deref().unwrap_or("no description"),
                    configuration.extends_from.len()
                ),
                icon: None,
                children: configuration.extends_from.iter().map(|parent| LanguageToolNode {
                    id: LanguageToolNodeId(format!("gradle-configuration-parent:{}:{}:{}:{parent}", configuration_id.0.len(), configuration_id.0, parent.len())),
                    label: parent.clone(),
                    secondary_label: Some("Declared parent; resolved dependencies unavailable".into()),
                    accessibility_label: format!(
                        "Configuration {} extends declared parent {parent}; resolved dependencies unavailable", configuration.name
                    ),
                    icon: None,
                    children: Vec::new(),
                    enabled: false,
                    activation_label: None,
                }).collect(),
                id: configuration_id,
                enabled: false,
                activation_label: None,
            }
        }).collect(),
        enabled: false,
        activation_label: None,
    }
}

fn maven_metadata_identity(parts: &[&str]) -> String {
    parts
        .iter()
        .map(|part| format!("{}:{part}", part.len()))
        .collect::<Vec<_>>()
        .join(":")
}

fn gradle_task_node_id(
    module_id: &LanguageToolNodeId,
    gradle: &GradleProject,
    path: &str,
) -> LanguageToolNodeId {
    LanguageToolNodeId(format!(
        "{}:gradle-task:{}:{}:{path}",
        module_id.0,
        gradle.build_root.path.as_unix_str(),
        gradle.project_path
    ))
}

fn ordered_operation_root_id() -> LanguageToolNodeId {
    LanguageToolNodeId("groovy-ordered-operations".into())
}

fn compose_ordered_operations(
    ordered: &[JvmBuildOperationReceipt],
    current: &HashMap<LanguageToolNodeId, JvmBuildOperationReceipt>,
) -> anyhow::Result<JvmBuildOperationReceipt> {
    anyhow::ensure!(
        !ordered.is_empty() && ordered.len() <= 32,
        "Select between one and 32 ordered operations"
    );
    let first = ordered
        .first()
        .ok_or_else(|| anyhow::anyhow!("Select a current operation to start an ordered run"))?;
    let first_operation = first.operations.first().ok_or_else(|| {
        anyhow::anyhow!("Operation metadata is unavailable; refresh and select it again")
    })?;
    let gradle = matches!(
        first_operation,
        project::jvm_project::JvmBuildOperation::GradleTask { .. }
    );
    for receipt in ordered {
        anyhow::ensure!(
            receipt.operations.len() == 1
                && receipt.server_id == first.server_id
                && receipt.worktree_id == first.worktree_id
                && receipt.generation == first.generation
                && receipt.module_root == first.module_root,
            "Ordered operations must belong to the same current project module; clear the selection before changing modules"
        );
        anyhow::ensure!(
            current.values().any(|candidate| candidate == receipt),
            "Ordered operation metadata is stale; refresh and select the operations again"
        );
        anyhow::ensure!(
            receipt.operations.iter().all(|operation| matches!(
                operation,
                project::jvm_project::JvmBuildOperation::GradleTask { .. }
            ) == gradle),
            "Maven and Gradle operations cannot share an ordered run"
        );
    }
    Ok(JvmBuildOperationReceipt {
        operations: ordered
            .iter()
            .flat_map(|receipt| receipt.operations.clone())
            .collect(),
        ..first.clone()
    })
}

fn ordered_operation_tree(ordered: &[JvmBuildOperationReceipt]) -> LanguageToolNode {
    let children = ordered
        .iter()
        .enumerate()
        .map(|(index, receipt)| {
            let operation = receipt.operations.first();
            let name = match operation {
                Some(project::jvm_project::JvmBuildOperation::MavenPhase { lifecycle, phase }) => {
                    format!("{lifecycle}: {phase}")
                }
                Some(project::jvm_project::JvmBuildOperation::MavenGoal {
                    group,
                    artifact,
                    version,
                    goal,
                }) => format!("{group}:{artifact}:{version}:{goal}"),
                Some(project::jvm_project::JvmBuildOperation::GradleTask { path }) => path.clone(),
                None => "Unavailable operation; clear and select again".into(),
            };
            let label = format!("{}. {name}", index + 1);
            LanguageToolNode {
                id: LanguageToolNodeId(format!("groovy-ordered-operation:{index}")),
                label: label.clone(),
                secondary_label: None,
                icon: None,
                accessibility_label: label,
                children: Vec::new(),
                enabled: false,
                activation_label: None,
            }
        })
        .collect();
    let label = format!("Ordered offline operations ({})", ordered.len());
    LanguageToolNode {
        id: ordered_operation_root_id(),
        label: label.clone(),
        secondary_label: None,
        icon: None,
        accessibility_label: format!("{label}. Run in the order shown."),
        children,
        enabled: true,
        activation_label: None,
    }
}

fn operation_selections_for_snapshot(
    snapshot: &JdtlsProjectSnapshot,
    current: bool,
    view: BuildToolView,
    path_style: util::paths::PathStyle,
) -> (
    HashMap<LanguageToolNodeId, JvmBuildOperationReceipt>,
    HashMap<LanguageToolNodeId, String>,
) {
    let mut selections = HashMap::new();
    let mut unavailable = HashMap::new();
    if !current || view == BuildToolView::Dependencies {
        return (selections, unavailable);
    }
    for module in &snapshot.projects {
        let id = jdtls_module_node_id(snapshot, &module.module_root, view);
        let mut admit = |id, operation| match JvmBuildOperationReceipt::new(
            snapshot,
            current,
            module.module_root.clone(),
            vec![operation],
            path_style,
        ) {
            Ok(receipt) => {
                selections.insert(id, receipt);
            }
            Err(error) => {
                unavailable.insert(id, error.to_string());
            }
        };
        if view == BuildToolView::Maven
            && let Some(maven) = &module.maven
        {
            for lifecycle in &maven.lifecycles {
                for phase in &lifecycle.phases {
                    admit(
                        LanguageToolNodeId(format!(
                            "{}:maven:phase:{}",
                            id.0,
                            maven_metadata_identity(&[&lifecycle.id, phase])
                        )),
                        JvmBuildOperation::MavenPhase {
                            lifecycle: lifecycle.id.clone(),
                            phase: phase.clone(),
                        },
                    );
                }
            }
            for plugin in &maven.plugins {
                let version = plugin.version.as_deref().unwrap_or_default();
                let plugin_id =
                    maven_metadata_identity(&[&plugin.group_id, &plugin.artifact_id, version]);
                for goal in &plugin.goals {
                    admit(
                        LanguageToolNodeId(format!(
                            "{}:maven:goal:{}",
                            id.0,
                            maven_metadata_identity(&[&plugin_id, &goal.name])
                        )),
                        JvmBuildOperation::MavenGoal {
                            group: plugin.group_id.clone(),
                            artifact: plugin.artifact_id.clone(),
                            version: version.to_string(),
                            goal: goal.name.clone(),
                        },
                    );
                }
            }
        }
        if view == BuildToolView::Gradle
            && let Some(gradle) = &module.gradle
        {
            for task in &gradle.tasks {
                admit(
                    gradle_task_node_id(&id, gradle, &task.path),
                    JvmBuildOperation::GradleTask {
                        path: task.path.clone(),
                    },
                );
            }
        }
    }
    (selections, unavailable)
}

fn annotate_operation_rows(
    rows: &mut [LanguageToolNode],
    selections: &HashMap<LanguageToolNodeId, JvmBuildOperationReceipt>,
    unavailable: &HashMap<LanguageToolNodeId, String>,
) {
    for row in rows {
        if selections.contains_key(&row.id) {
            row.accessibility_label = row.accessibility_label.replace("execution unavailable in this view","select this row and use Run selected offline operation; configurable arguments unavailable")
                .replace("execution unavailable","select this row and use Run selected offline operation; configurable arguments unavailable");
        }
        if let Some(reason) = unavailable.get(&row.id) {
            row.accessibility_label.push_str("; ");
            row.accessibility_label.push_str(reason);
            row.secondary_label = Some(reason.clone());
        }
        annotate_operation_rows(&mut row.children, selections, unavailable);
    }
}

fn maven_metadata_rows(
    module_id: &LanguageToolNodeId,
    maven: &project::jvm_project::MavenProject,
    basic_only: bool,
) -> Vec<LanguageToolNode> {
    let identity = maven_metadata_identity;
    let row = |suffix: String, label: String, detail: String, children: Vec<LanguageToolNode>| {
        LanguageToolNode {
            id: LanguageToolNodeId(format!("{}:maven:{suffix}", module_id.0)),
            accessibility_label: format!("{label}; {detail}"),
            label,
            secondary_label: Some(detail),
            icon: None,
            children,
            enabled: false,
            activation_label: None,
        }
    };
    let lifecycle_rows = maven
        .lifecycles
        .iter()
        .map(|lifecycle| {
            let phases: Vec<_> = lifecycle
                .phases
                .iter()
                .filter(|phase| !basic_only || BASIC_MAVEN_PHASES.contains(&phase.as_str()))
                .map(|phase| {
                    row(
                        format!("phase:{}", identity(&[&lifecycle.id, phase])),
                        phase.clone(),
                        "Lifecycle phase; execution unavailable in this view".into(),
                        Vec::new(),
                    )
                })
                .collect();
            let detail = if basic_only {
                format!(
                    "Basic Maven phases; {} of {} provider phases shown",
                    phases.len(),
                    lifecycle.phases.len()
                )
            } else {
                "Ordered Maven lifecycle phases".into()
            };
            row(
                format!("lifecycle:{}", lifecycle.id),
                lifecycle.id.clone(),
                detail,
                phases,
            )
        })
        .collect();
    let plugins = maven
        .plugins
        .iter()
        .map(|plugin| {
            let plugin_id = identity(&[
                &plugin.group_id,
                &plugin.artifact_id,
                plugin.version.as_deref().unwrap_or_default(),
            ]);
            let goals = plugin
                .goals
                .iter()
                .map(|goal| {
                    let declaration_identity = |name: &str| identity(&[&plugin_id, &goal.name, name]);
                    let parameters = if goal.parameters_known {
                        goal.parameters
                            .iter()
                            .map(|parameter| {
                                let required = if parameter.required {
                                    "Required for goal injection; not a required CLI argument"
                                } else {
                                    "Optional for goal injection"
                                };
                                let editable = if parameter.editable {
                                    "Configurable in POM"
                                } else {
                                    "Read-only goal parameter"
                                };
                                let detail = format!(
                                    "{}; POM alias {}; {required}; {editable}; {}; effective value and CLI property mapping unknown",
                                    parameter.declared_type.as_deref().unwrap_or("Type not declared"),
                                    parameter.alias.as_deref().unwrap_or("not declared"),
                                    parameter.description.as_deref().unwrap_or("Cached plugin declaration")
                                );
                                row(
                                    format!("parameter:{}", declaration_identity(&parameter.name)),
                                    parameter.name.clone(),
                                    detail,
                                    Vec::new(),
                                )
                            })
                            .collect()
                    } else {
                        vec![row(
                            format!("parameters-unknown:{}", declaration_identity("")),
                            "Parameter declarations unavailable".into(),
                            "This provider has no cached declarations; effective values and CLI property mapping unknown".into(),
                            Vec::new(),
                        )]
                    };
                    row(
                        format!("goal:{}", identity(&[&plugin_id, &goal.name])),
                        goal.name.clone(),
                        format!(
                            "{}; default phase {}; {}; execution unavailable in this view",
                            goal.description.as_deref().unwrap_or("Plugin goal"),
                            goal.phase.as_deref().unwrap_or("not declared"),
                            if goal.parameters_known {
                                format!("{} cached parameter declarations", goal.parameters.len())
                            } else {
                                "Parameter declaration coverage unknown".into()
                            }
                        ),
                        parameters,
                    )
                })
                .collect();
            row(
                format!("plugin:{plugin_id}"),
                format!(
                    "{}:{}:{}",
                    plugin.group_id,
                    plugin.artifact_id,
                    plugin.version.as_deref().unwrap_or("version unknown")
                ),
                format!(
                    "{}; cached goals {}",
                    if plugin.managed_only {
                        "Plugin management only; no build binding"
                    } else {
                        "Build plugin"
                    },
                    if plugin.goals_known {
                        "available"
                    } else {
                        "unavailable; check import and cache"
                    }
                ),
                goals,
            )
        })
        .collect();
    let definitions = maven.profile_definitions.iter().map(|definition| {
        let source = definition.source_pom.as_ref().map(|path| path.path.as_unix_str());
        // Preserve saved tree selection when an origin path is unavailable.
        let identity_source = source.unwrap_or("external parent; path hidden");
        let source = source.unwrap_or("origin path unavailable");
        row(format!("profile:{}", identity(&[&definition.id, &definition.source_group_id, &definition.source_artifact_id, &definition.source_version, identity_source, if definition.inherited { "inherited" } else { "local" }])),
            definition.id.clone(), format!("{}; source {}:{}:{} ({source}); {} at source; changing activation unavailable in this view",
                if definition.inherited { "Parent definition" } else { "Local definition" }, definition.source_group_id,
                definition.source_artifact_id, definition.source_version, if definition.active_at_source { "active" } else { "inactive" }), Vec::new())
    }).collect::<Vec<_>>();
    let activity = maven.profiles.iter().map(|(id, active)| row(
        format!("profile-activity:{}", identity(&[id])), id.clone(),
        format!("Provider local declaration or effective active profile: {}; changing activation unavailable in this view",
            if *active { "active" } else { "inactive" }), Vec::new(),
    )).collect();
    let profiles = vec![
        row(
            "profile-definitions".into(),
            "Definitions".into(),
            "Project and parent profile declarations with source origins".into(),
            definitions,
        ),
        row(
            "profile-activity".into(),
            "Activity".into(),
            "Local declarations and effective active IDs reported by Maven".into(),
            activity,
        ),
    ];
    vec![
        row(
            "reactor".into(),
            "Reactor".into(),
            maven
                .reactor_position
                .map(|position| {
                    format!(
                        "Cached reactor build position {}; import freshness unverified",
                        position + 1
                    )
                })
                .unwrap_or_else(|| {
                    format!(
                        "Build order unknown; {}",
                        maven
                            .reactor_order_problem
                            .as_deref()
                            .unwrap_or("provider did not supply order")
                    )
                }),
            maven
                .modules
                .iter()
                .map(|module| {
                    row(
                        format!("module:{}", module.path.as_unix_str()),
                        module.path.as_unix_str().into(),
                        "Effective module link".into(),
                        Vec::new(),
                    )
                })
                .collect(),
        ),
        row(
            "lifecycles".into(),
            "Lifecycle".into(),
            "Maven runtime phase order; execution unavailable in this view".into(),
            lifecycle_rows,
        ),
        row(
            "plugins".into(),
            "Plugins".into(),
            "Effective build plugins and managed-only entries".into(),
            plugins,
        ),
        row(
            "profiles".into(),
            "Profiles".into(),
            if maven.profile_definitions_known {
                "Cached profile origins; import freshness unverified"
            } else {
                "Profile declaration coverage incomplete"
            }
            .into(),
            profiles,
        ),
        row(
            "repositories".into(),
            "Repositories".into(),
            "Repository metadata is unavailable in this view".into(),
            Vec::new(),
        ),
        row(
            "configurations".into(),
            "Saved configurations".into(),
            "Not available in this view; use Tasks for configured commands".into(),
            Vec::new(),
        ),
    ]
}

fn kotlin_rows_for_view(
    snapshot: &KotlinWorkspaceSnapshot,
    view: BuildToolView,
) -> (
    Vec<LanguageToolNode>,
    HashMap<LanguageToolNodeId, DependencyNavigationTarget>,
) {
    let mut roots = Vec::new();
    let mut navigation = HashMap::new();
    for module in &snapshot.projects {
        if (view == BuildToolView::Gradle && module.build_system != KotlinBuildSystem::Gradle)
            || (view == BuildToolView::Maven && module.build_system != KotlinBuildSystem::Maven)
        {
            continue;
        }
        let module_id = LanguageToolNodeId(format!(
            "kotlin:{}:{}:{}",
            snapshot.worktree_id,
            module.module_root.path.as_unix_str(),
            module.name
        ));
        let mut children = Vec::new();
        for (index, dependency) in module.dependencies.iter().enumerate() {
            let (kind, name, scope) = match dependency {
                KotlinWorkspaceDependency::Module { name, scope } => {
                    ("module", name, scope.as_deref())
                }
                KotlinWorkspaceDependency::Artifact { coordinate, scope } => {
                    ("artifact", coordinate, scope.as_deref())
                }
            };
            let scope = scope.unwrap_or("unknown");
            children.push(LanguageToolNode {
                id: LanguageToolNodeId(format!("{}:{kind}:{index}", module_id.0)),
                label: name.clone(),
                secondary_label: Some(format!(
                    "Provider classpath scope {scope}; directness unknown"
                )),
                icon: None,
                accessibility_label: format!(
                    "Kotlin LSP {kind} {name}; classpath scope {scope}; direct or transitive relationship unknown"
                ),
                children: Vec::new(),
                enabled: false,
                activation_label: None,
            });
        }
        if module.unidentified_libraries > 0 {
            children.push(LanguageToolNode {
                id: LanguageToolNodeId(format!("{}:unnamed-libraries", module_id.0)),
                label: format!("{} unnamed libraries", module.unidentified_libraries),
                secondary_label: Some("Provider did not supply safe coordinates".to_string()),
                icon: None,
                accessibility_label: format!(
                    "{} provider-reported libraries without safe coordinates",
                    module.unidentified_libraries
                ),
                children: Vec::new(),
                enabled: false,
                activation_label: None,
            });
        }
        navigation.insert(
            module_id.clone(),
            DependencyNavigationTarget::ModuleManifest {
                root: module.module_root.clone(),
                build_system: Some(module.build_system),
            },
        );
        roots.push(LanguageToolNode {
            id: module_id,
            label: module.name.clone(),
            secondary_label: Some("Kotlin LSP module".to_string()),
            icon: None,
            accessibility_label: format!("Kotlin LSP module {}", module.name),
            children,
            enabled: true,
            activation_label: Some("Open module manifest".to_string()),
        });
    }
    (roots, navigation)
}

fn visible_module_manifest(
    module_root: &ProjectPath,
    build_system: Option<KotlinBuildSystem>,
    mut visible: impl FnMut(&RelPath) -> bool,
) -> Option<ProjectPath> {
    let mut manifest = |name| {
        let Ok(file_name) = RelPath::from_unix_str(name) else {
            return None;
        };
        let candidate = module_root.path.join(file_name);
        if visible(candidate.as_ref()) {
            Some(ProjectPath {
                worktree_id: module_root.worktree_id,
                path: candidate.into(),
            })
        } else {
            None
        }
    };
    let maven = manifest("pom.xml");
    let gradle = manifest("build.gradle.kts").or_else(|| manifest("build.gradle"));
    match build_system {
        Some(KotlinBuildSystem::Maven) => maven,
        Some(KotlinBuildSystem::Gradle) => gradle,
        Some(KotlinBuildSystem::Unknown) | None => match (maven, gradle) {
            (Some(_), Some(_)) => None,
            (maven, gradle) => maven.or(gradle),
        },
    }
}

fn mark_manifest_navigation(
    rows: &mut [LanguageToolNode],
    navigation: &HashMap<LanguageToolNodeId, DependencyNavigation>,
) {
    for row in rows {
        if !navigation.contains_key(&row.id) {
            row.enabled = false;
            row.activation_label = None;
            row.accessibility_label
                .push_str("; no unambiguous visible build manifest");
        }
    }
}

impl Panel for DependenciesPanel {
    fn persistent_name() -> &'static str {
        "Dependencies"
    }

    fn panel_key() -> &'static str {
        PANEL_KEY
    }

    fn position(&self, _: &Window, _: &App) -> DockPosition {
        self.position
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        matches!(
            position,
            DockPosition::Left | DockPosition::Bottom | DockPosition::Right
        )
    }

    fn set_position(&mut self, position: DockPosition, _: &mut Window, cx: &mut Context<Self>) {
        if self.position != position {
            self.position = position;
            self.preferences.dock_position = position.into();
            self.persist_preferences(cx);
            cx.notify();
        }
    }

    fn default_size(&self, _: &Window, _: &App) -> Pixels {
        px(280.)
    }

    fn icon(&self, _: &Window, _: &App) -> Option<ui::IconName> {
        None
    }

    fn icon_tooltip(&self, _: &Window, _: &App) -> Option<&'static str> {
        Some("Groovy Dependencies")
    }

    fn toggle_action(&self) -> Box<dyn Action> {
        Box::new(ToggleDependenciesPanel)
    }

    fn starts_open(&self, _: &Window, _: &App) -> bool {
        false
    }

    fn set_active(&mut self, active: bool, _: &mut Window, cx: &mut Context<Self>) {
        if self.active == active {
            return;
        }
        self.active = active;
        if active {
            self.host.take_dirty();
            self.refresh(cx);
        }
    }

    fn activation_priority(&self) -> u32 {
        8
    }
}

impl Focusable for DependenciesPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<PanelEvent> for DependenciesPanel {}

impl Drop for DependenciesPanel {
    fn drop(&mut self) {
        if let Some(write) = self.preferences_write.take() {
            write.detach();
        }
    }
}

impl Render for DependenciesPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = self.host.visible_rows_snapshot();
        let selected = self.host.selected().cloned();
        let status = status_message(self.host.status());
        let can_expand_all = self.host.can_expand_all();
        let can_collapse_all = self.host.can_collapse_all();
        let show_toolbar = self.preferences.show_toolbar(self.view);
        let can_collect_graph = self.selected_graph().is_some()
            && self.graph_query_token.is_none()
            && !self.project.read(cx).is_remote();
        let can_run_operation = self
            .host
            .selected()
            .is_some_and(|id| self.operation_selections.contains_key(id))
            && !self.host_unavailable(cx)
            && !self.project.read(cx).is_remote();
        let can_run_ordered = !self.host_unavailable(cx)
            && !self.project.read(cx).is_remote()
            && compose_ordered_operations(&self.ordered_operations, &self.operation_selections)
                .is_ok();
        let can_cancel_graph = self.graph_query_is_pending(cx);
        let can_apply_build_selection =
            self.build_phase_token.is_none() && self.selected_build_worktree(cx).is_ok();
        let can_cancel_import = self.build_phase_is_pending(cx);
        let can_inspect_graph = self
            .selected_graph_inspection(GraphInspectionMode::Neighbours, cx)
            .is_some();
        let click_panel = cx.weak_entity();
        let toggle_panel = cx.weak_entity();
        v_flex()
            .id("groovy-dependencies-panel")
            .role(Role::Group)
            .aria_label(format!("{} panel", self.view.label()))
            .key_context(PANEL_KEY)
            .track_focus(&self.focus_handle)
            .size_full()
            .on_action(cx.listener(|panel, _: &RunSelectedOfflineOperation, window, cx| panel.run_selected_offline_operation(window,cx)))
            .on_action(cx.listener(|panel, _: &AddSelectedOrderedOperation, _, cx| panel.add_selected_ordered_operation(cx)))
            .on_action(cx.listener(|panel, _: &RemoveLastOrderedOperation, _, cx| panel.remove_last_ordered_operation(cx)))
            .on_action(cx.listener(|panel, _: &ClearOrderedOperations, _, cx| panel.clear_ordered_operations(cx)))
            .on_action(cx.listener(|panel, _: &RunOrderedOfflineOperations, window, cx| panel.run_ordered_offline_operations(window, cx)))
            .on_action(cx.listener(|panel, _: &ApplySavedMavenSelection, _, cx| panel.apply_saved_build_selection(true, cx)))
            .on_action(cx.listener(|panel, _: &ImportGradleSelection, _, cx| panel.apply_saved_build_selection(false, cx)))
            .on_action(cx.listener(|panel, _: &CancelSelectedImport, _, cx| panel.cancel_selected_import(cx)))
            .on_action(cx.listener(|panel, _: &InspectGraphNeighbours, _, cx| panel.inspect_selected_graph(GraphInspectionMode::Neighbours, cx)))
            .on_action(cx.listener(|panel, _: &InspectGraphRootPaths, _, cx| panel.inspect_selected_graph(GraphInspectionMode::ShortestRootPaths, cx)))
            .on_action(cx.listener(|panel, _: &ClearGraphInspection, _, cx| panel.clear_graph_inspection(cx)))
            .on_action(cx.listener(|panel, _: &ClearGraphFactSearch, window, cx| panel.clear_graph_fact_search(window, cx)))
            .on_action(cx.listener(|panel, _: &CycleGraphFactFilter, _, cx| panel.cycle_graph_fact_filter(cx)))
            .on_action(cx.listener(|panel, _: &CycleMavenGraphScope, _, cx| panel.cycle_maven_graph_scope(cx)))
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::Refresh, _, cx| panel.refresh(cx)),
            )
            .on_action(cx.listener(|panel, _: &CollectSelectedGraph, _, cx| panel.collect_selected_graph(cx)))
            .on_action(cx.listener(|panel, _: &CancelGraphCollection, _, cx| panel.cancel_graph_collection(cx)))
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::ExpandAll, _, cx| {
                    panel.host.expand_all();
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::CollapseAll, _, cx| {
                    panel.host.collapse_all();
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectNext, _, cx| {
                    panel.host.select_next();
                    panel.host.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectPrevious, _, cx| {
                    panel.host.select_previous();
                    panel.host.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectFirst, _, cx| {
                    panel.host.select_first();
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectLast, _, cx| {
                    panel.host.select_last();
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectParent, _, cx| {
                    panel.host.select_parent();
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectFirstChild, _, cx| {
                    panel.host.select_first_child();
                    cx.notify();
                }),
            )
            .on_action(cx.listener(|panel, _: &ShowDependencyDiagram, _, cx| panel.show_dependency_diagram(cx)))
            .on_action(cx.listener(|panel, _: &CloseDependencyDiagram, _, cx| panel.close_dependency_diagram(cx)))
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::ToggleExpanded, _, cx| {
                    panel.host.toggle_selected();
                    cx.notify();
                }),
            )
            .on_action(cx.listener(
                |panel, _: &language_tool_tree::ActivateSelected, window, cx| {
                    panel.activate_selected(window, cx)
                },
            ))
            .child(
                div()
                    .key_context("GroovyDependenciesPanelControls")
                    .min_h_9()
                    .px_2()
                    .py_1()
                    .flex()
                    .items_center()
                    .flex_wrap()
                    .gap_1()
                    .justify_between()
                    .child(self.view.label())
                    .when(self.view != BuildToolView::Dependencies, |header| header.child(
                        Button::new("groovy-build-view-options", "Options")
                            .aria_label(format!("{} view options", self.view.label()))
                            .toggle_state(self.show_options)
                            .tab_index(0_isize)
                            .on_click(cx.listener(|panel, _, _, cx| {
                                panel.show_options = !panel.show_options;
                                cx.notify();
                            })),
                    ))
                    .when(self.view == BuildToolView::Gradle, |header| header.child(
                        Button::new("groovy-gradle-synchronize-metadata", "Synchronize metadata")
                            .aria_label(if self.project.read(cx).is_remote() {
                                "Gradle metadata synchronization unavailable for remote projects"
                            } else {
                                "Synchronize local Gradle metadata; executes trusted build configuration scripts"
                            })
                            .disabled(self.project.read(cx).is_remote())
                            .tab_index(0_isize)
                            .on_click(cx.listener(|panel, _, _, cx| panel.synchronize_gradle_metadata(cx)))
                    ))
                    .when(self.view != BuildToolView::Gradle, |header| header.child(
                        Button::new("groovy-apply-saved-maven-selection", "Apply saved Maven selection")
                            .aria_label(if self.project.read(cx).is_remote() { "Saved Maven selection unavailable for remote projects" } else { "Apply saved Maven roots and exclusions for the selected local project folder; executes trusted build configuration" })
                            .disabled(!can_apply_build_selection).tab_index(0_isize)
                            .on_click(cx.listener(|panel, _, _, cx| panel.apply_saved_build_selection(true, cx)))
                    ))
                    .when(self.view != BuildToolView::Maven, |header| header.child(
                        Button::new("groovy-import-gradle-selection", "Import Gradle selection")
                            .aria_label(if self.project.read(cx).is_remote() { "Gradle selection import unavailable for remote projects" } else { "Import effective Gradle selection for the selected local project folder; executes trusted build configuration" })
                            .disabled(!can_apply_build_selection).tab_index(0_isize)
                            .on_click(cx.listener(|panel, _, _, cx| panel.apply_saved_build_selection(false, cx)))
                    ))
                    .child(Button::new("groovy-cancel-selected-import", "Cancel selected import")
                        .aria_label("Cancel only the current build import started by this panel")
                        .disabled(!can_cancel_import).tab_index(0_isize)
                        .on_click(cx.listener(|panel, _, _, cx| panel.cancel_selected_import(cx))))
                    .child(
                        Button::new("groovy-graph-facts-filter", format!("Graph: {}", self.preferences.graph_fact_filter.label()))
                            .aria_label(format!("Dependency graph fact filter: {}; activate for {}; filters admitted metadata without resolving dependencies", self.preferences.graph_fact_filter.label(), self.preferences.graph_fact_filter.next().label()))
                            .aria_value(self.preferences.graph_fact_filter.label())
                            .tab_index(0_isize)
                            .on_click(cx.listener(|panel, _, _, cx| panel.cycle_graph_fact_filter(cx)))
                    )
                    .when(self.view != BuildToolView::Gradle, |header| header.child(
                        Button::new("groovy-maven-graph-scope", format!("Maven scope: {}", self.preferences.maven_graph_scope.label()))
                            .aria_label(format!("Maven dependency graph scope: {}; activate for {}; filters admitted edge scopes without resolving dependencies", self.preferences.maven_graph_scope.label(), self.preferences.maven_graph_scope.next().label()))
                            .aria_value(self.preferences.maven_graph_scope.label())
                            .tab_index(0_isize)
                            .on_click(cx.listener(|panel, _, _, cx| panel.cycle_maven_graph_scope(cx)))
                    ))
                    .child(Button::new("groovy-inspect-graph-neighbours", "Neighbours")
                        .aria_label("Inspect selected dependency neighbours in admitted graph facts")
                        .aria_value(if self.graph_inspection.as_ref().is_some_and(|inspection| inspection.mode == GraphInspectionMode::Neighbours) { "Active" } else { "Inactive" })
                        .disabled(!can_inspect_graph).tab_index(0_isize)
                        .on_click(cx.listener(|panel, _, _, cx| panel.inspect_selected_graph(GraphInspectionMode::Neighbours, cx))))
                    .child(Button::new("groovy-inspect-graph-root-paths", "Shortest root paths")
                        .aria_label("Inspect all shortest root paths to the selected dependency in admitted graph facts")
                        .aria_value(if self.graph_inspection.as_ref().is_some_and(|inspection| inspection.mode == GraphInspectionMode::ShortestRootPaths) { "Active" } else { "Inactive" })
                        .disabled(!can_inspect_graph).tab_index(0_isize)
                        .on_click(cx.listener(|panel, _, _, cx| panel.inspect_selected_graph(GraphInspectionMode::ShortestRootPaths, cx))))
                    .child(Button::new("groovy-clear-graph-inspection", "Clear inspection")
                        .aria_label("Clear selected dependency graph inspection")
                        .disabled(self.graph_inspection.is_none()).tab_index(0_isize)
                        .on_click(cx.listener(|panel, _, _, cx| panel.clear_graph_inspection(cx))))
                    .child(
                        Button::new("groovy-collect-offline-graph", "Collect offline graph")
                            .aria_label(if self.project.read(cx).is_remote() {
                                "Offline dependency graph collection unavailable for remote projects"
                            } else {
                                "Collect offline dependency graph for selected Maven project or Gradle resolvable configuration; executes trusted build configuration scripts and resolves cached metadata"
                            })
                            .disabled(!can_collect_graph)
                            .tab_index(0_isize)
                            .on_click(cx.listener(|panel, _, _, cx| panel.collect_selected_graph(cx)))
                    )
                    .child(
                        Button::new("groovy-cancel-graph-collection", "Cancel graph collection")
                            .aria_label("Cancel only this panel's current dependency graph collection")
                            .disabled(!can_cancel_graph)
                            .tab_index(0_isize)
                            .on_click(cx.listener(|panel, _, _, cx| panel.cancel_graph_collection(cx)))
                    )
                    .when(self.view!=BuildToolView::Dependencies,|header|header.child(
                        Button::new("groovy-run-selected-offline-operation","Run selected offline operation")
                            .aria_label(if self.project.read(cx).is_remote(){"Discovered offline build execution unavailable for remote projects"}
                                else{"Run selected offline Maven phase, plugin goal or Gradle task through Tasks; executes trusted local build scripts"})
                            .disabled(!can_run_operation).tab_index(0_isize)
                            .on_click(cx.listener(|panel,_,window,cx|panel.run_selected_offline_operation(window,cx)))))
                    .when(self.view != BuildToolView::Dependencies, |header| header.child(
                        v_flex().id("groovy-ordered-operation-controls").role(Role::Group).aria_label("Ordered offline operation controls")
                            .child(Button::new("groovy-add-ordered-operation", "Add selected operation")
                                .aria_label("Add selected current operation to the ordered offline run")
                                .disabled(!can_run_operation || self.ordered_operations.len() >= 32).tab_index(0_isize)
                                .on_click(cx.listener(|panel, _, _, cx| panel.add_selected_ordered_operation(cx))))
                            .child(Button::new("groovy-remove-last-ordered-operation", "Remove last operation")
                                .aria_label("Remove last ordered operation without starting execution")
                                .disabled(self.ordered_operations.is_empty()).tab_index(0_isize)
                                .on_click(cx.listener(|panel, _, _, cx| panel.remove_last_ordered_operation(cx))))
                            .child(Button::new("groovy-clear-ordered-operations", "Clear ordered selection")
                                .aria_label("Clear ordered operation selection; running Tasks are unchanged")
                                .disabled(self.ordered_operations.is_empty()).tab_index(0_isize)
                                .on_click(cx.listener(|panel, _, _, cx| panel.clear_ordered_operations(cx))))
                            .child(Button::new("groovy-run-ordered-offline-operations", "Run ordered offline operations")
                                .aria_label("Run ordered offline operations for the current local module through Tasks")
                                .aria_value(format!("{} selected", self.ordered_operations.len()))
                                .disabled(!can_run_ordered).tab_index(0_isize)
                                .on_click(cx.listener(|panel, _, window, cx| panel.run_ordered_offline_operations(window, cx))))))
                    .child(Button::new("groovy-show-dependency-diagram", "Show diagram")
                        .aria_label("Show a bounded diagram of current admitted dependency graph facts")
                        .tab_index(0_isize).toggle_state(self.graph_diagram.is_some())
                        .on_click(cx.listener(|panel, _, _, cx| panel.show_dependency_diagram(cx))))
                    .child(Button::new("groovy-close-dependency-diagram", "Close diagram")
                        .aria_label("Close dependency diagram; retain current tree facts")
                        .tab_index(0_isize).disabled(self.graph_diagram.is_none())
                        .on_click(cx.listener(|panel, _, _, cx| panel.close_dependency_diagram(cx))))
                    .when(show_toolbar, |header| header.child(
                        h_flex()
                            .gap_1()
                            .child(
                                IconButton::new(
                                    "groovy-dependencies-expand-all",
                                    IconName::ExpandVertical,
                                )
                                .aria_label("Expand All")
                                .tooltip(Tooltip::text("Expand All"))
                                .disabled(!can_expand_all)
                                .on_click(cx.listener(
                                    |panel, _, _, cx| {
                                        panel.host.expand_all();
                                        cx.notify();
                                    },
                                )),
                            )
                            .child(
                                IconButton::new(
                                    "groovy-dependencies-collapse-all",
                                    IconName::ListCollapse,
                                )
                                .aria_label("Collapse All")
                                .tooltip(Tooltip::text("Collapse All"))
                                .disabled(!can_collapse_all)
                                .on_click(cx.listener(
                                    |panel, _, _, cx| {
                                        panel.host.collapse_all();
                                        cx.notify();
                                    },
                                )),
                            )
                            .child(
                                ui::IconButton::new(
                                    "groovy-dependencies-refresh",
                                    ui::IconName::RefreshTitle,
                                )
                                .aria_label("Refresh Dependencies")
                                .on_click(cx.listener(|panel, _, _, cx| panel.refresh(cx))),
                            ),
                    )),
            )
            .child(
                h_flex()
                    .id("groovy-build-tool-views")
                    .key_context("GroovyDependenciesPanelControls")
                    .role(Role::Group)
                    .aria_label("Build tool views")
                    .px_2()
                    .gap_1()
                    .children(
                        [
                            BuildToolView::Dependencies,
                            BuildToolView::Gradle,
                            BuildToolView::Maven,
                        ]
                        .into_iter()
                        .map(|view| {
                            Button::new(format!("groovy-build-view-{}", view.label()), view.label())
                                .aria_label(format!("{} view", view.label()))
                                .label_size(LabelSize::Small)
                                .toggle_state(self.view == view)
                                .tab_index(0_isize)
                                .on_click(cx.listener(move |panel, _, window, cx| {
                                    panel.set_view(view, window, cx)
                                }))
                        }),
                    ),
            )
            .child(
                h_flex()
                    .key_context("GroovyDependenciesPanelControls")
                    .px_2().py_1().gap_1()
                    .child(
                        div().id("groovy-graph-fact-search")
                            .role(Role::SearchInput)
                            .aria_label("Search admitted dependency graph facts")
                            .aria_value(self.graph_search.read(cx).text(cx))
                            .aria_description("Case-insensitive literal search in admitted graph labels and facts; does not resolve dependencies")
                            .on_a11y_action(gpui::AccessibleAction::SetValue, {
                                let editor = self.graph_search.downgrade();
                                move |data, window, cx| {
                                    let Some(gpui::accesskit::ActionData::Value(text)) = data else { return; };
                                    if let Some(editor) = editor.upgrade() {
                                        editor.update(cx, |editor, cx| editor.set_text(text.to_string(), window, cx));
                                    }
                                }
                            })
                            .track_focus(&self.graph_search.read(cx).focus_handle(cx))
                            .tab_index(0_isize)
                            .flex_1().min_w_0()
                            .child(self.graph_search.clone())
                    )
                    .child(Button::new("groovy-clear-graph-search", "Clear")
                        .aria_label("Clear dependency graph fact search")
                        .disabled(self.graph_search.read(cx).text(cx).is_empty())
                        .tab_index(0_isize)
                        .on_click(cx.listener(|panel, _, window, cx| panel.clear_graph_fact_search(window, cx))))
            )
            .when(self.show_options && self.view != BuildToolView::Dependencies, |panel| {
                let options = match self.view {
                    BuildToolView::Gradle => vec![BuildViewOption::GroupModules, BuildViewOption::GroupGradleTasks, BuildViewOption::ShowToolbar],
                    BuildToolView::Maven => vec![BuildViewOption::GroupModules, BuildViewOption::ShowBasicMavenPhases, BuildViewOption::ShowMavenArtifactId, BuildViewOption::ShowMavenVersion, BuildViewOption::ShowToolbar],
                    BuildToolView::Dependencies => Vec::new(),
                };
                panel.child(v_flex()
                    .id("groovy-build-view-options-controls")
                    .key_context("GroovyDependenciesPanelControls")
                    .role(Role::Group)
                    .aria_label(format!("{} view presentation options", self.view.label()))
                    .px_2().gap_1()
                    .children(options.into_iter().map(|option| {
                        Button::new(format!("groovy-build-option-{}", option.label()), option.label())
                            .aria_label(option.label())
                            .toggle_state(option.enabled(&self.preferences, self.view))
                            .tab_index(0_isize)
                            .on_click(cx.listener(move |panel, _, _, cx| panel.toggle_option(option, cx)))
                    }))
                    .child(Button::new("groovy-build-option-show-ignored", "Show Ignored")
                        .aria_label("Show Ignored unavailable: provider ignore semantics are not implemented")
                        .disabled(true)))
            })
            .when_some(self.build_phase_notice.clone(), |panel, notice| panel.child(
                div().id("groovy-selected-import-status").role(Role::Status)
                    .aria_label("Selected JVM build import status").aria_value(notice.clone())
                    .aria_live(gpui::accesskit::Live::Polite).px_2().child(notice)
            ))
            .when_some(self.operation_notice.clone(), |panel,notice|panel.child(
                div().id("groovy-build-operation-status").role(Role::Status)
                    .aria_label("JVM build operation admission status").aria_value(notice.clone())
                    .aria_live(gpui::accesskit::Live::Polite).px_2().child(notice)))
            .when_some(self.graph_notice.clone(), |panel, notice| panel.child(
                div().id("groovy-dependency-graph-status").role(Role::Status)
                    .aria_label("Dependency graph status").aria_value(notice.clone())
                    .aria_live(gpui::accesskit::Live::Polite).px_2().child(notice)
            ))
            .when_some(self.preferences_notice.clone(), |panel, notice| panel.child(
                div().id("groovy-build-preferences-status").role(Role::Status)
                    .aria_label("Build view preferences status").aria_value(notice.clone())
                    .aria_live(gpui::accesskit::Live::Polite).px_2().child(notice)
            ))
            .when_some(self.graph_diagram.clone(), |panel, diagram| panel.child(
                div().id("groovy-current-dependency-diagram").role(Role::Group)
                    .aria_label("Current dependency diagram")
                    .aria_description("Read-only full admitted graph; tree facts retain keyboard selection and source navigation. Tree fact/scope/search filters do not change this diagram. Full diagram parity and native interaction remain unverified.")
                    .max_h(px(240.)).overflow_y_scroll().px_2()
                    .child(markdown::MarkdownElement::new(diagram, markdown::MarkdownStyle::themed(markdown::MarkdownFont::Preview, window, cx)))
            ))
            .child(language_tool_tree(
                rows,
                selected,
                status,
                self.host.scroll_handle().clone(),
                move |id, click_count, window, cx| {
                    if let Err(error) = click_panel.update(cx, |panel, cx| {
                        window.focus(&panel.focus_handle, cx);
                        panel.host.select(id);
                        if click_count > 1 {
                            panel.activate_selected(window, cx);
                        }
                        cx.notify();
                    }) {
                        log::debug!("Dependencies panel closed during click: {error:#}");
                    }
                },
                move |id, _, cx| {
                    if let Err(error) = toggle_panel.update(cx, |panel, cx| {
                        panel.host.select(id.clone());
                        panel.host.toggle(&id);
                        cx.notify();
                    }) {
                        log::debug!("Dependencies panel closed during toggle: {error:#}");
                    }
                },
                |_, _, _, _| {},
            ))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    use futures::{FutureExt as _, StreamExt as _, channel::oneshot};
    use gpui::{InputEvent as _, TestAppContext};
    use language::{FakeLspAdapter, Language, LanguageConfig, LanguageMatcher};
    use project::{
        jvm_project::{
            JdtlsLibraryArtifact, JdtlsLibraryFile, JdtlsProject, KotlinWorkspaceProject,
        },
        trusted_worktrees::{self, DbTrustedPaths, PathTrust},
    };
    use settings::WorktreeId;
    use theme::LoadThemes;
    use util::rel_path::RelPath;

    use super::*;

    fn path(worktree_id: WorktreeId, value: &str) -> ProjectPath {
        ProjectPath {
            worktree_id,
            path: Arc::from(RelPath::from_unix_str(value).expect("fixture path")),
        }
    }

    #[gpui::test]
    async fn panel_without_trust_registry_hides_dependency_data(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-no-trust",
            serde_json::json!({"pom.xml": "<project/>"}),
        )
        .await;
        let project = Project::test(fs, ["/groovy-no-trust".as_ref()], cx).await;
        assert!(cx.update(|cx| TrustedWorktrees::try_get_global(cx).is_none()));
        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("untrusted fixture")
                .read(cx)
                .id()
        });
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        let denied = lsp_store
            .update(cx, |store, cx| {
                store.synchronize_local_gradle_project(Some(worktree_id), cx)
            })
            .expect_err("missing trust owner denies execution");
        assert!(denied.to_string().contains("requires project trust"));

        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Groovy panel without trust registry");
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.refresh(cx);
        });
        for maven in [true, false] {
            panel.update(cx, |panel, cx| {
                panel.apply_saved_build_selection(maven, cx);
                assert!(panel.build_phase_token.is_none() && panel.build_phase_waiter.is_none());
                assert!(
                    panel
                        .build_phase_notice
                        .as_deref()
                        .is_some_and(|notice| notice.contains("Trust the project"))
                );
            });
        }
        panel.read_with(cx, |panel, _| {
            assert!(matches!(
                panel.host.status(),
                language_tools::language_tool_tree::LanguageToolTreeStatus::Restricted(_)
            ));
            assert!(panel.host.visible_rows().is_empty());
            assert!(panel.navigation.is_empty());
        });
    }

    #[test]
    fn build_view_preferences_reject_foreign_or_unbounded_state() {
        let state = BuildViewPreferences {
            view: BuildToolView::Maven,
            dock_position: settings::DockPosition::Bottom,
            show_maven_version: false,
            ..Default::default()
        };
        let serialized = serde_json::to_string(&state).expect("bounded preferences");
        assert_eq!(
            BuildViewPreferences::restore(Some(&serialized)).expect("valid preferences"),
            state
        );
        assert_eq!(
            BuildViewPreferences::restore(None).expect("absent preferences"),
            BuildViewPreferences::default()
        );
        let previous = r#"{"version":1,"view":"gradle","group_gradle_tasks":false,"show_maven_artifact_id":false,"show_maven_version":false,"show_gradle_toolbar":false,"show_maven_toolbar":true}"#;
        let migrated =
            BuildViewPreferences::restore(Some(previous)).expect("bounded schema 1 migration");
        assert_eq!(migrated.version, VIEW_STATE_VERSION);
        assert_eq!(migrated.view, BuildToolView::Gradle);
        assert!(!migrated.group_gradle_tasks);
        assert!(!migrated.show_maven_artifact_id);
        assert!(!migrated.show_maven_version);
        assert!(!migrated.show_gradle_toolbar);
        assert!(migrated.show_maven_toolbar);
        assert!(!migrated.group_gradle_modules && !migrated.group_maven_modules);
        assert!(!migrated.show_basic_maven_phases);
        assert_eq!(migrated.dock_position, settings::DockPosition::Right);
        let mut previous_three: serde_json::Value =
            serde_json::from_str(&serialized).expect("schema 3 fixture");
        previous_three["version"] = serde_json::json!(3);
        previous_three
            .as_object_mut()
            .expect("old schema")
            .remove("graph_fact_filter");
        previous_three
            .as_object_mut()
            .expect("schema 3")
            .remove("maven_graph_scope");
        previous_three
            .as_object_mut()
            .expect("record")
            .remove("dock_position");
        let mut expected_three = state.clone();
        expected_three.dock_position = settings::DockPosition::Right;
        assert_eq!(
            BuildViewPreferences::restore(Some(&previous_three.to_string()))
                .expect("schema 3 migration"),
            expected_three
        );
        let mut previous_two: serde_json::Value =
            serde_json::from_str(&serialized).expect("schema 2 fixture");
        previous_two["version"] = serde_json::json!(2);
        previous_two
            .as_object_mut()
            .expect("old schema")
            .remove("graph_fact_filter");
        previous_two
            .as_object_mut()
            .expect("schema 2")
            .remove("maven_graph_scope");
        previous_two
            .as_object_mut()
            .expect("record")
            .remove("show_basic_maven_phases");
        previous_two
            .as_object_mut()
            .expect("record")
            .remove("dock_position");
        let mut expected = state;
        expected.show_basic_maven_phases = false;
        expected.dock_position = settings::DockPosition::Right;
        assert_eq!(
            BuildViewPreferences::restore(Some(&previous_two.to_string()))
                .expect("schema 2 migration"),
            expected
        );
        previous_two["unexpected"] = serde_json::json!(null);
        assert!(BuildViewPreferences::restore(Some(&previous_two.to_string())).is_err());
        let duplicate = previous.replace(
            "\"group_gradle_tasks\":false",
            "\"group_gradle_tasks\":false,\"group_gradle_tasks\":true",
        );
        assert!(BuildViewPreferences::restore(Some(&duplicate)).is_err());
        let incomplete = serialized.replace("\"group_gradle_modules\":false,", "");
        assert!(BuildViewPreferences::restore(Some(&incomplete)).is_err());
        let mut foreign: serde_json::Value = serde_json::from_str(&serialized).expect("state JSON");
        foreign["project_path"] = serde_json::json!("/private/provider/root");
        assert!(BuildViewPreferences::restore(Some(&foreign.to_string())).is_err());
        foreign
            .as_object_mut()
            .expect("state object")
            .remove("project_path");
        foreign["version"] = serde_json::json!(VIEW_STATE_VERSION + 1);
        assert!(BuildViewPreferences::restore(Some(&foreign.to_string())).is_err());
        assert!(
            BuildViewPreferences::restore(Some(&" ".repeat(MAX_VIEW_STATE_BYTES + 1))).is_err()
        );
        assert!(BuildViewPreferences::restore(Some("{broken")).is_err());
        assert!(!serialized.contains("project_path"));
        assert!(!serialized.contains("run_id"));
        assert!(!serialized.contains("generation"));
        assert_ne!(
            workspace_scoped_state_key(VIEW_STATE_KEY, None, Some("first")),
            workspace_scoped_state_key(VIEW_STATE_KEY, None, Some("second"))
        );
        for position in [
            settings::DockPosition::Left,
            settings::DockPosition::Bottom,
            settings::DockPosition::Right,
        ] {
            let preferences = BuildViewPreferences {
                dock_position: position,
                ..Default::default()
            };
            assert_eq!(
                BuildViewPreferences::restore(Some(
                    &serde_json::to_string(&preferences).expect("position preferences")
                ))
                .expect("supported dock position"),
                preferences
            );
        }
        for position in ["top", "floating", "window", "private-canary"] {
            let mut foreign: serde_json::Value =
                serde_json::from_str(&serialized).expect("position fixture");
            foreign["dock_position"] = serde_json::json!(position);
            assert!(BuildViewPreferences::restore(Some(&foreign.to_string())).is_err());
        }
    }

    fn captured_graphs() -> (
        project::jvm_project::MavenDependencyGraph,
        project::jvm_project::GradleDependencyGraph,
    ) {
        let maven = serde_json::from_str(include_str!(
            "../../project/test_data/groovy_jvm_maven/dependency-graph-v2.json"
        ))
        .expect("captured Maven graph");
        let response: serde_json::Value = serde_json::from_str(include_str!(
            "../../project/test_data/groovy_jvm_gradle/dependency-graph-v1.json"
        ))
        .expect("captured Gradle response");
        let gradle = serde_json::from_value(serde_json::json!({
            "projectPath": response["projects"][0]["projectPath"], "name": response["projects"][0]["name"],
            "resolution": serde_json::from_str::<serde_json::Value>(response["projects"][0]["graphJson"].as_str().expect("graph JSON")).expect("resolution graph"),
        })).expect("captured Gradle facts");
        (maven, gradle)
    }

    #[test]
    fn graph_inspection_shortest_paths_preserve_diamonds_cycles_and_unreachable_nodes() {
        let edges = [
            ("root", Some("left")),
            ("root", Some("right")),
            ("left", Some("selected")),
            ("right", Some("selected")),
            ("left", Some("root")),
            ("selected", Some("selected")),
            ("selected", Some("extra")),
            ("extra", Some("selected")),
            ("selected", None),
            ("unreachable", Some("selected")),
        ];
        let (paths, distance) = graph_inspection_edges(
            "root",
            "selected",
            &edges,
            GraphInspectionMode::ShortestRootPaths,
        );
        assert_eq!(distance, Some(2));
        assert_eq!(paths, vec![0, 1, 2, 3]);
        let (neighbours, _) =
            graph_inspection_edges("root", "selected", &edges, GraphInspectionMode::Neighbours);
        assert_eq!(neighbours, vec![2, 3, 5, 6, 7, 8, 9]);
        let (paths, distance) = graph_inspection_edges(
            "root",
            "unreachable",
            &edges,
            GraphInspectionMode::ShortestRootPaths,
        );
        assert_eq!(distance, None);
        assert!(paths.is_empty());
        let (paths, distance) = graph_inspection_edges(
            "root",
            "root",
            &edges,
            GraphInspectionMode::ShortestRootPaths,
        );
        assert_eq!(distance, Some(0));
        assert!(paths.is_empty());
        let duplicates = [("root", Some("selected")), ("root", Some("selected"))];
        assert_eq!(
            graph_inspection_edges(
                "root",
                "selected",
                &duplicates,
                GraphInspectionMode::ShortestRootPaths
            ),
            (vec![0, 1], Some(1))
        );
        let long_nodes = (0..4096).map(|index| index.to_string()).collect::<Vec<_>>();
        let long_edges = long_nodes
            .windows(2)
            .map(|pair| (pair[0].as_str(), Some(pair[1].as_str())))
            .collect::<Vec<_>>();
        let (paths, distance) = graph_inspection_edges(
            "0",
            "4095",
            &long_edges,
            GraphInspectionMode::ShortestRootPaths,
        );
        assert_eq!(distance, Some(4095));
        assert_eq!(paths.len(), 4095);
    }

    fn rendered_diagram(source: &str) -> String {
        let diagram = source
            .split("```mermaid\n")
            .nth(1)
            .expect("generated Mermaid")
            .strip_suffix("```\n")
            .expect("single fenced graph");
        let svg = mermaid_render::render_to_svg(diagram, &mermaid_render::MermaidTheme::default())
            .expect("shared renderer accepts generated graph");
        assert!(svg.contains("<svg"));
        assert!(
            !svg.contains("<script")
                && !svg.contains("onclick=")
                && !svg.contains("<foreignObject")
                && !svg.contains("<a ")
        );
        svg
    }

    #[test]
    fn dependency_diagrams_render_cycles_parallel_edges_and_inert_labels() {
        let nodes = vec![
            ("opaque/root".into(), "project:root:1".into()),
            (
                "same-coordinate-first".into(),
                "org.example:shared:1".into(),
            ),
            (
                "same-coordinate-second".into(),
                "org.example:shared:1".into(),
            ),
            (
                "quoted".into(),
                "Котлин 日本語 \"quoted\" | `; click n0 callback; #35; %%{init:theme}%%".into(),
            ),
            ("unresolved".into(), "Unresolved; cache unavailable".into()),
        ];
        let edges = vec![
            (
                "opaque/root".into(),
                "same-coordinate-first".into(),
                "compile".into(),
                false,
            ),
            (
                "opaque/root".into(),
                "same-coordinate-second".into(),
                "test".into(),
                false,
            ),
            (
                "same-coordinate-first".into(),
                "quoted".into(),
                "constraint".into(),
                true,
            ),
            (
                "same-coordinate-second".into(),
                "quoted".into(),
                "dependency".into(),
                false,
            ),
            (
                "quoted".into(),
                "same-coordinate-first".into(),
                "cycle".into(),
                false,
            ),
            (
                "quoted".into(),
                "unresolved".into(),
                "unresolved".into(),
                false,
            ),
            (
                "quoted".into(),
                "unresolved".into(),
                "parallel".into(),
                false,
            ),
        ];
        let source = dependency_diagram_markdown(&nodes, &edges, "Coverage unknown.")
            .expect("bounded graph");
        assert_eq!(
            source.matches("[\"").count(),
            5,
            "occurrences are not deduplicated"
        );
        assert_eq!(
            source.matches("| n").count(),
            7,
            "parallel/cyclic edges remain explicit"
        );
        assert!(!source.contains("click n0") && !source.contains("%%{init"));
        let svg = rendered_diagram(&source);
        assert!(
            svg.contains("Котлин") && svg.contains("日本語"),
            "Unicode labels must survive the shared renderer"
        );
    }

    #[test]
    fn dependency_diagrams_refuse_limits_foreign_references_and_unsupported_labels() {
        let nodes = (0..64)
            .map(|index| (index.to_string(), "bounded".into()))
            .collect::<Vec<_>>();
        let edges = (0..128)
            .map(|_| ("0".into(), "63".into(), "bounded".into(), false))
            .collect::<Vec<_>>();
        assert!(dependency_diagram_markdown(&nodes, &edges, "").is_ok());
        let mut oversized = nodes.clone();
        oversized.push(("65".into(), "bounded".into()));
        assert!(dependency_diagram_markdown(&oversized, &edges, "").is_err());
        let mut too_many_edges = edges.clone();
        too_many_edges.push(edges[0].clone());
        assert!(dependency_diagram_markdown(&nodes, &too_many_edges, "").is_err());
        for label in [
            "X".repeat(257),
            "new\nline".into(),
            "<script>failure</script>".into(),
        ] {
            assert!(dependency_diagram_markdown(&[("0".into(), label)], &[], "").is_err());
        }
        let duplicate = vec![("0".into(), "first".into()), ("0".into(), "second".into())];
        assert!(dependency_diagram_markdown(&duplicate, &[], "").is_err());
        for edge in [
            ("foreign".into(), "0".into(), "edge".into(), false),
            ("0".into(), "foreign".into(), "edge".into(), false),
        ] {
            assert!(dependency_diagram_markdown(&nodes, &[edge], "").is_err());
        }
        let byte_budget = edges
            .iter()
            .map(|(from, to, _, style)| (from.clone(), to.clone(), "X".repeat(256), *style))
            .collect::<Vec<_>>();
        assert!(dependency_diagram_markdown(&nodes, &byte_budget, "").is_err());
    }

    #[test]
    fn captured_graph_inspection_retains_maven_occurrences_and_gradle_components() {
        let (maven, gradle) = captured_graphs();
        let maven_edges = maven
            .edges
            .iter()
            .map(|edge| (edge.from_node.as_str(), Some(edge.target_node.as_str())))
            .collect::<Vec<_>>();
        for edge in &maven.edges {
            let (paths, distance) = graph_inspection_edges(
                &maven.root_node,
                &edge.target_node,
                &maven_edges,
                GraphInspectionMode::ShortestRootPaths,
            );
            assert!(distance.is_some());
            assert!(!paths.is_empty());
            assert!(
                paths
                    .iter()
                    .any(|index| maven.edges[*index].target_node == edge.target_node)
            );
        }
        assert!(
            maven
                .edges
                .iter()
                .any(|edge| edge.target_node != edge.selected_node),
            "captured mediation stays separate from ancestry"
        );
        let gradle_edges = gradle
            .resolution
            .edges
            .iter()
            .map(|edge| (edge.from.as_str(), edge.selected.as_deref()))
            .collect::<Vec<_>>();
        for component in &gradle.resolution.components {
            let (neighbours, _) = graph_inspection_edges(
                &gradle.resolution.root_component,
                &component.id,
                &gradle_edges,
                GraphInspectionMode::Neighbours,
            );
            assert!(neighbours.iter().all(|index| {
                let edge = &gradle.resolution.edges[*index];
                edge.from == component.id || edge.selected.as_deref() == Some(component.id.as_str())
            }));
        }
        let constraint = gradle
            .resolution
            .edges
            .iter()
            .find(|edge| edge.constraint && edge.selected.is_some())
            .expect("captured constraint");
        let selected = constraint
            .selected
            .as_deref()
            .expect("constraint selected version");
        let relations =
            gradle_inspection_relations(&gradle, GraphInspectionMode::ShortestRootPaths);
        let (paths, distance) = graph_inspection_edges(
            &gradle.resolution.root_component,
            selected,
            &relations,
            GraphInspectionMode::ShortestRootPaths,
        );
        assert_eq!(
            distance,
            Some(2),
            "version constraint cannot shorten actual dependency inclusion"
        );
        assert!(
            paths
                .iter()
                .all(|index| !gradle.resolution.edges[*index].constraint)
        );
        let neighbours = gradle_inspection_relations(&gradle, GraphInspectionMode::Neighbours);
        assert!(
            neighbours
                .iter()
                .any(|(_, target)| *target == Some(selected))
        );
        let mediated = [("root", Some("requested")), ("unrelated", Some("selected"))];
        assert_eq!(
            graph_inspection_edges(
                "root",
                "selected",
                &mediated,
                GraphInspectionMode::ShortestRootPaths
            ),
            (vec![], None)
        );
    }

    #[test]
    fn maven_graph_scopes_preserve_exact_occurrences_and_coverage() {
        let module = LanguageToolNodeId("scope-fixture".into());
        let (mut graph, _) = captured_graphs();
        for (scope, expected_edges) in [
            (MavenGraphScope::Compile, 12),
            (MavenGraphScope::Provided, 1),
            (MavenGraphScope::Test, 1),
            (MavenGraphScope::Runtime, 0),
        ] {
            let projected =
                maven_graph_node_filtered(&module, Some(&graph), GraphFactFilter::All, scope);
            let edges = &projected
                .children
                .iter()
                .find(|row| row.label == "Dependency edges")
                .expect("captured scope edges")
                .children;
            assert_eq!(edges.len(), expected_edges);
        }
        let scopes = [
            "compile",
            "runtime",
            "test",
            "provided",
            "system",
            "import",
            "",
            "custom-α",
        ];
        for (edge, scope) in graph.edges.iter_mut().zip(scopes.iter().cycle()) {
            edge.scope = (*scope).into();
        }
        graph
            .unavailable_descriptors
            .push(project::jvm_project::MavenUnavailableDescriptor {
                component: graph
                    .edges
                    .get(2)
                    .expect("test-scope edge")
                    .component
                    .clone(),
                problem: "offline-metadata-unavailable-check-cache-and-refresh".into(),
            });
        graph.collection_complete = false;
        graph.structural_coverage_known = false;
        let mut scope = MavenGraphScope::All;
        for _ in 0..8 {
            for filter in [
                GraphFactFilter::All,
                GraphFactFilter::Direct,
                GraphFactFilter::Conflicts,
                GraphFactFilter::Unavailable,
            ] {
                let root = maven_graph_node_filtered(&module, Some(&graph), filter, scope);
                let occurrences = &root
                    .children
                    .iter()
                    .find(|row| row.label == "Occurrences")
                    .expect("occurrences")
                    .children;
                let edges = &root
                    .children
                    .iter()
                    .find(|row| row.label == "Dependency edges")
                    .expect("edges")
                    .children;
                let expected = graph
                    .edges
                    .iter()
                    .filter(|edge| {
                        scope.matches(&edge.scope)
                            && match filter {
                                GraphFactFilter::All => true,
                                GraphFactFilter::Direct => edge.direct,
                                GraphFactFilter::Conflicts => edge.omitted,
                                GraphFactFilter::Unavailable => {
                                    graph.unavailable_descriptors.iter().any(|descriptor| {
                                        descriptor.component == edge.component
                                            || descriptor.component == edge.selected
                                    })
                                }
                            }
                    })
                    .collect::<Vec<_>>();
                assert_eq!(edges.len(), expected.len());
                for edge in expected {
                    let identity =
                        graph_fact_id(&module, "maven-edge", &[&edge.from_node, &edge.target_node]);
                    let row = edges
                        .iter()
                        .find(|row| row.id.0 == format!("{}:occurrence:0", identity.0))
                        .expect("retained distinct provider edge occurrence");
                    for identity in [&edge.from_node, &edge.target_node, &edge.selected_node] {
                        let identity = graph_fact_id(&module, "maven-node", &[identity]);
                        assert!(
                            occurrences
                                .iter()
                                .any(|row| row.id.0 == format!("{}:occurrence:0", identity.0))
                        );
                    }
                    assert!(
                        row.accessibility_label
                            .contains(&format!("scope {};", edge.scope))
                    );
                }
                assert!(occurrences.iter().any(|row| row.id
                    == LanguageToolNodeId(format!(
                        "{}:occurrence:0",
                        graph_fact_id(&module, "maven-node", &[&graph.root_node]).0
                    ))));
                assert!(
                    root.accessibility_label
                        .contains(&format!("Maven scope {};", scope.label()))
                );
                assert!(
                    root.accessibility_label
                        .contains("source origins and artifacts unavailable")
                );
                assert!(
                    root.accessibility_label
                        .contains("collection complete false")
                );
                let mut searched = vec![root];
                search_project_graph_rows(&mut searched, "no-match-scope-fixture");
                assert_eq!(searched.len(), 1);
                assert!(searched[0].children.is_empty());
            }
            scope = scope.next();
        }
        assert_eq!(scope, MavenGraphScope::All);
        assert!(MavenGraphScope::Other.matches(""));
        assert!(MavenGraphScope::Other.matches("custom-α"));
        assert!(MavenGraphScope::Other.matches("COMPILE"));
        assert!(!MavenGraphScope::Compile.matches("COMPILE"));
        assert!(!MavenGraphScope::Other.matches("compile"));
        graph.edges.clear();
        let empty = maven_graph_node_filtered(
            &module,
            Some(&graph),
            GraphFactFilter::All,
            MavenGraphScope::Test,
        );
        assert!(empty.accessibility_label.contains("0 retained edges"));
        assert_eq!(
            empty
                .children
                .iter()
                .find(|row| row.label == "Occurrences")
                .expect("root occurrence")
                .children
                .len(),
            1
        );
        assert!(
            maven_graph_node_filtered(
                &module,
                None,
                GraphFactFilter::All,
                MavenGraphScope::Compile
            )
            .accessibility_label
            .contains("Not collected")
        );
    }

    #[test]
    fn maven_scope_preferences_migrate_strict_schema_five() {
        let state = BuildViewPreferences {
            maven_graph_scope: MavenGraphScope::Test,
            graph_fact_filter: GraphFactFilter::Conflicts,
            dock_position: settings::DockPosition::Bottom,
            ..Default::default()
        };
        let serialized = serde_json::to_value(&state).expect("scope preferences");
        assert_eq!(
            BuildViewPreferences::restore(Some(&serialized.to_string())).expect("scope restore"),
            state
        );
        let mut previous = serialized.clone();
        previous["version"] = serde_json::json!(5);
        previous
            .as_object_mut()
            .expect("schema 5")
            .remove("maven_graph_scope");
        let migrated =
            BuildViewPreferences::restore(Some(&previous.to_string())).expect("schema 5 migration");
        assert_eq!(migrated.maven_graph_scope, MavenGraphScope::All);
        assert_eq!(migrated.graph_fact_filter, GraphFactFilter::Conflicts);
        assert_eq!(migrated.dock_position, state.dock_position);
        previous["maven_graph_scope"] = serde_json::json!("test");
        assert!(BuildViewPreferences::restore(Some(&previous.to_string())).is_err());
        let mut invalid = serialized;
        invalid["maven_graph_scope"] = serde_json::json!("private-canary");
        assert!(BuildViewPreferences::restore(Some(&invalid.to_string())).is_err());
    }

    #[test]
    fn graph_fact_filters_preserve_provider_semantics_and_stable_rows() {
        let module = LanguageToolNodeId("graph-filter-fixture".into());
        let (mut maven, mut gradle) = captured_graphs();
        let conflicted = gradle
            .resolution
            .components
            .iter()
            .find(|component| component.conflict_resolution)
            .expect("captured conflict");
        let duplicate = gradle
            .resolution
            .edges
            .iter()
            .find(|edge| edge.selected.as_deref() == Some(conflicted.id.as_str()))
            .expect("captured conflict edge")
            .clone();
        gradle.resolution.edges.push(duplicate);

        let rows = |root: LanguageToolNode| {
            let mut host = LanguageToolTreeHost::default();
            host.replace_snapshot(LanguageToolSnapshot {
                roots: vec![root],
                status: LanguageToolProviderStatus::Current,
            });
            host.expand_all();
            host.visible_rows().to_vec()
        };
        for provider in [GraphFixtureProvider::Maven, GraphFixtureProvider::Gradle] {
            let build = |filter| match provider {
                GraphFixtureProvider::Maven => {
                    maven_graph_node_filtered(&module, Some(&maven), filter, MavenGraphScope::All)
                }
                GraphFixtureProvider::Gradle => {
                    gradle_graph_node_filtered(&module, Some(&gradle), filter)
                }
            };
            let all = rows(build(GraphFactFilter::All));
            let all_ids = all.iter().map(|row| row.id.clone()).collect::<HashSet<_>>();
            let direct = rows(build(GraphFactFilter::Direct));
            let conflicts = rows(build(GraphFactFilter::Conflicts));
            let unavailable = rows(build(GraphFactFilter::Unavailable));
            assert!(direct.len() < all.len());
            assert!(conflicts.len() < all.len());
            assert!(unavailable.len() < all.len());
            for filtered in [&direct, &conflicts, &unavailable] {
                assert!(filtered.iter().all(|row| all_ids.contains(&row.id)));
                assert!(
                    filtered
                        .first()
                        .expect("root")
                        .accessibility_label
                        .contains("source origins and artifacts unavailable")
                );
            }
            let edge_rows = |values: &[language_tool_tree::VisibleLanguageToolRow]| {
                values
                    .iter()
                    .filter(|row| row.label.contains(" → "))
                    .map(|row| row.accessibility_label.clone())
                    .collect::<Vec<_>>()
            };
            match provider {
                GraphFixtureProvider::Maven => {
                    assert!(!edge_rows(&direct).is_empty());
                    assert!(
                        edge_rows(&direct)
                            .iter()
                            .all(|label| label.contains("Direct dependency"))
                    );
                    assert!(!edge_rows(&conflicts).is_empty());
                    assert!(
                        edge_rows(&conflicts)
                            .iter()
                            .all(|label| label.contains("omitted by mediation true"))
                    );
                    assert!(
                        !unavailable
                            .iter()
                            .any(|row| row.accessibility_label.contains("Descriptor unavailable"))
                    );
                    assert!(
                        unavailable
                            .first()
                            .expect("Maven root")
                            .accessibility_label
                            .contains("exact cycle occurrences unknown")
                    );
                }
                GraphFixtureProvider::Gradle => {
                    assert!(!edge_rows(&direct).is_empty());
                    assert!(
                        edge_rows(&direct)
                            .iter()
                            .all(|label| label.contains("Root edge"))
                    );
                    assert!(
                        conflicts.iter().any(|row| row
                            .accessibility_label
                            .contains("conflict resolution true"))
                    );
                    assert!(!edge_rows(&unavailable).is_empty());
                    assert!(
                        edge_rows(&unavailable)
                            .iter()
                            .all(|label| label.contains("→ Unresolved"))
                    );
                }
            }
            let mut host = LanguageToolTreeHost::default();
            host.replace_snapshot(LanguageToolSnapshot {
                roots: vec![build(GraphFactFilter::All)],
                status: LanguageToolProviderStatus::Current,
            });
            host.expand_all();
            let removed = all
                .iter()
                .find(|row| !conflicts.iter().any(|other| other.id == row.id))
                .expect("filtered row");
            host.select(removed.id.clone());
            host.replace_snapshot(LanguageToolSnapshot {
                roots: vec![build(GraphFactFilter::Conflicts)],
                status: LanguageToolProviderStatus::Current,
            });
            assert_ne!(host.selected(), Some(&removed.id));
            host.replace_snapshot(LanguageToolSnapshot {
                roots: vec![],
                status: LanguageToolProviderStatus::Restricted("Trust revoked".into()),
            });
            assert!(host.selected().is_none());
        }
        let expected = rows(gradle_graph_node_filtered(
            &module,
            Some(&gradle),
            GraphFactFilter::Conflicts,
        ))
        .iter()
        .map(|row| row.id.clone())
        .collect::<HashSet<_>>();
        gradle.resolution.components.reverse();
        gradle.resolution.edges.reverse();
        let changed = rows(gradle_graph_node_filtered(
            &module,
            Some(&gradle),
            GraphFactFilter::Conflicts,
        ))
        .iter()
        .map(|row| row.id.clone())
        .collect::<HashSet<_>>();
        assert_eq!(expected, changed);
        maven
            .unavailable_descriptors
            .push(project::jvm_project::MavenUnavailableDescriptor {
                component: maven
                    .nodes
                    .iter()
                    .find(|node| node.id != maven.root_node)
                    .expect("dependency occurrence")
                    .component
                    .clone(),
                problem: "offline-metadata-unavailable-check-cache-and-refresh".into(),
            });
        maven.collection_complete = false;
        maven.structural_coverage_known = false;
        maven.problem = Some("offline-metadata-unavailable-check-cache-and-refresh".into());
        let missing = rows(maven_graph_node_filtered(
            &module,
            Some(&maven),
            GraphFactFilter::Unavailable,
            MavenGraphScope::All,
        ));
        assert!(
            missing
                .iter()
                .any(|row| row.accessibility_label.contains("Descriptor unavailable"))
        );
        assert!(
            missing
                .first()
                .expect("partial metadata")
                .accessibility_label
                .contains("collection complete false")
        );
        maven.edges.retain(|edge| !edge.omitted);
        let no_conflicts = rows(maven_graph_node_filtered(
            &module,
            Some(&maven),
            GraphFactFilter::Conflicts,
            MavenGraphScope::All,
        ));
        assert!(no_conflicts.iter().all(|row| !row.label.contains(" → ")));
        assert!(
            no_conflicts
                .first()
                .expect("empty filter root")
                .accessibility_label
                .contains("0 retained edges")
        );
        assert!(
            maven_graph_node_filtered(
                &module,
                None,
                GraphFactFilter::Unavailable,
                MavenGraphScope::All
            )
            .accessibility_label
            .contains("Not collected")
        );
    }

    #[test]
    fn graph_fact_search_is_bounded_and_preserves_ancestors_and_identities() {
        let module = LanguageToolNodeId("graph-search-fixture".into());
        let (maven, gradle) = captured_graphs();
        let flat = |roots: Vec<LanguageToolNode>| {
            let mut host = LanguageToolTreeHost::default();
            host.replace_snapshot(LanguageToolSnapshot {
                roots,
                status: LanguageToolProviderStatus::Current,
            });
            host.expand_all();
            host.visible_rows().to_vec()
        };
        for graph in [
            maven_graph_node(&module, Some(&maven)),
            gradle_graph_node(&module, Some(&gradle)),
        ] {
            let all = flat(vec![graph.clone()]);
            let query = graph_search_query("  SHARED  ")
                .expect("literal")
                .expect("nonempty");
            let mut roots = vec![LanguageToolNode {
                id: module.clone(),
                label: "Build tasks and manifest".into(),
                secondary_label: None,
                accessibility_label: "Build module".into(),
                icon: None,
                children: vec![
                    graph.clone(),
                    LanguageToolNode {
                        id: LanguageToolNodeId("task:test".into()),
                        label: "test".into(),
                        secondary_label: None,
                        accessibility_label: "Test task".into(),
                        icon: None,
                        children: vec![],
                        enabled: false,
                        activation_label: None,
                    },
                ],
                enabled: true,
                activation_label: Some("Open manifest".into()),
            }];
            search_project_graph_rows(&mut roots, &query);
            let matched = flat(roots);
            assert!(matched.iter().any(|row| row.label.contains("shared")));
            assert!(matched.iter().any(|row| row.id.0 == "task:test"));
            assert!(
                matched
                    .iter()
                    .filter(|row| row.id.0.starts_with("graph:"))
                    .all(|row| all.iter().any(|original| original.id == row.id))
            );
            assert!(matched.len() < all.len() + 2);
            assert!(matched.iter().any(|row| {
                row.id == graph.id
                    && row
                        .accessibility_label
                        .contains("Matching admitted graph facts")
            }));
            let mut no_match = vec![graph.clone()];
            search_project_graph_rows(&mut no_match, "never-present-private-canary");
            assert_eq!(flat(no_match.clone()).len(), 1);
            assert!(
                no_match
                    .first()
                    .expect("root")
                    .accessibility_label
                    .contains("No matching admitted graph facts")
            );
            assert!(
                !no_match
                    .first()
                    .expect("root")
                    .accessibility_label
                    .contains("private-canary")
            );
            let mut invalid = vec![graph];
            suppress_invalid_graph_search(&mut invalid);
            assert_eq!(flat(invalid.clone()).len(), 1);
            assert!(
                invalid
                    .first()
                    .expect("root")
                    .accessibility_label
                    .contains("Invalid graph search")
            );
        }
        let mut unicode = vec![graph_fact_row(
            graph_fact_id(&module, "unicode", &[]),
            "Dependency Härt-α 📦".into(),
            "Partial scope metadata".into(),
            vec![],
        )];
        search_project_graph_rows(
            &mut unicode,
            &graph_search_query("HÄRT-Α 📦")
                .expect("Unicode")
                .expect("query"),
        );
        assert!(
            unicode
                .first()
                .expect("Unicode row")
                .accessibility_label
                .contains("Matching admitted graph facts")
        );
        let mut unknown = vec![maven_graph_node(&module, None)];
        search_project_graph_rows(&mut unknown, "leaf");
        assert!(
            unknown
                .first()
                .expect("uncollected root")
                .accessibility_label
                .contains("Not collected")
        );
        assert_eq!(graph_search_query(" ").expect("empty"), None);
        assert!(graph_search_query(&"α".repeat(MAX_GRAPH_SEARCH_BYTES / 2)).is_ok());
        for invalid in [
            "α".repeat(MAX_GRAPH_SEARCH_BYTES / 2 + 1),
            "private-canary\n".into(),
            "private-canary\0".into(),
        ] {
            let error = graph_search_query(&invalid)
                .expect_err("bounded search")
                .to_string();
            assert!(!error.contains("private-canary"));
        }
    }

    #[test]
    fn graph_filter_preferences_migrate_strict_previous_schema() {
        let state = BuildViewPreferences {
            dock_position: settings::DockPosition::Bottom,
            graph_fact_filter: GraphFactFilter::Conflicts,
            ..Default::default()
        };
        assert_eq!(
            BuildViewPreferences::restore(Some(
                &serde_json::to_string(&state).expect("current prefs")
            ))
            .expect("restored filter"),
            state
        );
        let mut previous = serde_json::to_value(&state).expect("prefs");
        previous["version"] = serde_json::json!(4);
        previous
            .as_object_mut()
            .expect("old prefs")
            .remove("graph_fact_filter");
        previous
            .as_object_mut()
            .expect("schema 4")
            .remove("maven_graph_scope");
        let migrated =
            BuildViewPreferences::restore(Some(&previous.to_string())).expect("schema 4 migration");
        assert_eq!(migrated.dock_position, settings::DockPosition::Bottom);
        assert_eq!(migrated.graph_fact_filter, GraphFactFilter::All);
        assert_eq!(migrated.version, VIEW_STATE_VERSION);
        previous["graph_fact_filter"] = serde_json::json!("conflicts");
        assert!(BuildViewPreferences::restore(Some(&previous.to_string())).is_err());
        let mut bad = serde_json::to_value(&state).expect("prefs");
        bad["graph_fact_filter"] = serde_json::json!("private-canary");
        assert!(BuildViewPreferences::restore(Some(&bad.to_string())).is_err());
    }

    #[test]
    fn admitted_graph_rows_preserve_provider_facts_without_recursive_expansion() {
        let module = LanguageToolNodeId("fixture-module".into());
        let (maven, gradle) = captured_graphs();
        for (node, expected) in [
            (
                maven_graph_node(&module, Some(&maven)),
                4 + maven.nodes.len() + maven.edges.len() + maven.unavailable_descriptors.len(),
            ),
            (
                gradle_graph_node(&module, Some(&gradle)),
                3 + gradle.resolution.components.len() + gradle.resolution.edges.len(),
            ),
        ] {
            let mut pending = vec![&node];
            while let Some(row) = pending.pop() {
                assert!(!row.enabled && row.activation_label.is_none());
                pending.extend(row.children.iter());
            }
            let mut host = LanguageToolTreeHost::default();
            host.replace_snapshot(LanguageToolSnapshot {
                roots: vec![node],
                status: LanguageToolProviderStatus::Partial("Captured partial coverage".into()),
            });
            host.expand_all();
            assert_eq!(host.visible_rows().len(), expected);
            let rows = host.visible_rows();
            assert!(
                rows.iter()
                    .all(|row| row.level <= 3 && !row.accessibility_label.is_empty())
            );
            assert_eq!(
                rows.iter().map(|row| &row.id).collect::<HashSet<_>>().len(),
                expected
            );
            host.select_last();
            let selected = host.selected().cloned();
            host.mark_dirty();
            assert_eq!(host.selected().cloned(), selected);
        }
        let maven_rows = maven_graph_node(&module, Some(&maven));
        let edges = maven_rows
            .children
            .iter()
            .find(|row| row.label == "Dependency edges")
            .expect("Maven edges");
        assert!(edges.children.iter().any(|row| {
            row.accessibility_label
                .contains("omitted by mediation true")
        }));
        assert!(
            edges
                .children
                .iter()
                .any(|row| row.accessibility_label.contains("scope compile"))
        );
        assert!(
            maven_rows
                .accessibility_label
                .contains("exact cycle occurrences unknown")
        );
        let gradle_rows = gradle_graph_node(&module, Some(&gradle));
        let components = gradle_rows
            .children
            .iter()
            .find(|row| row.label == "Components")
            .expect("Gradle components");
        assert!(
            components
                .children
                .iter()
                .any(|row| row.accessibility_label.contains("conflict resolution true"))
        );
        assert!(components.children.iter().any(|row| {
            row.accessibility_label
                .contains("composite substitution true")
        }));
        let edges = gradle_rows
            .children
            .iter()
            .find(|row| row.label == "Dependency edges")
            .expect("Gradle edges");
        assert!(
            edges
                .children
                .iter()
                .any(|row| row.label.ends_with("→ Unresolved"))
        );
        assert!(
            edges
                .children
                .iter()
                .any(|row| row.accessibility_label.contains("constraint true"))
        );
        assert!(
            gradle_rows
                .accessibility_label
                .contains("resolution complete false")
        );
    }

    #[test]
    fn graph_rows_keep_selection_across_reorder_and_distinguish_occurrences() {
        let module = LanguageToolNodeId("module:with:delimiters".into());
        let (mut maven, mut gradle) = captured_graphs();
        for provider in [GraphFixtureProvider::Maven, GraphFixtureProvider::Gradle] {
            let original = match provider {
                GraphFixtureProvider::Maven => maven_graph_node(&module, Some(&maven)),
                GraphFixtureProvider::Gradle => gradle_graph_node(&module, Some(&gradle)),
            };
            let mut host = LanguageToolTreeHost::default();
            host.replace_snapshot(LanguageToolSnapshot {
                roots: vec![original],
                status: LanguageToolProviderStatus::Current,
            });
            host.expand_all();
            host.select_last();
            let selected = host.selected().cloned();
            let identities = host
                .visible_rows()
                .iter()
                .map(|row| row.id.clone())
                .collect::<HashSet<_>>();
            maven.nodes.reverse();
            maven.components.reverse();
            maven.edges.reverse();
            gradle.resolution.components.reverse();
            gradle.resolution.edges.reverse();
            let changed = match provider {
                GraphFixtureProvider::Maven => maven_graph_node(&module, Some(&maven)),
                GraphFixtureProvider::Gradle => gradle_graph_node(&module, Some(&gradle)),
            };
            host.replace_snapshot(LanguageToolSnapshot {
                roots: vec![changed],
                status: LanguageToolProviderStatus::Current,
            });
            assert_eq!(host.selected().cloned(), selected);
            assert_eq!(
                host.visible_rows()
                    .iter()
                    .map(|row| row.id.clone())
                    .collect::<HashSet<_>>(),
                identities
            );
            host.replace_snapshot(LanguageToolSnapshot {
                roots: vec![],
                status: LanguageToolProviderStatus::Restricted("Trust revoked".into()),
            });
            assert!(host.selected().is_none());
        }
        let duplicate = gradle.resolution.edges.first().expect("edge").clone();
        gradle.resolution.edges.push(duplicate);
        let mut host = LanguageToolTreeHost::default();
        host.replace_snapshot(LanguageToolSnapshot {
            roots: vec![gradle_graph_node(&module, Some(&gradle))],
            status: LanguageToolProviderStatus::Current,
        });
        host.expand_all();
        assert_eq!(
            host.visible_rows().len(),
            host.visible_rows()
                .iter()
                .map(|row| &row.id)
                .collect::<HashSet<_>>()
                .len()
        );
        assert!(
            maven_graph_node(&module, None)
                .accessibility_label
                .contains("Not collected")
        );
        assert!(
            gradle_graph_node(&module, None)
                .accessibility_label
                .contains("Not collected")
        );
    }

    #[test]
    fn declared_configurations_preserve_identity_without_module_aliases() {
        use project::jvm_project::GradleConfiguration;

        let module_id = LanguageToolNodeId("jdtls:1:application:Gradle".into());
        let mut gradle = GradleProject {
            dependency_graph: None,
            build_root: path(WorktreeId::from_usize(1), ""),
            project_path: ":application".into(),
            tasks: Vec::new(),
            source_sets_known: false,
            source_sets: Vec::new(),
            configurations_known: false,
            configurations: Vec::new(),
        };
        let unknown = gradle_configurations_node(&module_id, &gradle);
        gradle.configurations_known = true;
        let empty = gradle_configurations_node(&module_id, &gradle);
        assert_eq!(unknown.id, empty.id);
        assert!(
            unknown
                .accessibility_label
                .contains("declarations unavailable")
        );
        assert!(
            empty
                .accessibility_label
                .contains("No declared configurations")
        );
        gradle.configurations = vec![
            GradleConfiguration {
                name: "Gradle".into(),
                description: None,
                resolvable: false,
                consumable: true,
                extends_from: Vec::new(),
            },
            GradleConfiguration {
                name: "runtime α".into(),
                description: None,
                resolvable: true,
                consumable: false,
                extends_from: vec!["Gradle".into()],
            },
        ];
        let original = gradle_configurations_node(&module_id, &gradle);
        assert_eq!(original.id, empty.id);
        let alias = LanguageToolNodeId(format!("{}:gradle-configuration:6:Gradle", module_id.0));
        assert_ne!(original.children[0].id, alias);
        let other = gradle_configurations_node(&alias, &gradle);
        assert_ne!(original.children[0].id, other.children[0].id);
        let identities = original
            .children
            .iter()
            .map(|row| row.id.clone())
            .collect::<HashSet<_>>();
        gradle.configurations.reverse();
        gradle.configurations[0].description = Some("Updated declaration".into());
        gradle.configurations[0].consumable = true;
        let updated = gradle_configurations_node(&module_id, &gradle);
        assert_eq!(
            identities,
            updated
                .children
                .iter()
                .map(|row| row.id.clone())
                .collect::<HashSet<_>>()
        );
        for row in original
            .children
            .iter()
            .chain(original.children.iter().flat_map(|row| &row.children))
        {
            assert!(!row.enabled && row.activation_label.is_none());
            assert!(
                row.accessibility_label
                    .contains("resolved dependencies unavailable")
            );
        }
    }

    #[test]
    fn module_groups_preserve_composite_builds_and_module_navigation() {
        use project::jvm_project::{GradleProject, MavenProject};
        let worktree_id = WorktreeId::from_usize(1);
        let modules = [
            ("libraries/api", "", ":api"),
            ("libraries/implementation", "", ":implementation"),
            ("included/api", "included", ":api"),
            ("modules/service", "", ":services:service"),
        ];
        let snapshot = JdtlsProjectSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(1),
            worktree_id,
            generation: 2,
            partial: false,
            projects: modules
                .into_iter()
                .map(|(directory, build, project_path)| JdtlsProject {
                    module_root: path(worktree_id, directory),
                    membership: JdtlsProjectMembership::Unknown,
                    imported_build_kind: project::jvm_project::JdtlsImportedBuildKind::Unknown,
                    source_roots: Vec::new(),
                    project_references: Vec::new(),
                    library_entries: 0,
                    library_files: Vec::new(),
                    library_artifacts: Vec::new(),
                    project_vm_known: false,
                    project_vm_java_major: None,
                    server_launch_java_major: None,
                    java_source_level: None,
                    java_target_level: None,
                    gradle: Some(GradleProject {
                        dependency_graph: None,
                        build_root: path(worktree_id, build),
                        project_path: project_path.into(),
                        tasks: Vec::new(),
                        source_sets_known: false,
                        source_sets: Vec::new(),
                        configurations_known: false,
                        configurations: Vec::new(),
                    }),
                    maven: Some(MavenProject {
                        dependency_graph: None,
                        pom: path(worktree_id, &format!("{directory}/pom.xml")),
                        modules: Vec::new(),
                        group_id: "evidence".into(),
                        artifact_id: directory.replace('/', "-"),
                        version: "1.0".into(),
                        packaging: "jar".into(),
                        profiles: BTreeMap::new(),
                        profile_definitions_known: false,
                        profile_definitions: Vec::new(),
                        reactor_position: None,
                        reactor_order_problem: None,
                        plugins: Vec::new(),
                        lifecycles: Vec::new(),
                    }),
                })
                .collect(),
        };
        for view in [BuildToolView::Gradle, BuildToolView::Maven] {
            let flat = BuildViewPreferences::default();
            let grouped = BuildViewPreferences {
                group_gradle_modules: true,
                group_maven_modules: true,
                ..Default::default()
            };
            let (flat_rows, flat_navigation) = project_rows_for_view(&snapshot, view, &flat);
            let (grouped_rows, grouped_navigation) =
                project_rows_for_view(&snapshot, view, &grouped);
            assert_eq!(flat_navigation, grouped_navigation);
            assert_eq!(grouped_rows.len(), 3);
            assert!(
                grouped_rows
                    .iter()
                    .all(|group| !group.enabled && group.activation_label.is_none())
            );
            let original_ids = flat_rows
                .iter()
                .map(|row| row.id.clone())
                .collect::<HashSet<_>>();
            let grouped_ids = grouped_rows
                .iter()
                .flat_map(|group| &group.children)
                .map(|row| row.id.clone())
                .collect::<HashSet<_>>();
            assert_eq!(original_ids, grouped_ids);
            assert!(
                grouped_rows
                    .iter()
                    .all(|group| !grouped_navigation.contains_key(&group.id))
            );
            if view == BuildToolView::Gradle {
                assert_eq!(
                    grouped_rows
                        .iter()
                        .filter(|group| group.label.ends_with("· :"))
                        .count(),
                    2,
                    "composite builds with the same logical parent stay separate"
                );
                assert!(
                    grouped_rows
                        .iter()
                        .any(|group| group.label == "Project root · :services")
                );
            } else {
                assert!(grouped_rows.iter().any(
                    |group| group.label == "Directory: libraries" && group.children.len() == 2
                ));
            }
        }
    }

    #[test]
    fn basic_maven_phases_preserve_provider_order_and_identities() {
        use project::jvm_project::{
            MavenGoal, MavenGoalParameter, MavenLifecycle, MavenPlugin, MavenProject,
        };

        let module_id = LanguageToolNodeId("module:application".into());
        let mut provider = MavenProject {
            dependency_graph: None,
            pom: path(WorktreeId::from_usize(1), "application/pom.xml"),
            modules: Vec::new(),
            group_id: "evidence".into(),
            artifact_id: "application".into(),
            version: "1.0".into(),
            packaging: "jar".into(),
            profiles: BTreeMap::from([("cached".into(), true)]),
            profile_definitions_known: true,
            profile_definitions: Vec::new(),
            reactor_position: Some(0),
            reactor_order_problem: None,
            plugins: vec![MavenPlugin {
                group_id: "org.apache.maven.plugins".into(),
                artifact_id: "maven-compiler-plugin".into(),
                version: Some("3.14.0".into()),
                managed_only: false,
                goals_known: true,
                goal_prefix: Some("compiler".into()),
                goals: vec![MavenGoal {
                    name: "testCompile".into(),
                    phase: Some("test-compile".into()),
                    description: None,
                    parameters_known: false,
                    parameters: Vec::new(),
                }],
            }],
            lifecycles: vec![
                MavenLifecycle {
                    id: "clean".into(),
                    phases: ["pre-clean", "clean", "post-clean"]
                        .map(String::from)
                        .into(),
                },
                MavenLifecycle {
                    id: "default".into(),
                    phases: [
                        "validate",
                        "initialize",
                        "generate-sources",
                        "process-sources",
                        "generate-resources",
                        "process-resources",
                        "compile",
                        "process-classes",
                        "generate-test-sources",
                        "process-test-sources",
                        "generate-test-resources",
                        "process-test-resources",
                        "test-compile",
                        "process-test-classes",
                        "test",
                        "prepare-package",
                        "package",
                        "pre-integration-test",
                        "integration-test",
                        "post-integration-test",
                        "verify",
                        "install",
                        "deploy",
                    ]
                    .map(String::from)
                    .into(),
                },
                MavenLifecycle {
                    id: "site".into(),
                    phases: ["pre-site", "site", "post-site", "site-deploy"]
                        .map(String::from)
                        .into(),
                },
                MavenLifecycle {
                    id: "custom".into(),
                    phases: vec!["custom-phase".into()],
                },
            ],
        };
        let full = maven_metadata_rows(&module_id, &provider, false);
        let basic = maven_metadata_rows(&module_id, &provider, true);
        let lifecycles = |rows: &Vec<LanguageToolNode>| {
            rows.iter()
                .find(|row| row.label == "Lifecycle")
                .expect("provider lifecycles")
                .children
                .clone()
        };
        let full_lifecycles = lifecycles(&full);
        let basic_lifecycles = lifecycles(&basic);
        assert_eq!(
            basic_lifecycles
                .iter()
                .flat_map(|row| &row.children)
                .map(|row| row.label.as_str())
                .collect::<Vec<_>>(),
            [
                "clean", "validate", "compile", "test", "package", "verify", "install", "deploy",
                "site"
            ]
        );
        for (basic_lifecycle, full_lifecycle) in basic_lifecycles.iter().zip(&full_lifecycles) {
            assert_eq!(basic_lifecycle.id, full_lifecycle.id);
            for phase in &basic_lifecycle.children {
                let original = full_lifecycle
                    .children
                    .iter()
                    .find(|row| row.id == phase.id)
                    .expect("stable provider phase identity");
                assert_eq!(phase.label, original.label);
                assert_eq!(phase.accessibility_label, original.accessibility_label);
                assert!(!phase.enabled);
            }
        }
        let empty = basic_lifecycles.last().expect("custom lifecycle retained");
        assert!(empty.children.is_empty());
        assert!(
            empty
                .accessibility_label
                .contains("0 of 1 provider phases shown")
        );
        for row in basic.iter().filter(|row| row.label != "Lifecycle") {
            let original = full
                .iter()
                .find(|original| original.id == row.id)
                .expect("non-lifecycle data preserved");
            assert_eq!(format!("{row:?}"), format!("{original:?}"));
        }
        let unknown_goal = &full
            .iter()
            .find(|row| row.label == "Plugins")
            .expect("plugins")
            .children[0]
            .children[0];
        assert!(
            unknown_goal
                .accessibility_label
                .contains("coverage unknown")
        );
        assert_eq!(unknown_goal.children.len(), 1);
        assert_eq!(
            unknown_goal.children[0].label,
            "Parameter declarations unavailable"
        );
        let goal = &mut provider.plugins[0].goals[0];
        goal.parameters_known = true;
        goal.parameters = vec![
            MavenGoalParameter {
                name: "release".into(),
                alias: Some("javaRelease".into()),
                declared_type: Some("java.lang.String".into()),
                required: true,
                editable: true,
                description: Some("Compilation target".into()),
            },
            MavenGoalParameter {
                name: "project".into(),
                alias: None,
                declared_type: None,
                required: false,
                editable: false,
                description: None,
            },
        ];
        let mut empty_goal = goal.clone();
        empty_goal.name = "compile".into();
        empty_goal.parameters.clear();
        provider.plugins[0].goals.push(empty_goal);
        let full = maven_metadata_rows(&module_id, &provider, false);
        let basic = maven_metadata_rows(&module_id, &provider, true);
        let full_plugins = full
            .iter()
            .find(|row| row.label == "Plugins")
            .expect("plugins");
        let basic_plugins = basic
            .iter()
            .find(|row| row.label == "Plugins")
            .expect("plugins");
        assert_eq!(format!("{full_plugins:?}"), format!("{basic_plugins:?}"));
        let goal = &full_plugins.children[0].children[0];
        assert_eq!(goal.id, unknown_goal.id);
        assert!(
            goal.accessibility_label
                .contains("2 cached parameter declarations")
        );
        let parameter = &goal.children[0];
        assert_eq!(parameter.label, "release");
        assert!(
            parameter
                .accessibility_label
                .contains("POM alias javaRelease")
        );
        assert!(parameter.accessibility_label.contains("java.lang.String"));
        assert!(
            parameter
                .accessibility_label
                .contains("not a required CLI argument")
        );
        assert!(
            parameter
                .accessibility_label
                .contains("Configurable in POM")
        );
        assert!(parameter.accessibility_label.contains("Compilation target"));
        assert!(
            parameter
                .accessibility_label
                .contains("effective value and CLI property mapping unknown")
        );
        assert!(
            goal.children[1]
                .accessibility_label
                .contains("Read-only goal parameter")
        );
        for parameter in &goal.children {
            assert!(!parameter.enabled);
            assert!(parameter.activation_label.is_none());
            assert_ne!(parameter.id, goal.id);
        }
        let known_empty = &full_plugins.children[0].children[1];
        assert!(known_empty.children.is_empty());
        assert!(
            known_empty
                .accessibility_label
                .contains("0 cached parameter declarations")
        );
        let mut other_goal = provider.plugins[0].goals[0].clone();
        other_goal.name = "compile".into();
        provider.plugins[0].goals[1] = other_goal;
        let updated = maven_metadata_rows(&module_id, &provider, false);
        let plugins = updated
            .iter()
            .find(|row| row.label == "Plugins")
            .expect("plugins");
        assert_ne!(
            plugins.children[0].children[0].children[0].id,
            plugins.children[0].children[1].children[0].id
        );
    }

    #[test]
    fn provider_facts_preserve_named_and_unnamed_libraries() {
        let worktree_id = WorktreeId::from_usize(1);
        let snapshot = JdtlsProjectSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(1),
            worktree_id,
            generation: 2,
            projects: vec![JdtlsProject {
                membership: JdtlsProjectMembership::Unknown,
                imported_build_kind: project::jvm_project::JdtlsImportedBuildKind::Unknown,
                module_root: path(worktree_id, "application"),
                source_roots: Vec::new(),
                project_references: vec!["library".to_string()],
                library_entries: 3,
                gradle: None,
                maven: None,
                library_files: vec![JdtlsLibraryFile {
                    name: "junit-jupiter-5.11.4.jar".to_string(),
                    gradle_source_sets: vec!["main".to_string(), "test".to_string()],
                }],
                library_artifacts: vec![JdtlsLibraryArtifact {
                    coordinate: "org.junit.jupiter:junit-jupiter-api:5.11.4".to_string(),
                    scope: Some("test".to_string()),
                }],
                project_vm_known: true,
                project_vm_java_major: Some(17),
                server_launch_java_major: Some(25),
                java_source_level: Some(17),
                java_target_level: Some(21),
            }],
            partial: false,
        };
        let (rows, navigation) = project_rows_for_view(
            &snapshot,
            BuildToolView::Dependencies,
            &BuildViewPreferences::default(),
        );
        assert_eq!(rows.len(), 1);
        assert!(
            rows[0]
                .secondary_label
                .as_deref()
                .is_some_and(|label| label.contains("Java source level 17"))
        );
        assert!(
            rows[0]
                .accessibility_label
                .contains("project VM Java 17 reported by JDTLS")
        );
        assert!(
            rows[0]
                .accessibility_label
                .contains("Java target level 21 reported by JDTLS")
        );
        let mut unreported_target = snapshot.clone();
        unreported_target.projects[0].java_target_level = None;
        unreported_target.projects[0].server_launch_java_major = None;
        let (unreported_rows, _) = project_rows_for_view(
            &unreported_target,
            BuildToolView::Dependencies,
            &BuildViewPreferences::default(),
        );
        assert!(
            unreported_rows[0]
                .accessibility_label
                .contains("Java target level not reported")
        );
        assert!(
            !unreported_rows[0]
                .accessibility_label
                .contains("Java target level 17")
        );
        assert!(rows[0].accessibility_label.contains("JDTLS launch Java 25"));
        assert!(
            unreported_rows[0]
                .accessibility_label
                .contains("JDTLS launch Java unknown")
        );
        assert!(
            !unreported_rows[0]
                .accessibility_label
                .contains("JDTLS launch Java 17")
        );
        assert!(
            !unreported_rows[0]
                .accessibility_label
                .contains("JDTLS launch Java 21")
        );
        assert_eq!(rows[0].children.len(), 2);
        assert_eq!(rows[0].children[0].label, "library");
        assert!(rows[0].children[0].accessibility_label.contains("unknown"));
        assert_eq!(rows[0].children[1].label, "3 external libraries");
        assert_eq!(rows[0].children[1].children.len(), 3);
        assert_eq!(
            rows[0].children[1].children[0].label,
            "org.junit.jupiter:junit-jupiter-api:5.11.4"
        );
        assert!(
            rows[0].children[1].children[0]
                .accessibility_label
                .contains("scope test")
        );
        assert_eq!(
            rows[0].children[1].children[1].label,
            "junit-jupiter-5.11.4.jar"
        );
        assert!(
            rows[0].children[1].children[1]
                .accessibility_label
                .contains("used by source sets main, test")
        );
        assert_eq!(rows[0].children[1].children[2].label, "1 unnamed library");
        assert!(
            rows[0].children[1]
                .accessibility_label
                .contains("transitive")
        );
        assert!(rows[0].children[1].accessibility_label.contains("Maven"));
        assert!(rows[0].children[1].accessibility_label.contains("Gradle"));
        let mut gradle_snapshot = snapshot.clone();
        gradle_snapshot.projects[0].library_artifacts.clear();
        let (gradle_rows, _) = project_rows_for_view(
            &gradle_snapshot,
            BuildToolView::Dependencies,
            &BuildViewPreferences::default(),
        );
        let gradle_library_label = &gradle_rows[0].children[1].accessibility_label;
        assert!(gradle_library_label.contains("Gradle source-set use"));
        assert!(!gradle_library_label.contains("Maven"));
        assert_eq!(navigation.len(), 1);
        assert_eq!(
            navigation.get(&rows[0].id),
            Some(&DependencyNavigationTarget::ModuleManifest {
                root: path(worktree_id, "application"),
                build_system: None,
            })
        );
        assert!(!navigation.contains_key(&rows[0].children[0].id));
        assert!(!navigation.contains_key(&rows[0].children[1].id));

        let target = DependencyNavigation {
            path: path(worktree_id, "application/pom.xml"),
            server_id: snapshot.server_id,
            generation: snapshot.generation,
            provider: DependencyProvider::Jdtls,
            kind: DependencyNavigationKind::File,
        };
        assert!(target.matches(
            snapshot.server_id,
            snapshot.worktree_id,
            snapshot.generation,
            true
        ));
        assert!(!target.matches(
            snapshot.server_id,
            snapshot.worktree_id,
            snapshot.generation,
            false
        ));
        let newer = JdtlsProjectSnapshot {
            metadata: Default::default(),
            generation: snapshot.generation + 1,
            ..snapshot
        };
        assert!(!target.matches(newer.server_id, newer.worktree_id, newer.generation, true));
    }

    #[test]
    fn current_java_and_kotlin_provider_facts_remain_visible_together() {
        let worktree_id = WorktreeId::from_usize(8);
        let jdtls = JdtlsProjectSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(1),
            worktree_id,
            generation: 2,
            projects: vec![JdtlsProject {
                membership: JdtlsProjectMembership::Unknown,
                imported_build_kind: project::jvm_project::JdtlsImportedBuildKind::Unknown,
                module_root: path(worktree_id, "application"),
                source_roots: Vec::new(),
                project_references: Vec::new(),
                library_entries: 1,
                gradle: None,
                maven: None,
                library_files: Vec::new(),
                library_artifacts: vec![JdtlsLibraryArtifact {
                    coordinate: "org.junit.jupiter:junit-jupiter-api:5.11.4".to_string(),
                    scope: Some("test".to_string()),
                }],
                project_vm_known: true,
                project_vm_java_major: None,
                server_launch_java_major: None,
                java_source_level: None,
                java_target_level: None,
            }],
            partial: false,
        };
        let kotlin = KotlinWorkspaceSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(2),
            worktree_id,
            generation: 3,
            projects: vec![KotlinWorkspaceProject {
                name: "application.main".to_string(),
                build_system: KotlinBuildSystem::Gradle,
                module_root: path(worktree_id, "application"),
                source_roots: Vec::new(),
                dependencies: vec![KotlinWorkspaceDependency::Artifact {
                    coordinate: "org.example:api:1.2.3".to_string(),
                    scope: Some("compile".to_string()),
                }],
                unidentified_libraries: 0,
            }],
            partial: false,
        };

        let (projections, omitted_stale) = provider_projections_for_view(
            Some((&jdtls, true)),
            Some((&kotlin, true)),
            BuildToolView::Dependencies,
            &BuildViewPreferences::default(),
        );
        assert!(!omitted_stale);
        assert_eq!(projections.len(), 2);
        assert_eq!(projections[0].rows[0].label, "JDTLS: application");
        assert!(
            !projections[0].rows[0]
                .accessibility_label
                .contains("Java source level")
        );
        assert_eq!(
            projections[0].rows[0].children[0].children[0].label,
            "org.junit.jupiter:junit-jupiter-api:5.11.4"
        );
        assert_eq!(projections[1].rows[0].label, "Kotlin LSP: application.main");
        assert_eq!(
            projections[1].rows[0].children[0].label,
            "org.example:api:1.2.3"
        );
        assert_ne!(projections[0].rows[0].id, projections[1].rows[0].id);

        let (projections, omitted_stale) = provider_projections_for_view(
            Some((&jdtls, false)),
            Some((&kotlin, true)),
            BuildToolView::Dependencies,
            &BuildViewPreferences::default(),
        );
        assert!(omitted_stale);
        assert_eq!(projections.len(), 1);
        assert_eq!(projections[0].rows[0].label, "application.main");
        let (grouped, omitted_stale) = provider_projections_for_view(
            Some((&jdtls, false)),
            Some((&kotlin, true)),
            BuildToolView::Gradle,
            &BuildViewPreferences {
                group_gradle_modules: true,
                ..Default::default()
            },
        );
        assert!(omitted_stale);
        assert_eq!(grouped.len(), 1);
        assert!(grouped[0].partial);
        let group = grouped[0]
            .rows
            .first()
            .expect("current Kotlin directory group");
        assert!(
            group
                .accessibility_label
                .contains("logical Gradle grouping unavailable")
        );
        let module = group.children.first().expect("provider module retained");
        assert_eq!(module.id, projections[0].rows[0].id);
        assert!(grouped[0].navigation_targets.contains_key(&module.id));
        assert!(!grouped[0].navigation_targets.contains_key(&group.id));
    }

    #[test]
    fn kotlin_provider_rows_keep_scopes_and_unknown_directness_in_shared_tree() {
        let worktree_id = WorktreeId::from_usize(4);
        let snapshot = KotlinWorkspaceSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(9),
            worktree_id,
            generation: 3,
            projects: vec![KotlinWorkspaceProject {
                name: "application.main".to_string(),
                build_system: KotlinBuildSystem::Gradle,
                module_root: path(worktree_id, "application"),
                source_roots: Vec::new(),
                dependencies: vec![
                    KotlinWorkspaceDependency::Module {
                        name: "library.main".to_string(),
                        scope: Some("compile".to_string()),
                    },
                    KotlinWorkspaceDependency::Artifact {
                        coordinate: "org.example:api:1.2.3".to_string(),
                        scope: Some("test".to_string()),
                    },
                ],
                unidentified_libraries: 1,
            }],
            partial: true,
        };
        let (rows, navigation) = kotlin_rows_for_view(&snapshot, BuildToolView::Dependencies);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].label, "application.main");
        assert_eq!(rows[0].children.len(), 3);
        assert_eq!(rows[0].children[1].label, "org.example:api:1.2.3");
        assert!(
            rows[0].children[1]
                .accessibility_label
                .contains("classpath scope test")
        );
        assert!(
            rows[0].children[1]
                .accessibility_label
                .contains("transitive relationship unknown")
        );
        assert_eq!(rows[0].children[2].label, "1 unnamed libraries");
        assert_eq!(
            navigation.get(&rows[0].id),
            Some(&DependencyNavigationTarget::ModuleManifest {
                root: path(worktree_id, "application"),
                build_system: Some(KotlinBuildSystem::Gradle),
            })
        );
        assert!(!navigation.contains_key(&rows[0].children[1].id));
        assert_eq!(
            visible_module_manifest(
                &path(worktree_id, "application"),
                Some(KotlinBuildSystem::Gradle),
                |candidate| candidate.as_unix_str() == "application/build.gradle.kts"
            )
            .expect("Gradle manifest")
            .path
            .as_unix_str(),
            "application/build.gradle.kts"
        );
    }

    #[test]
    fn manifest_navigation_uses_provider_build_system_and_rejects_ambiguity() {
        let root = path(WorktreeId::from_usize(5), "application");
        let both_visible = |candidate: &RelPath| {
            matches!(
                candidate.as_unix_str(),
                "application/pom.xml" | "application/build.gradle.kts"
            )
        };
        assert_eq!(
            visible_module_manifest(&root, Some(KotlinBuildSystem::Gradle), both_visible)
                .expect("provider-selected Gradle manifest")
                .path
                .as_unix_str(),
            "application/build.gradle.kts"
        );
        assert_eq!(
            visible_module_manifest(&root, Some(KotlinBuildSystem::Maven), both_visible)
                .expect("provider-selected Maven manifest")
                .path
                .as_unix_str(),
            "application/pom.xml"
        );
        assert!(visible_module_manifest(&root, None, both_visible).is_none());
        assert!(
            visible_module_manifest(&root, Some(KotlinBuildSystem::Unknown), both_visible)
                .is_none()
        );
        let snapshot = KotlinWorkspaceSnapshot {
            metadata: Default::default(),
            server_id: lsp::LanguageServerId(1),
            worktree_id: root.worktree_id,
            generation: 1,
            projects: vec![KotlinWorkspaceProject {
                name: "application.main".to_string(),
                build_system: KotlinBuildSystem::Unknown,
                module_root: root,
                source_roots: Vec::new(),
                dependencies: Vec::new(),
                unidentified_libraries: 0,
            }],
            partial: false,
        };
        let (mut rows, _) = kotlin_rows_for_view(&snapshot, BuildToolView::Dependencies);
        mark_manifest_navigation(&mut rows, &HashMap::new());
        assert!(!rows[0].enabled);
        assert!(rows[0].activation_label.is_none());
        assert!(rows[0].accessibility_label.contains("no unambiguous"));
    }

    #[gpui::test]
    async fn kotlin_panel_reads_temporary_provider_export_and_cleans_it_up(
        cx: &mut TestAppContext,
    ) {
        struct KotlinWorkspaceImportState;
        impl lsp::notification::Notification for KotlinWorkspaceImportState {
            type Params = serde_json::Value;
            const METHOD: &'static str = "intellij/workspaceImportState";
        }

        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-kotlin-panel",
            serde_json::json!({
                "application": {
                    "build.gradle.kts": "plugins { kotlin(\"jvm\") }",
                    "Main.java": "class Main {}",
                    "Main.kt": "class Main"
                }
            }),
        )
        .await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/groovy-kotlin-panel".as_ref()], cx)
                .await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktree_store.read_with(cx, |store, cx| {
            store
                .worktrees()
                .next()
                .expect("fixture worktree")
                .read(cx)
                .id()
        });
        let trusted_worktrees =
            cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust global"));
        trusted_worktrees.update(cx, |trusted, cx| {
            trusted.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            );
        });
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Kotlin".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["kt".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let export_directory = Arc::new(std::sync::Mutex::new(None::<std::path::PathBuf>));
        let mut fake_servers = languages.register_fake_lsp(
            "Kotlin",
            FakeLspAdapter {
                name: "kotlin-lsp",
                capabilities: lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec!["exportWorkspace".to_string()],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                initializer: Some(Box::new({
                    let export_directory = export_directory.clone();
                    move |fake_server| {
                        fake_server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                            let export_directory = export_directory.clone();
                            move |params, _| {
                                let export_directory = export_directory.clone();
                                async move {
                                    anyhow::ensure!(params.command == "exportWorkspace");
                                    let directory = params.arguments.first()
                                        .and_then(serde_json::Value::as_str)
                                        .ok_or_else(|| anyhow::anyhow!("missing export directory"))?;
                                    let directory = std::path::PathBuf::from(directory);
                                    *export_directory.lock().expect("export directory lock") = Some(directory.clone());
                                    std::fs::write(
                                        directory.join("workspace.json"),
                                        serde_json::to_vec(&serde_json::json!({
                                            "externalSystem": "GRADLE",
                                            "libraries": [{"name": "Gradle: org.example:api:1.2.3"}],
                                            "modules": [{
                                                "name": "application.main",
                                                "externalProjectPath": "/groovy-kotlin-panel/application",
                                                "dependencies": [{
                                                    "type": "library",
                                                    "name": "Gradle: org.example:api:1.2.3",
                                                    "scope": "compile"
                                                }]
                                            }]
                                        }))?,
                                    )?;
                                    Ok(None)
                                }
                            }
                        });
                    }
                })),
                ..FakeLspAdapter::default()
            },
        );
        let mut java_servers = languages.register_fake_lsp(
            "Java",
            FakeLspAdapter {
                name: "jdtls",
                capabilities: lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![
                            "java.project.getAll".to_string(),
                            "java.project.getSettings".to_string(),
                        ],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                initializer: Some(Box::new(|fake_server| {
                    fake_server.set_request_handler::<lsp::request::ExecuteCommand, _, _>(
                        |params, _| async move {
                            match params.command.as_str() {
                                "java.project.getAll" => Ok(Some(serde_json::json!([
                                    "file:///groovy-kotlin-panel/application"
                                ]))),
                                "java.project.getSettings" => Ok(Some(serde_json::json!({
                                    "org.eclipse.jdt.ls.core.sourcePaths": [],
                                    "org.eclipse.jdt.ls.core.classpathEntries": [{
                                        "kind": 1,
                                        "path": "/private/cache/junit-jupiter.jar",
                                        "attributes": {"gradle_used_by_scope": "main,test"}
                                    }]
                                }))),
                                _ => anyhow::bail!("unexpected project command"),
                            }
                        },
                    );
                })),
                ..FakeLspAdapter::default()
            },
        );
        let _kotlin_buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/groovy-kotlin-panel/application/Main.kt", cx)
            })
            .await
            .expect("open Kotlin source");
        let fake_server = fake_servers.next().await.expect("Kotlin provider");
        let _java_buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/groovy-kotlin-panel/application/Main.java", cx)
            })
            .await
            .expect("open Java source");
        let _java_server = java_servers.next().await.expect("JDTLS provider");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();

        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Groovy Dependencies panel");
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.refresh(cx);
        });
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        for _ in 0..40 {
            cx.background_executor
                .timer(std::time::Duration::from_millis(50))
                .await;
            cx.run_until_parked();
            if panel.read_with(cx, |panel, _| {
                panel.host.status()
                    == &language_tools::language_tool_tree::LanguageToolTreeStatus::Current
                    && panel.host.visible_rows().len() == 2
            }) {
                break;
            }
        }
        panel.read_with(cx, |panel, cx| {
            assert_eq!(
                panel.host.status(),
                &language_tools::language_tool_tree::LanguageToolTreeStatus::Current
            );
            let rows = panel.host.visible_rows();
            let store = project.read(cx).lsp_store();
            let store = store.read(cx);
            assert_eq!(
                rows.len(),
                2,
                "rows {:?}; Java snapshot {:?}; pending {}; failed {}; servers {:?}",
                rows.iter().map(|row| &row.label).collect::<Vec<_>>(),
                store.jdtls_project_snapshot(worktree_id),
                store.jdtls_project_query_pending(worktree_id),
                store.jdtls_project_query_failed(worktree_id),
                store
                    .language_server_statuses()
                    .map(|(id, status)| (id, status.name.clone(), status.worktree))
                    .collect::<Vec<_>>()
            );
            assert_eq!(rows[0].label, "JDTLS: application");
            assert_eq!(rows[1].label, "Kotlin LSP: application.main");
            assert!(rows[0].accessibility_label.contains("JDTLS module"));
            assert!(rows[1].accessibility_label.contains("Kotlin LSP module"));
            assert_eq!(panel.navigation.len(), 2);
            let manifest = panel.navigation.get(&rows[1].id).expect("Gradle manifest");
            assert_eq!(
                manifest.path.path.as_unix_str(),
                "application/build.gradle.kts"
            );
        });
        let export_directory = export_directory
            .lock()
            .expect("export directory lock")
            .clone()
            .expect("provider export directory");
        assert!(
            !export_directory.exists(),
            "temporary export must be removed"
        );
        assert!(
            !std::path::Path::new("/groovy-kotlin-panel/workspace.json").exists(),
            "provider export must not target the project root"
        );
        fake_server.notify::<KotlinWorkspaceImportState>(serde_json::json!({
            "phase": "FINISHED",
            "folders": [{"tool": "gradle", "status": "FAILED"}]
        }));
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(matches!(
                panel.host.status(),
                language_tools::language_tool_tree::LanguageToolTreeStatus::StaleError(message)
                    if message.contains("Kotlin project import failed")
            ));
            assert_eq!(panel.host.visible_rows().len(), 1);
            assert_eq!(panel.host.visible_rows()[0].label, "application");
        });
        fake_server.notify::<KotlinWorkspaceImportState>(serde_json::json!({
            "phase": "FINISHED",
            "folders": [{"tool": "gradle", "status": "SUCCESS"}]
        }));
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        for _ in 0..40 {
            cx.background_executor
                .timer(std::time::Duration::from_millis(50))
                .await;
            cx.run_until_parked();
            if panel.read_with(cx, |panel, _| {
                panel.host.status()
                    == &language_tools::language_tool_tree::LanguageToolTreeStatus::Current
                    && panel.host.visible_rows().len() == 2
            }) {
                break;
            }
        }
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel.host.status(),
                &language_tools::language_tool_tree::LanguageToolTreeStatus::Current
            );
            assert_eq!(panel.host.visible_rows().len(), 2);
        });
        fs.insert_file(
            std::path::Path::new("/groovy-kotlin-panel/application/build.gradle.kts"),
            b"plugins { kotlin(\"jvm\") }; version = 2".to_vec(),
        )
        .await;
        cx.run_until_parked();
        let (server_id, snapshot_worktree_id, generation, current) =
            project.read_with(cx, |project, cx| {
                project.lsp_store().read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .kotlin_workspace_snapshot(worktree_id)
                        .expect("stale provider snapshot");
                    (
                        snapshot.server_id,
                        snapshot.worktree_id,
                        snapshot.generation,
                        current,
                    )
                })
            });
        panel.read_with(cx, |panel, _| {
            assert!(matches!(
                panel.host.status(),
                language_tools::language_tool_tree::LanguageToolTreeStatus::Stale(_)
            ));
            let navigation = panel
                .navigation
                .values()
                .next()
                .expect("stale module target");
            assert!(!navigation.matches(server_id, snapshot_worktree_id, generation, current,));
        });
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum MavenGraphOutcome {
        Success,
        PanelControls,
        OperationControls,
        OrderedOperationControls,
        Unavailable,
        Failure,
        Coordinates,
        Cancellation,
        InputChange,
        Restart,
        Revocation,
        ScopeChange,
        OtherScope,
    }

    #[gpui::test]
    async fn collaborative_graph_controls_reject_client_collection(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/remote-graph-guard",
            serde_json::json!({"pom.xml":"<project/>"}),
        )
        .await;
        let project = Project::test(fs, ["/remote-graph-guard".as_ref()], cx).await;
        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("fixture worktree")
                .read(cx)
                .id()
        });
        project.update(cx, |project, _| project.mark_as_collab_for_testing());
        assert!(project.read_with(cx, |project, _| project.is_remote()));
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let panel = DependenciesPanel::load(
            workspace.downgrade(),
            cx.update(|window, cx| window.to_async(cx)),
        )
        .await
        .expect("nonlocal graph panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace
                .right_dock()
                .update(cx, |dock, cx| dock.set_open(true, window, cx));
            workspace.focus_panel::<DependenciesPanel>(window, cx);
        });
        let selection_id = LanguageToolNodeId("held-selection-before-host-change".into());
        panel.update(cx, |panel, cx| {
            panel.host.replace_snapshot(LanguageToolSnapshot {
                roots: vec![graph_fact_row(
                    selection_id.clone(),
                    "Held project selection".into(),
                    "Test host-change race".into(),
                    Vec::new(),
                )],
                status: LanguageToolProviderStatus::Current,
            });
            panel.graph_selections.insert(
                selection_id.clone(),
                GraphSelection {
                    server_id: lsp::LanguageServerId(1),
                    generation: 1,
                    target: GraphQueryTarget::Maven(path(worktree_id, "pom.xml")),
                },
            );
            panel.host.select(selection_id);
            cx.notify();
        });
        cx.update(|window, _| window.activate_a11y_for_test());
        for view in [
            BuildToolView::Dependencies,
            BuildToolView::Maven,
            BuildToolView::Gradle,
        ] {
            panel.update(cx, |panel, cx| {
                panel.view = view;
                cx.notify();
            });
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value = serde_json::from_str(
                        &window.debug_a11y_tree_json().expect("remote controls"),
                    )
                    .expect("a11y JSON");
                    let collect = tree["nodes"]
                    .as_object()
                    .expect("nodes")
                    .values()
                    .find(|node| {
                        node["aria"]["label"]
                            == "Offline dependency graph collection unavailable for remote projects"
                    })
                    .expect("explicit nonlocal boundary");
                    assert_eq!(collect["aria"]["disabled"], true);
                    for (label, visible) in [
                        (
                            "Discovered offline build execution unavailable for remote projects",
                            view != BuildToolView::Dependencies,
                        ),
                        (
                            "Saved Maven selection unavailable for remote projects",
                            view != BuildToolView::Gradle,
                        ),
                        (
                            "Gradle selection import unavailable for remote projects",
                            view != BuildToolView::Maven,
                        ),
                    ] {
                        if !visible {
                            continue;
                        }
                        assert!(
                            tree["nodes"]
                                .as_object()
                                .expect("nodes")
                                .values()
                                .any(|node| node["aria"]["label"] == label
                                    && node["aria"]["disabled"] == true)
                        );
                    }
                });
            }
        }
        for maven in [true, false] {
            panel.update(cx, |panel, cx| {
                assert!(panel.selected_build_worktree(cx).is_err());
                panel.apply_saved_build_selection(maven, cx);
                assert!(panel.build_phase_token.is_none() && panel.build_phase_waiter.is_none());
                assert!(
                    panel
                        .build_phase_notice
                        .as_deref()
                        .is_some_and(|notice| notice.contains("Remote import is not supported"))
                );
            });
            assert!(lsp_store.read_with(cx, |store, _| {
                store.jdtls_project_import_token(worktree_id).is_none()
            }));
        }
        let spawns = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(OperationTerminal(spawns.clone()))
        });
        panel.update_in(cx, |panel, window, cx| {
            panel.run_selected_offline_operation(window, cx);
            assert!(
                panel
                    .operation_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("host is unavailable"))
            );
        });
        let receipt = project::jvm_project::JvmBuildOperationReceipt {
            server_id: lsp::LanguageServerId(1),
            worktree_id,
            generation: 1,
            module_root: path(worktree_id, ""),
            operations: vec![project::jvm_project::JvmBuildOperation::MavenPhase {
                lifecycle: "default".into(),
                phase: "verify".into(),
            }],
        };
        panel.update_in(cx, |panel, window, cx| {
            panel.ordered_operations = vec![receipt.clone()];
            panel.add_selected_ordered_operation(cx);
            assert!(
                panel
                    .operation_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("local project host"))
            );
            panel.run_ordered_offline_operations(window, cx);
            assert!(
                panel
                    .operation_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("local project host"))
            );
            panel.rebuild(cx);
            assert!(panel.ordered_operations.is_empty());
        });
        workspace.update_in(cx, |workspace, window, cx| {
            let error = workspace
                .run_jvm_operation(receipt, window, cx)
                .expect_err("client execution unavailable")
                .to_string();
            assert!(error.contains("local host") || error.contains("Trust"));
        });
        cx.run_until_parked();
        assert!(spawns.lock().expect("Task events").is_empty());
        cx.dispatch_action(CollectSelectedGraph);
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel.graph_notice.as_deref(),
                Some(
                    "Remote graph collection is unavailable; inspect or refresh existing host data"
                )
            );
            assert!(panel.graph_query_token.is_none());
            assert!(panel.graph_query.is_none());
        });
        assert!(!lsp_store.read_with(cx, |store, _| {
            store.jdtls_project_query_pending(worktree_id)
        }));
        assert!(
            !workspace
                .read_with(cx, |workspace, _| workspace.notification_ids())
                .is_empty()
        );
    }

    #[gpui::test]
    async fn maven_panel_collects_cancels_and_recovers_offline_graph(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Maven,
            MavenGraphOutcome::PanelControls,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_panel_collects_cancels_and_recovers_offline_graph(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::PanelControls,
        )
        .await;
    }

    #[gpui::test]
    async fn maven_graph_admission_uses_current_owner_and_advances_receipt(
        cx: &mut TestAppContext,
    ) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::Success).await;
    }
    #[gpui::test]
    async fn maven_graph_unavailable_command_never_invalidates_current_metadata(
        cx: &mut TestAppContext,
    ) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::Unavailable).await;
    }
    #[gpui::test]
    async fn maven_graph_provider_failure_is_sanitized_and_explicitly_recovered(
        cx: &mut TestAppContext,
    ) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::Failure).await;
    }
    #[gpui::test]
    async fn maven_graph_coordinates_must_match_current_cached_metadata(cx: &mut TestAppContext) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::Coordinates).await;
    }
    #[gpui::test]
    async fn maven_graph_cancellation_drops_owned_request_and_requires_refresh(
        cx: &mut TestAppContext,
    ) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::Cancellation).await;
    }
    #[gpui::test]
    async fn maven_graph_external_input_change_rejects_held_reply(cx: &mut TestAppContext) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::InputChange).await;
    }
    #[gpui::test]
    async fn maven_graph_restart_discards_held_reply_and_does_not_recollect(
        cx: &mut TestAppContext,
    ) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::Restart).await;
    }
    #[gpui::test]
    async fn maven_graph_trust_revocation_discards_held_reply(cx: &mut TestAppContext) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::Revocation).await;
    }
    #[gpui::test]
    async fn maven_graph_initialized_folder_change_rejects_held_reply(cx: &mut TestAppContext) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::ScopeChange).await;
    }

    #[gpui::test]
    async fn maven_graph_unrelated_workspace_folder_preserves_owned_scope(cx: &mut TestAppContext) {
        maven_graph_lifecycle(cx, MavenGraphOutcome::OtherScope).await;
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum GraphFixtureProvider {
        Maven,
        Gradle,
    }

    fn query_fixture_graph(
        store: &mut project::lsp_store::LspStore,
        worktree_id: WorktreeId,
        selected_project: ProjectPath,
        cx: &mut gpui::Context<project::lsp_store::LspStore>,
        provider: GraphFixtureProvider,
        configuration: &str,
    ) -> anyhow::Result<gpui::Task<anyhow::Result<()>>> {
        match provider {
            GraphFixtureProvider::Maven => {
                store.query_local_maven_dependency_graph(worktree_id, selected_project, cx)
            }
            GraphFixtureProvider::Gradle => store.query_local_gradle_dependency_graph(
                worktree_id,
                selected_project,
                configuration.into(),
                cx,
            ),
        }
    }

    struct OperationTerminal(std::sync::Arc<std::sync::Mutex<Vec<task::SpawnInTerminal>>>);

    impl workspace::TerminalProvider for OperationTerminal {
        fn spawn(
            &self,
            task: task::SpawnInTerminal,
            _window: &mut Window,
            _cx: &mut App,
        ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
            self.0.lock().expect("Task events").push(task);
            gpui::Task::ready(Some(Ok(std::process::ExitStatus::default())))
        }
    }

    #[gpui::test]
    async fn maven_panel_runs_current_phase_and_exact_plugin_goal_through_tasks(
        cx: &mut TestAppContext,
    ) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Maven,
            MavenGraphOutcome::OperationControls,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_panel_runs_current_literal_task_paths_through_tasks(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::OperationControls,
        )
        .await;
    }

    #[gpui::test]
    async fn maven_panel_composes_ordered_current_operations_through_tasks(
        cx: &mut TestAppContext,
    ) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Maven,
            MavenGraphOutcome::OrderedOperationControls,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_panel_composes_ordered_current_operations_through_tasks(
        cx: &mut TestAppContext,
    ) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::OrderedOperationControls,
        )
        .await;
    }

    async fn exercise_operation_controls(
        project: &Entity<Project>,
        lsp_store: &Entity<project::LspStore>,
        worktree_id: WorktreeId,
        provider: GraphFixtureProvider,
        fs: &Arc<project::FakeFs>,
        commands: &Arc<AtomicUsize>,
        version: &Arc<AtomicUsize>,
        ordered_only: bool,
        cx: &mut TestAppContext,
    ) {
        use project::Fs as _;
        async fn refresh_operations(
            panel: &Entity<DependenciesPanel>,
            lsp_store: &Entity<project::LspStore>,
            worktree_id: WorktreeId,
            provider: GraphFixtureProvider,
            cx: &mut gpui::VisualTestContext,
        ) {
            cx.run_until_parked();
            cx.background_executor
                .timer(std::time::Duration::from_millis(50))
                .await;
            cx.run_until_parked();
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            if provider == GraphFixtureProvider::Gradle {
                lsp_store
                    .update(cx, |store, cx| {
                        store.synchronize_local_gradle_project(Some(worktree_id), cx)
                    })
                    .expect("Explicit synchronization after restored Gradle wrapper");
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
                lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            }
            for _ in 0..40 {
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.background_executor
                    .timer(std::time::Duration::from_millis(50))
                    .await;
                cx.run_until_parked();
                panel.update(cx, |panel, cx| panel.rebuild(cx));
                if panel.read_with(cx, |panel, cx| {
                    panel.operation_selections.len() == 2
                        && lsp_store
                            .read(cx)
                            .jdtls_project_snapshot(worktree_id)
                            .is_some_and(|(_, current)| current)
                }) {
                    return;
                }
            }
            panic!(
                "Current operation metadata did not recover: {:?}",
                panel.read_with(cx, |panel, _| panel.host.status().clone())
            );
        }
        let view = if provider == GraphFixtureProvider::Maven {
            BuildToolView::Maven
        } else {
            BuildToolView::Gradle
        };
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(OperationTerminal(events.clone()))
        });
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("operation panel");
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.view = view;
            match view {
                BuildToolView::Maven => panel.preferences.show_maven_toolbar = false,
                BuildToolView::Gradle => panel.preferences.show_gradle_toolbar = false,
                _ => {}
            }
            panel.rebuild(cx);
            panel.host.expand_all();
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace
                .right_dock()
                .update(cx, |dock, cx| dock.set_open(true, window, cx));
            workspace.focus_panel::<DependenciesPanel>(window, cx);
        });
        let selections = panel.read_with(cx, |panel, _| {
            assert_eq!(panel.operation_selections.len(), 2);
            let mut selections = panel
                .operation_selections
                .iter()
                .map(|(id, receipt)| (id.clone(), receipt.clone()))
                .collect::<Vec<_>>();
            selections.sort_by(|left, right| left.0.0.cmp(&right.0.0));
            selections
        });
        cx.update(|window, _| window.activate_a11y_for_test());
        let label = "Run selected offline Maven phase, plugin goal or Gradle task through Tasks; executes trusted local build scripts";
        let focus = |cx: &mut gpui::VisualTestContext, label: &str| {
            let mut found = false;
            for _ in 0..96 {
                found = cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value =
                        serde_json::from_str(&window.debug_a11y_tree_json().expect("AX"))
                            .expect("AX JSON");
                    if tree["gpui_focus"]
                        .as_str()
                        .is_some_and(|id| tree["nodes"][id]["aria"]["label"] == label)
                    {
                        true
                    } else {
                        window.focus_next(cx);
                        false
                    }
                });
                if found {
                    break;
                }
            }
            assert!(
                found,
                "shared Run control receives focus with toolbar hidden"
            );
        };
        let press = |cx: &mut gpui::VisualTestContext| {
            cx.update(|window, cx| {
                let key = gpui::Keystroke::parse("enter").expect("Enter");
                window.dispatch_event(
                    gpui::KeyDownEvent {
                        keystroke: key.clone(),
                        is_held: false,
                        prefer_character_input: false,
                    }
                    .to_platform_input(),
                    cx,
                );
                window.dispatch_event(gpui::KeyUpEvent { keystroke: key }.to_platform_input(), cx);
            });
            cx.run_until_parked();
        };
        if ordered_only {
            let add_label = "Add selected current operation to the ordered offline run";
            let run_label =
                "Run ordered offline operations for the current local module through Tasks";
            let add_order = |cx: &mut gpui::VisualTestContext| {
                for index in [1, 0, 1] {
                    let (id, _) = selections
                        .get(index)
                        .expect("two current typed operation rows");
                    panel.update(cx, |panel, cx| {
                        panel.host.select(id.clone());
                        cx.notify();
                    });
                    focus(cx, add_label);
                    press(cx);
                }
            };
            panel.update_in(cx, |panel, window, cx| {
                panel.run_ordered_offline_operations(window, cx)
            });
            cx.run_until_parked();
            assert!(events.lock().expect("events").is_empty());
            add_order(cx);
            panel.read_with(cx, |panel, _| {
                assert_eq!(panel.ordered_operations.len(), 3);
                for (receipt, index) in panel.ordered_operations.iter().zip([1, 0, 1]) {
                    assert_eq!(receipt, &selections.get(index).expect("row").1);
                }
                let receipts = vec![selections.first().expect("current row").1.clone(); 32];
                assert_eq!(
                    compose_ordered_operations(&receipts, &panel.operation_selections)
                        .expect("32 ordered operations")
                        .operations
                        .len(),
                    32
                );
                let mut too_many = receipts.clone();
                too_many.push(selections.first().expect("row").1.clone());
                assert!(
                    compose_ordered_operations(&too_many, &panel.operation_selections).is_err()
                );
                let mut foreign = selections.first().expect("row").1.clone();
                foreign.module_root.path = path(worktree_id, "library").path;
                let mut current = panel.operation_selections.clone();
                current.insert(LanguageToolNodeId("foreign-module".into()), foreign.clone());
                assert!(
                    compose_ordered_operations(
                        &[receipts.first().expect("first").clone(), foreign],
                        &current
                    )
                    .is_err()
                );
                let mut wrong_generation = receipts.first().expect("first").clone();
                wrong_generation.generation += 1;
                current.insert(
                    LanguageToolNodeId("foreign-generation".into()),
                    wrong_generation.clone(),
                );
                assert!(
                    compose_ordered_operations(
                        &[receipts.first().expect("first").clone(), wrong_generation],
                        &current
                    )
                    .is_err()
                );
                let mut mixed = receipts.first().expect("first").clone();
                mixed.operations = if provider == GraphFixtureProvider::Maven {
                    vec![JvmBuildOperation::GradleTask {
                        path: ":application:verify".into(),
                    }]
                } else {
                    vec![JvmBuildOperation::MavenPhase {
                        lifecycle: "default".into(),
                        phase: "verify".into(),
                    }]
                };
                current.insert(
                    LanguageToolNodeId("mixed-build-system".into()),
                    mixed.clone(),
                );
                assert!(
                    compose_ordered_operations(
                        &[receipts.first().expect("first").clone(), mixed],
                        &current
                    )
                    .is_err()
                );
                assert!(
                    panel
                        .host
                        .visible_rows()
                        .iter()
                        .any(|row| row.id == ordered_operation_root_id())
                );
            });
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value =
                        serde_json::from_str(&window.debug_a11y_tree_json().expect("AX"))
                            .expect("AX JSON");
                    let nodes = tree["nodes"].as_object().expect("nodes");
                    assert!(nodes.values().any(|node| {
                        node["aria"]["label"]
                            .as_str()
                            .is_some_and(|label| label.contains("Ordered offline operations (3)"))
                    }));
                    assert!(nodes.values().any(|node| node["aria"]["label"] == run_label
                        && node["aria"]["role"] == "Button"
                        && node["aria"]["value"] == "3 selected"));
                });
            }
            assert!(
                events.lock().expect("events").is_empty(),
                "composition is input state, not execution"
            );
            focus(
                cx,
                "Remove last ordered operation without starting execution",
            );
            press(cx);
            assert_eq!(
                panel.read_with(cx, |panel, _| panel.ordered_operations.len()),
                2
            );
            focus(
                cx,
                "Clear ordered operation selection; running Tasks are unchanged",
            );
            press(cx);
            assert!(panel.read_with(cx, |panel, _| panel.ordered_operations.is_empty()));
            add_order(cx);
            let combined = panel.read_with(cx, |panel, _| {
                compose_ordered_operations(&panel.ordered_operations, &panel.operation_selections)
                    .expect("combined receipt")
            });
            let expected = lsp_store.read_with(cx, |store, _| {
                let (snapshot, current) =
                    store.jdtls_project_snapshot(worktree_id).expect("snapshot");
                combined
                    .plan(snapshot, current, util::paths::PathStyle::Unix)
                    .expect("existing plan")
            });
            focus(cx, run_label);
            press(cx);
            {
                let events = events.lock().expect("events");
                assert_eq!(events.len(), 1);
                let task = events.first().expect("one shared Tasks invocation");
                assert_eq!(task.args, expected.template.args);
                assert_eq!(task.literal_prefix_args, task.args.len());
            }
            let held = panel.read_with(cx, |panel, _| panel.ordered_operations.clone());
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            panel.update_in(cx, |panel, window, cx| {
                panel.rebuild(cx);
                assert!(panel.ordered_operations.is_empty());
                assert!(
                    panel
                        .operation_notice
                        .as_deref()
                        .is_some_and(|notice| notice.contains("cleared"))
                );
                panel.ordered_operations = held;
                panel.run_ordered_offline_operations(window, cx);
            });
            cx.run_until_parked();
            assert_eq!(events.lock().expect("events").len(), 1);
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            panel.update(cx, |panel, cx| {
                panel.rebuild(cx);
                panel.host.expand_all();
                assert!(panel.ordered_operations.is_empty());
            });
            add_order(cx);
            let wrapper = if provider == GraphFixtureProvider::Maven {
                "/groovy-graph/mvnw"
            } else {
                "/groovy-graph/gradlew"
            };
            fs.remove_file(std::path::Path::new(wrapper), Default::default())
                .await
                .expect("remove owned wrapper");
            cx.run_until_parked();
            panel.update_in(cx, |panel, window, cx| {
                panel.run_ordered_offline_operations(window, cx)
            });
            cx.run_until_parked();
            assert_eq!(events.lock().expect("events").len(), 1);
            fs.insert_file(wrapper, b"#!/bin/sh".to_vec()).await;
            cx.run_until_parked();
            panel.update(cx, |panel, cx| {
                panel.clear_ordered_operations(cx);
                panel.host.expand_all();
            });
            refresh_operations(&panel, lsp_store, worktree_id, provider, cx).await;
            panel.update(cx, |panel, _| panel.host.expand_all());
            add_order(cx);
            assert_eq!(
                panel.read_with(cx, |panel, _| panel.ordered_operations.len()),
                3
            );
            focus(cx, run_label);
            press(cx);
            assert_eq!(
                events.lock().expect("events").len(),
                2,
                "Recovered ordered run: {:?}",
                panel.read_with(cx, |panel, _| panel.operation_notice.clone())
            );
            let store = project.read_with(cx, |project, _| project.worktree_store());
            let trust = cx.read(|cx| TrustedWorktrees::try_get_global(cx).expect("trust"));
            trust.update(cx, |trust, cx| {
                trust.restrict(
                    store.downgrade(),
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                )
            });
            cx.run_until_parked();
            panel.update_in(cx, |panel, window, cx| {
                panel.rebuild(cx);
                assert!(panel.ordered_operations.is_empty());
                panel.run_ordered_offline_operations(window, cx);
            });
            cx.run_until_parked();
            assert_eq!(events.lock().expect("events").len(), 2);
            assert_eq!(
                commands.load(Ordering::SeqCst),
                0,
                "all execution uses existing Tasks"
            );
            return;
        }
        for (id, receipt) in &selections {
            panel.update(cx, |panel, cx| {
                panel.host.expand_all();
                panel.host.select(id.clone());
                assert_eq!(panel.host.selected(), Some(id));
                cx.notify();
            });
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value =
                        serde_json::from_str(&window.debug_a11y_tree_json().expect("AX"))
                            .expect("AX JSON");
                    assert!(
                        tree["nodes"]
                            .as_object()
                            .expect("nodes")
                            .values()
                            .any(|node| node["aria"]["label"] == label
                                && node["aria"]["role"] == "Button")
                    );
                });
            }
            focus(cx, label);
            press(cx);
            let plan = lsp_store.read_with(cx, |store, _| {
                let (snapshot, current) =
                    store.jdtls_project_snapshot(worktree_id).expect("snapshot");
                receipt
                    .plan(snapshot, current, util::paths::PathStyle::Unix)
                    .expect("current expected plan")
            });
            let events = events.lock().expect("events");
            let task = events.last().expect("existing Tasks terminal event");
            assert_eq!(task.args, plan.template.args);
            assert_eq!(task.literal_prefix_args, task.args.len());
            assert_eq!(
                task.command.as_deref(),
                Some(if provider == GraphFixtureProvider::Maven {
                    "/groovy-graph/mvnw"
                } else {
                    "/groovy-graph/gradlew"
                })
            );
        }
        assert_eq!(events.lock().expect("events").len(), 2);
        assert_eq!(
            commands.load(Ordering::SeqCst),
            0,
            "Run uses Tasks, never a provider execution/graph engine"
        );
        panel.read_with(cx, |panel, _| {
            assert!(
                panel
                    .operation_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("sent to Tasks"))
            )
        });
        let mut invalid = selections[0].1.clone();
        invalid.generation += 1;
        panel.update_in(cx, |panel, window, cx| {
            panel
                .operation_selections
                .insert(selections[0].0.clone(), invalid);
            panel.host.select(selections[0].0.clone());
            panel.run_selected_offline_operation(window, cx);
        });
        cx.run_until_parked();
        assert_eq!(events.lock().expect("events").len(), 2);
        panel.read_with(cx, |panel, _| {
            let error = panel.operation_notice.as_deref().expect("stale error");
            assert!(error.contains("refresh"));
            assert!(!error.contains("private-canary") && !error.contains("groovy-graph"));
        });
        lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
        panel.update_in(cx, |panel, window, cx| {
            panel.rebuild(cx);
            assert!(panel.operation_selections.is_empty());
            panel.run_selected_offline_operation(window, cx);
        });
        assert_eq!(events.lock().expect("events").len(), 2);
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            panel.rebuild(cx);
            panel.host.expand_all();
            assert_eq!(panel.operation_selections.len(), 2);
            panel.host.select(selections[0].0.clone());
        });
        let wrapper = if provider == GraphFixtureProvider::Maven {
            "/groovy-graph/mvnw"
        } else {
            "/groovy-graph/gradlew"
        };
        fs.remove_file(std::path::Path::new(wrapper), Default::default())
            .await
            .expect("remove disposable wrapper");
        cx.run_until_parked();
        panel.update_in(cx, |panel, window, cx| {
            panel.run_selected_offline_operation(window, cx)
        });
        cx.run_until_parked();
        assert_eq!(events.lock().expect("events").len(), 2);
        panel.read_with(cx, |panel, _| {
            assert!(
                panel.operation_selections.is_empty(),
                "Wrapper removal invalidates provider data"
            );
            assert!(
                panel
                    .operation_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("refresh")),
                "Stale wrapper data requires refresh: {:?}",
                panel.operation_notice
            )
        });
        fs.insert_file(wrapper, b"#!/bin/sh".to_vec()).await;
        refresh_operations(&panel, lsp_store, worktree_id, provider, cx).await;
        panel.update(cx, |panel, cx| {
            panel.rebuild(cx);
            panel.host.expand_all();
            panel.host.select(selections[0].0.clone());
        });
        focus(cx, label);
        press(cx);
        assert_eq!(events.lock().expect("events").len(), 3);
        if provider == GraphFixtureProvider::Maven {
            version.store(2, Ordering::SeqCst);
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            panel.update(cx, |panel, cx| {
                panel.rebuild(cx);
                assert_eq!(panel.operation_selections.len(), 1);
                assert!(panel.operation_selections.values().all(|receipt| matches!(
                    receipt.operations[0],
                    JvmBuildOperation::MavenPhase { .. }
                )));
            });
        }
        let store = project.read_with(cx, |project, _| project.worktree_store());
        let trust = cx.read(|cx| TrustedWorktrees::try_get_global(cx).expect("trust"));
        trust.update(cx, |trust, cx| {
            trust.restrict(
                store.downgrade(),
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        cx.run_until_parked();
        panel.update_in(cx, |panel, window, cx| {
            panel.rebuild(cx);
            assert!(panel.operation_selections.is_empty());
            panel.run_selected_offline_operation(window, cx);
        });
        cx.run_until_parked();
        assert_eq!(events.lock().expect("events").len(), 3);
        assert_eq!(commands.load(Ordering::SeqCst), 0);
    }

    async fn maven_graph_lifecycle(cx: &mut TestAppContext, outcome: MavenGraphOutcome) {
        graph_lifecycle(cx, GraphFixtureProvider::Maven, outcome).await;
    }

    #[gpui::test]
    async fn gradle_graph_admission_preserves_resolution_and_switches_configuration(
        cx: &mut TestAppContext,
    ) {
        graph_lifecycle(cx, GraphFixtureProvider::Gradle, MavenGraphOutcome::Success).await;
    }

    #[gpui::test]
    async fn gradle_graph_unavailable_command_preserves_metadata(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::Unavailable,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_graph_failure_is_sanitized_and_explicitly_recovered(cx: &mut TestAppContext) {
        graph_lifecycle(cx, GraphFixtureProvider::Gradle, MavenGraphOutcome::Failure).await;
    }

    #[gpui::test]
    async fn gradle_graph_project_identity_must_match_current_metadata(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::Coordinates,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_graph_cancellation_drops_owned_request(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::Cancellation,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_graph_external_build_input_change_rejects_reply(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::InputChange,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_graph_restart_discards_reply_without_recollection(cx: &mut TestAppContext) {
        graph_lifecycle(cx, GraphFixtureProvider::Gradle, MavenGraphOutcome::Restart).await;
    }

    #[gpui::test]
    async fn gradle_graph_trust_revocation_discards_reply(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::Revocation,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_graph_owned_folder_change_rejects_reply(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::ScopeChange,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_graph_unrelated_folder_preserves_scope(cx: &mut TestAppContext) {
        graph_lifecycle(
            cx,
            GraphFixtureProvider::Gradle,
            MavenGraphOutcome::OtherScope,
        )
        .await;
    }

    async fn graph_lifecycle(
        cx: &mut TestAppContext,
        provider: GraphFixtureProvider,
        outcome: MavenGraphOutcome,
    ) {
        use settings::SettingsStore;
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/groovy-graph", serde_json::json!({
            "pom.xml":"<project/>","settings.gradle":"include application", "build.gradle":"plugins {}", "application":{"pom.xml":"<project/>","build.gradle":"plugins {}","Main.java":"class Main {}"},
            ".env":{"pom.xml":"private"}, "notpom.txt":"unrelated",
        })).await;
        if matches!(
            outcome,
            MavenGraphOutcome::OperationControls | MavenGraphOutcome::OrderedOperationControls
        ) {
            fs.insert_file("/groovy-graph/mvnw", b"#!/bin/sh".to_vec())
                .await;
            fs.insert_file("/groovy-graph/gradlew", b"#!/bin/sh".to_vec())
                .await;
        }
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/groovy-graph".as_ref()], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktree_store.read_with(cx, |store, cx| {
            store.worktrees().next().expect("worktree").read(cx).id()
        });
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        let selected_project = path(
            worktree_id,
            if provider == GraphFixtureProvider::Maven {
                "application/pom.xml"
            } else {
                "application"
            },
        );
        assert!(
            lsp_store
                .update(cx, |store, cx| query_fixture_graph(
                    store,
                    worktree_id,
                    selected_project.clone(),
                    cx,
                    provider,
                    "parityGraph"
                ))
                .is_err()
        );
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"));
        trust.update(cx, |trusted, cx| {
            trusted.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let commands = Arc::new(AtomicUsize::new(0));
        let fail = Arc::new(AtomicBool::new(outcome == MavenGraphOutcome::Failure));
        let version = Arc::new(AtomicUsize::new(
            if outcome == MavenGraphOutcome::Coordinates {
                2
            } else {
                1
            },
        ));
        let (mut release, held) = oneshot::channel();
        let gate = Arc::new(std::sync::Mutex::new(Some(held)));
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let mut servers = languages.register_fake_lsp("Java", FakeLspAdapter {
            name:"jdtls", capabilities:lsp::ServerCapabilities {
                workspace: Some(lsp::WorkspaceServerCapabilities {workspace_folders: Some(lsp::WorkspaceFoldersServerCapabilities { supported: Some(true), change_notifications: Some(lsp::OneOf::Left(true)) }), ..Default::default()}),
                execute_command_provider:Some(lsp::ExecuteCommandOptions {
                    commands:["java.project.getAll","java.project.getSettings","groovy.maven.importedProjects","groovy.gradle.importedProjects","groovy.maven.projects","groovy.gradle.projectDetails","groovy.maven.dependencyGraph","groovy.gradle.dependencyGraph","groovy.maven.importReactor"]
                        .into_iter().filter(|command| *command != if provider == GraphFixtureProvider::Maven { "groovy.gradle.dependencyGraph" } else {"groovy.maven.dependencyGraph"})
                        .filter(|command| outcome != MavenGraphOutcome::Unavailable || !command.ends_with(".dependencyGraph")).map(str::to_owned).collect(),
                    ..Default::default()
                }), ..Default::default()
            }, initializer:Some(Box::new({
                let commands=commands.clone();let fail=fail.clone();let version=version.clone();let gate=gate.clone();
                move |server| {
                    server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                        let commands=commands.clone();let fail=fail.clone();let version=version.clone();let gate=gate.clone();
                        move |parameters, _| {
                            let commands=commands.clone();let fail=fail.clone();let version=version.clone();let gate=gate.clone();
                            async move {
                                match parameters.command.as_str() {
                                    "java.project.getAll" => Ok(Some(serde_json::json!(["file:///groovy-graph/application/"]))),
                                    "java.project.getSettings" => Ok(Some(serde_json::json!({"org.eclipse.jdt.ls.core.sourcePaths":[],"org.eclipse.jdt.ls.core.classpathEntries":[]}))),
                                    "groovy.maven.importedProjects" | "groovy.gradle.importedProjects" => {
                                        assert_eq!(parameters.arguments,[serde_json::json!({"rootUri":"file:///groovy-graph/","cachedOnly":true})]);
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":"file:///groovy-graph/","projects":if parameters.command == if provider == GraphFixtureProvider::Maven { "groovy.maven.importedProjects" } else { "groovy.gradle.importedProjects" } {vec![serde_json::json!({"projectUri":"file:///groovy-graph/application/","javaNature":true})]} else {vec![]},"partial":false,"fullSynchronizationKnown":false})))
                                    }
                                    "groovy.maven.projects" => Ok(Some(serde_json::json!({"version":1,"rootUri":"file:///groovy-graph/","cachedOnly":true,"unavailablePomUris":[],"lifecycles":if matches!(outcome, MavenGraphOutcome::OperationControls | MavenGraphOutcome::OrderedOperationControls) {vec![serde_json::json!({"id":"default","phases":["verify"]})]}else{vec![]},"reactorOrderKnown":false,
                                        "projects": if provider == GraphFixtureProvider::Maven {vec![serde_json::json!({"pomUri":"file:///groovy-graph/application/pom.xml","groupId":"dev.made.groovy.fixture.graph","artifactId":"application","version":version.load(Ordering::SeqCst).to_string(),"packaging":"pom","moduleUris":[],"profiles":{},"inheritedProfileDefinitionsKnown":false,"plugins":if matches!(outcome, MavenGraphOutcome::OperationControls | MavenGraphOutcome::OrderedOperationControls) {vec![serde_json::json!({"groupId":"org.apache.maven.plugins","artifactId":"maven-surefire-plugin","version":if version.load(Ordering::SeqCst)==2 {None}else{Some("3.5.2")},"managedOnly":false,"goalsKnown":true,"goals":[{"name":"test","phase":"test","description":null}]})]}else{vec![]}})]} else {vec![]}}))),
                                    "groovy.gradle.projectDetails" => {
                                        let details = serde_json::json!({"javaSourceSetsKnown":true,"sourceSets":[],"tasks":if matches!(outcome, MavenGraphOutcome::OperationControls | MavenGraphOutcome::OrderedOperationControls) {vec![serde_json::json!({"path":":application:verify","optionsComplete":false,"options":[]}),serde_json::json!({"path":":application:日本語 $ZED_FILE $(printf private-canary)","optionsComplete":false,"options":[]})]}else{vec![]},"configurationsKnown":true,"dependencyGraphKnown":false,
                                            "configurations":[{"name":"parityGraph","description":null,"resolvable":true,"consumable":false,"extendsFrom":[]},
                                            {"name":"runtime 📦 α","description":null,"resolvable":true,"consumable":false,"extendsFrom":[]},
                                            {"name":"apiElements","description":null,"resolvable":false,"consumable":true,"extendsFrom":[]}]});
                                        Ok(Some(serde_json::json!({"version":2,"rootUri":"file:///groovy-graph/","sourceSetsKnown":false,
                                            "projects":if provider==GraphFixtureProvider::Gradle {vec![serde_json::json!({"buildRootUri":"file:/groovy-graph/","moduleUri":"file:/groovy-graph/application/","projectPath": if version.load(Ordering::SeqCst)==2 {":changed"} else {":application"},"name":"application","tasks":if matches!(outcome, MavenGraphOutcome::OperationControls | MavenGraphOutcome::OrderedOperationControls) {vec![serde_json::json!({"path":":application:verify","name":"verify","group":"verification","description":null,"optionsKnown":false}),serde_json::json!({"path":":application:日本語 $ZED_FILE $(printf private-canary)","name":"日本語 $ZED_FILE $(printf private-canary)","group":null,"description":null,"optionsKnown":false})]}else{vec![]},"detailsJson":details.to_string()})]} else {vec![]}})))
                                    }
                                    "groovy.maven.importReactor" => {
                                        assert_eq!(parameters.arguments, [serde_json::json!({"rootUri":"file:///groovy-graph/","pomUri":"file:///groovy-graph/application/pom.xml","excludedProjectUris":[],"offline":true})]);
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":"file:///groovy-graph/","reactorPomUri":"file:///groovy-graph/application/pom.xml","importedPomUris":["file:///groovy-graph/application/pom.xml"],"excludedProjectUris":[],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false})))
                                    }
                                    "groovy.maven.dependencyGraph" | "groovy.gradle.dependencyGraph" => {
                                        let configuration = parameters.arguments[0]["configuration"].as_str().unwrap_or("parityGraph").to_owned();
                                        if provider == GraphFixtureProvider::Maven {
                                            assert_eq!(parameters.arguments,[serde_json::json!({"rootUri":"file:///groovy-graph/","pomUri":"file:///groovy-graph/application/pom.xml","offline":true,"cachedOnly":true})]);
                                        } else {
                                            assert_eq!(parameters.arguments,[serde_json::json!({"rootUri":"file:///groovy-graph/","moduleUri":"file:///groovy-graph/application/","configuration":configuration,"offline":true})]);
                                        }
                                        commands.fetch_add(1, Ordering::SeqCst);
                                        let held = gate.lock().expect("gate").take();if let Some(held)=held {held.await.expect("release graph");}
                                        if fail.load(Ordering::SeqCst) {anyhow::bail!("private-canary token=value /secret/provider-cache");}
                                        let mut response: serde_json::Value = serde_json::from_str(if provider == GraphFixtureProvider::Maven {include_str!("../../project/test_data/groovy_jvm_maven/dependency-graph-v2.json")} else {include_str!("../../project/test_data/groovy_jvm_gradle/dependency-graph-v1.json")}).expect("captured graph");
                                        if provider == GraphFixtureProvider::Gradle {
                                            let mut graph: serde_json::Value = serde_json::from_str(response["projects"][0]["graphJson"].as_str().expect("graph JSON")).expect("graph");
                                            graph["configuration"] = serde_json::json!(configuration);
                                            response["projects"][0]["graphJson"] = serde_json::json!(graph.to_string());
                                        }
                                        Ok(Some(response))
                                    }
                                    _ => anyhow::bail!("Unexpected graph fixture command"),
                                }
                            }
                        }
                    });
                }
            })), ..Default::default()
        });
        let (buffer, _open_handle) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/groovy-graph/application/Main.java", cx)
            })
            .await
            .expect("source");
        let mut provider_connections = vec![servers.next().await.expect("provider")];
        let pump = |cx: &mut TestAppContext| {
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
        };
        let refresh = |cx: &mut TestAppContext| {
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            pump(cx);
            if provider == GraphFixtureProvider::Gradle {
                lsp_store
                    .update(cx, |store, cx| {
                        store.synchronize_local_gradle_project(Some(worktree_id), cx)
                    })
                    .expect("explicit Gradle metadata synchronization");
                pump(cx);
                let error = lsp_store
                    .update(cx, |store, cx| {
                        store.query_local_gradle_dependency_graph(
                            worktree_id,
                            path(worktree_id, "application"),
                            "parityGraph".into(),
                            cx,
                        )
                    })
                    .expect_err("configuration requires a fresh ownership read");
                assert!(error.to_string().contains("ownership"));
                lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
                pump(cx);
            }
        };
        refresh(cx);
        if provider == GraphFixtureProvider::Maven && outcome == MavenGraphOutcome::Success {
            lsp_store
                .update(cx, |store, cx| {
                    store.import_local_maven_reactor(
                        worktree_id,
                        selected_project.clone(),
                        Vec::new(),
                        cx,
                    )
                })
                .expect("selected reactor import")
                .await
                .expect("phase receipt");
            pump(cx);
            lsp_store.read_with(cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("imported selection");
                assert!(current);
                assert_eq!(
                    snapshot
                        .metadata
                        .as_ref()
                        .expect("retained phase facts")
                        .imported_poms
                        .as_slice(),
                    std::slice::from_ref(&selected_project)
                );
            });
        }
        let original_server_id = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .expect("current metadata")
                .0
                .server_id
        });
        let client_server = lsp_store.read_with(cx, |store, _| {
            store
                .language_server_for_id(original_server_id)
                .expect("running client provider")
        });
        assert_eq!(
            commands.load(Ordering::SeqCst),
            0,
            "read/startup never collects graph"
        );
        lsp_store.read_with(cx, |store, _| {
            let (snapshot, current) = store.jdtls_project_snapshot(worktree_id).expect("metadata");
            assert!(current);
            match provider {
                GraphFixtureProvider::Maven => assert!(
                    snapshot
                        .projects
                        .iter()
                        .find_map(|project| project.maven.as_ref())
                        .expect("Maven metadata")
                        .dependency_graph
                        .is_none()
                ),
                GraphFixtureProvider::Gradle => assert!(
                    snapshot
                        .projects
                        .iter()
                        .find_map(|project| project.gradle.as_ref())
                        .expect("Gradle metadata")
                        .dependency_graph
                        .is_none()
                ),
            }
        });
        for invalid in [
            path(WorktreeId::from_usize(99), "application/pom.xml"),
            path(worktree_id, ".env/pom.xml"),
            path(worktree_id, "notpom.txt"),
            path(worktree_id, "missing/pom.xml"),
            path(worktree_id, "pom.xml"),
        ] {
            assert!(
                lsp_store
                    .update(cx, |store, cx| query_fixture_graph(
                        store,
                        worktree_id,
                        invalid,
                        cx,
                        provider,
                        "parityGraph"
                    ))
                    .is_err()
            );
        }
        if provider == GraphFixtureProvider::Gradle {
            for configuration in ["missing", "apiElements", "bad name"] {
                assert!(
                    lsp_store
                        .update(cx, |store, cx| query_fixture_graph(
                            store,
                            worktree_id,
                            selected_project.clone(),
                            cx,
                            provider,
                            configuration
                        ))
                        .is_err()
                );
            }
        }
        assert_eq!(commands.load(Ordering::SeqCst), 0);
        if outcome == MavenGraphOutcome::Unavailable {
            let error = lsp_store
                .update(cx, |store, cx| {
                    query_fixture_graph(
                        store,
                        worktree_id,
                        selected_project,
                        cx,
                        provider,
                        "parityGraph",
                    )
                })
                .expect_err("missing command");
            assert!(error.to_string().contains("extension setup"));
            assert!(lsp_store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)
            }));
            return;
        }
        if matches!(
            outcome,
            MavenGraphOutcome::OperationControls | MavenGraphOutcome::OrderedOperationControls
        ) {
            exercise_operation_controls(
                &project,
                &lsp_store,
                worktree_id,
                provider,
                &fs,
                &commands,
                &version,
                outcome == MavenGraphOutcome::OrderedOperationControls,
                cx,
            )
            .await;
            return;
        }
        if outcome == MavenGraphOutcome::PanelControls {
            let view = if provider == GraphFixtureProvider::Maven {
                BuildToolView::Maven
            } else {
                BuildToolView::Gradle
            };
            let (container, cx) = cx.add_window_view(|window, cx| {
                workspace::MultiWorkspace::test_new(project.clone(), window, cx)
            });
            let workspace = container.read_with(cx, |container, _| container.workspace().clone());
            let async_cx = cx.update(|window, cx| window.to_async(cx));
            let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
                .await
                .expect("mounted graph panel");
            panel.update(cx, |panel, cx| {
                panel.active = true;
                panel.view = view;
                panel.rebuild(cx);
            });
            workspace.update_in(cx, |workspace, window, cx| {
                workspace.add_panel(panel.clone(), window, cx);
                workspace
                    .right_dock()
                    .update(cx, |dock, cx| dock.set_open(true, window, cx));
                workspace.focus_panel::<DependenciesPanel>(window, cx);
            });
            pump(cx);
            let select = |cx: &mut TestAppContext| {
                let selected = lsp_store.read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("current panel data");
                    assert!(current);
                    let module = snapshot
                        .projects
                        .iter()
                        .find(|module| module.module_root.path.as_unix_str() == "application")
                        .expect("typed module");
                    let module_id = jdtls_module_node_id(snapshot, &module.module_root, view);
                    if provider == GraphFixtureProvider::Maven {
                        module_id
                    } else {
                        gradle_configuration_node_id(&module_id, "parityGraph")
                    }
                });
                panel.update(cx, |panel, cx| {
                    panel.host.expand_all();
                    panel.host.select(selected);
                    assert!(panel.selected_graph().is_some());
                    cx.notify();
                });
            };
            select(cx);
            if provider == GraphFixtureProvider::Gradle {
                panel.update(cx, |panel, cx| {
                    let invalid = panel
                        .host
                        .visible_rows()
                        .iter()
                        .find(|row| row.label == "apiElements")
                        .expect("nonresolvable declaration")
                        .id
                        .clone();
                    panel.host.select(invalid);
                    assert!(panel.selected_graph().is_none());
                    panel.collect_selected_graph(cx);
                });
                assert_eq!(
                    commands.load(Ordering::SeqCst),
                    0,
                    "nonresolvable selection cannot dispatch"
                );
                select(cx);
            }
            workspace.update(cx, |workspace, cx| workspace.clear_all_notifications(cx));
            assert_eq!(
                commands.load(Ordering::SeqCst),
                0,
                "mounting/selecting never resolves metadata"
            );
            cx.update(|window, _| window.activate_a11y_for_test());
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value =
                        serde_json::from_str(&window.debug_a11y_tree_json().expect("panel tree"))
                            .expect("accessibility JSON");
                    let nodes = tree["nodes"].as_object().expect("accessible nodes");
                    assert!(
                        nodes
                            .values()
                            .any(|node| node["aria"]["label"].as_str().is_some_and(
                                |label| label.starts_with("Collect offline dependency graph")
                            ))
                    );
                    assert!(nodes.values().any(|node| node["aria"]["label"]
                        == "Cancel only this panel's current dependency graph collection"));
                });
            }
            let mut collect_focused = false;
            for _ in 0..32 {
                collect_focused = cx.update(|window, cx| {
                    window.focus_next(cx);
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value = serde_json::from_str(
                        &window.debug_a11y_tree_json().expect("focused controls"),
                    )
                    .expect("a11y JSON");
                    let Some(focus) = tree["gpui_focus"].as_str() else {
                        return false;
                    };
                    tree["nodes"][focus]["aria"]["label"]
                        .as_str()
                        .is_some_and(|label| label.starts_with("Collect offline dependency graph"))
                });
                if collect_focused {
                    break;
                }
            }
            assert!(
                collect_focused,
                "shared tab traversal reaches named Collect control"
            );
            let press_button = |cx: &mut gpui::VisualTestContext, key: &str| {
                let keystroke = gpui::Keystroke::parse(key).expect("activation key");
                cx.update(|window, cx| {
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
                cx.run_until_parked();
            };
            press_button(cx, "enter");
            cx.run_until_parked();
            assert_eq!(commands.load(Ordering::SeqCst), 1);
            assert!(panel.read_with(cx, |panel, cx| panel.graph_query_is_pending(cx)));
            let mut cancel_focused = false;
            for _ in 0..32 {
                cancel_focused = cx.update(|window, cx| {
                    window.focus_next(cx);
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value =
                        serde_json::from_str(&window.debug_a11y_tree_json().expect("cancel focus"))
                            .expect("a11y JSON");
                    let Some(focus) = tree["gpui_focus"].as_str() else {
                        return false;
                    };
                    tree["nodes"][focus]["aria"]["label"]
                        == "Cancel only this panel's current dependency graph collection"
                });
                if cancel_focused {
                    break;
                }
            }
            assert!(
                cancel_focused,
                "shared tab traversal reaches owned Cancel control"
            );
            press_button(cx, "space");
            cx.run_until_parked();
            release.cancellation().await;
            assert!(
                release.is_canceled(),
                "panel cancellation drops only its owned request"
            );
            assert!(!panel.read_with(cx, |panel, cx| panel.graph_query_is_pending(cx)));
            refresh(cx);
            select(cx);
            workspace.update(cx, |workspace, cx| workspace.clear_all_notifications(cx));
            fail.store(true, Ordering::SeqCst);
            cx.dispatch_action(CollectSelectedGraph);
            pump(cx);
            assert_eq!(commands.load(Ordering::SeqCst), 2);
            assert!(
                lsp_store.read_with(cx, |store, _| store.jdtls_project_query_failed(worktree_id))
            );
            assert!(
                !workspace
                    .read_with(cx, |workspace, _| workspace.notification_ids())
                    .is_empty(),
                "provider failure must reach the workspace UI"
            );
            assert!(panel.read_with(cx, |panel, _| panel.graph_query_token.is_none()));
            panel.read_with(cx, |panel, _| {
                let notice = panel
                    .graph_notice
                    .as_deref()
                    .expect("actionable graph error");
                assert!(
                    notice.contains("refresh")
                        && !notice.contains("private-canary")
                        && !notice.contains("token=value")
                        && !notice.contains("provider-cache")
                );
            });
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value = serde_json::from_str(
                        &window.debug_a11y_tree_json().expect("graph failure status"),
                    )
                    .expect("a11y JSON");
                    let status = tree["nodes"]
                        .as_object()
                        .expect("nodes")
                        .values()
                        .find(|node| node["aria"]["label"] == "Dependency graph status")
                        .expect("named persistent graph status");
                    assert_eq!(status["aria"]["role"], "Status");
                    assert_eq!(status["aria"]["live"], "Polite");
                    assert!(status["aria"]["value"].as_str().is_some_and(|value| {
                        value.contains("refresh") && !value.contains("private-canary")
                    }));
                });
            }

            fail.store(false, Ordering::SeqCst);
            refresh(cx);
            select(cx);
            cx.dispatch_action(CollectSelectedGraph);
            pump(cx);
            assert_eq!(commands.load(Ordering::SeqCst), 3);
            panel.update(cx, |panel, cx| {
                panel.host.expand_all();
                cx.notify();
            });
            panel.read_with(cx, |panel, _| {
                assert!(panel.graph_query_token.is_none());
                assert!(
                    panel.graph_notice.is_none(),
                    "successful recovery clears the previous failure status"
                );
                assert!(
                    panel.host.visible_rows().iter().any(
                        |row| row.id.0.starts_with("graph:") && row.label == "Dependency edges"
                    )
                );
            });
            let inspection_target = lsp_store.read_with(cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("inspection snapshot");
                assert!(current);
                let module = snapshot
                    .projects
                    .iter()
                    .find(|module| module.module_root.path.as_unix_str() == "application")
                    .expect("inspection module");
                let id = jdtls_module_node_id(snapshot, &module.module_root, view);
                if provider == GraphFixtureProvider::Maven {
                    let graph = module
                        .maven
                        .as_ref()
                        .expect("Maven")
                        .dependency_graph
                        .as_ref()
                        .expect("Maven graph");
                    let node = graph
                        .nodes
                        .iter()
                        .find(|node| node.id != graph.root_node)
                        .expect("dependency occurrence");
                    graph_component_row_id(graph_fact_id(&id, "maven-node", &[&node.id]))
                } else {
                    let graph = module
                        .gradle
                        .as_ref()
                        .expect("Gradle")
                        .dependency_graph
                        .as_ref()
                        .expect("Gradle graph");
                    let component = graph
                        .resolution
                        .components
                        .iter()
                        .find(|component| component.id != graph.resolution.root_component)
                        .expect("dependency component");
                    graph_component_row_id(graph_fact_id(
                        &id,
                        "gradle-component",
                        &[&graph.resolution.configuration, &component.id],
                    ))
                }
            });
            panel.update(cx, |panel, cx| {
                assert!(
                    panel
                        .host
                        .visible_rows()
                        .iter()
                        .any(|row| row.id == inspection_target)
                );
                panel.host.select(inspection_target.clone());
                assert!(
                    panel
                        .selected_graph_inspection(GraphInspectionMode::Neighbours, cx)
                        .is_some()
                );
                cx.notify();
            });
            for (mode, label) in [
                (
                    GraphInspectionMode::Neighbours,
                    "Inspect selected dependency neighbours in admitted graph facts",
                ),
                (
                    GraphInspectionMode::ShortestRootPaths,
                    "Inspect all shortest root paths to the selected dependency in admitted graph facts",
                ),
            ] {
                let mut focused = false;
                for _ in 0..64 {
                    focused = cx.update(|window, cx| {
                        window.draw(cx).clear(cx);
                        let tree: serde_json::Value = serde_json::from_str(
                            &window.debug_a11y_tree_json().expect("inspection controls"),
                        )
                        .expect("inspection JSON");
                        if tree["gpui_focus"]
                            .as_str()
                            .is_some_and(|id| tree["nodes"][id]["aria"]["label"] == label)
                        {
                            true
                        } else {
                            window.focus_next(cx);
                            false
                        }
                    });
                    if focused {
                        break;
                    }
                }
                assert!(focused, "named shared inspection Button receives focus");
                press_button(cx, "enter");
                panel.read_with(cx, |panel, _| {
                    assert_eq!(
                        panel
                            .graph_inspection
                            .as_ref()
                            .expect("inspection active")
                            .mode,
                        mode
                    );
                    assert!(
                        panel
                            .host
                            .visible_rows()
                            .iter()
                            .any(|row| row.label == mode.label())
                    );
                    let ids = panel
                        .host
                        .visible_rows()
                        .iter()
                        .map(|row| &row.id)
                        .collect::<HashSet<_>>();
                    assert_eq!(
                        ids.len(),
                        panel.host.visible_rows().len(),
                        "inspection must not duplicate shared tree row identities"
                    );
                });
                for _ in 0..2 {
                    cx.update(|window, cx| {
                        window.draw(cx).clear(cx);
                        let tree: serde_json::Value = serde_json::from_str(
                            &window.debug_a11y_tree_json().expect("active inspection"),
                        )
                        .expect("inspection JSON");
                        assert!(tree["nodes"].as_object().expect("nodes").values().any(
                            |node| node["aria"]["label"] == label
                                && node["aria"]["role"] == "Button"
                                && node["aria"]["value"] == "Active"
                        ));
                    });
                }
                assert_eq!(
                    commands.load(Ordering::SeqCst),
                    3,
                    "inspection cannot query or execute the provider"
                );
            }
            panel.update(cx, |panel, cx| {
                panel.host.select_first();
                panel.inspect_selected_graph(GraphInspectionMode::Neighbours, cx);
                assert!(
                    panel.graph_inspection.is_some(),
                    "invalid selection preserves current inspection"
                );
                assert!(
                    panel
                        .graph_notice
                        .as_deref()
                        .is_some_and(|notice| notice.contains("select a dependency row"))
                );
                panel.host.select(inspection_target.clone());
                panel.inspect_selected_graph(GraphInspectionMode::ShortestRootPaths, cx);
                assert!(
                    panel.graph_notice.is_none(),
                    "valid inspection clears selection failure"
                );
            });
            let receipt = panel.read_with(cx, |panel, _| {
                panel.graph_inspection.clone().expect("current inspection")
            });
            let focus_diagram_control = |label: &str, cx: &mut gpui::VisualTestContext| {
                let mut focused = false;
                for _ in 0..64 {
                    focused = cx.update(|window, cx| {
                        window.draw(cx).clear(cx);
                        let tree: serde_json::Value = serde_json::from_str(
                            &window.debug_a11y_tree_json().expect("diagram control tree"),
                        )
                        .expect("diagram JSON");
                        if tree["gpui_focus"]
                            .as_str()
                            .is_some_and(|id| tree["nodes"][id]["aria"]["label"] == label)
                        {
                            true
                        } else {
                            window.focus_next(cx);
                            false
                        }
                    });
                    if focused {
                        break;
                    }
                }
                assert!(focused, "diagram control receives shared keyboard focus");
            };
            focus_diagram_control(
                "Show a bounded diagram of current admitted dependency graph facts",
                cx,
            );
            press_button(cx, "enter");
            panel.read_with(cx, |panel, cx| {
                let diagram = panel
                    .graph_diagram
                    .as_ref()
                    .expect("current shared Markdown diagram");
                let source = diagram.read(cx).source();
                assert!(source.contains(if provider == GraphFixtureProvider::Maven {
                    "Maven occurrence ancestry"
                } else {
                    "Gradle configuration graph"
                }));
                rendered_diagram(source.as_ref());
            });
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value = serde_json::from_str(
                        &window
                            .debug_a11y_tree_json()
                            .expect("diagram accessibility tree"),
                    )
                    .expect("diagram JSON");
                    assert!(
                        tree["nodes"]
                            .as_object()
                            .expect("nodes")
                            .values()
                            .any(|node| node["aria"]["label"] == "Current dependency diagram"
                                && node["aria"]["role"] == "Group")
                    );
                });
            }
            panel.update(cx, |panel, cx| {
                panel.graph_inspection = None;
                panel.rebuild(cx);
                assert!(
                    panel.graph_diagram_selection.is_some() && panel.graph_diagram.is_some(),
                    "diagram receipt is independent of path inspection"
                );
            });
            let diagram_entity = panel.read_with(cx, |panel, _| {
                panel
                    .graph_diagram
                    .as_ref()
                    .expect("diagram entity")
                    .entity_id()
            });
            for _ in 0..4 {
                cx.dispatch_action(CycleGraphFactFilter);
                panel.read_with(cx, |panel, _| {
                    assert_eq!(
                        panel
                            .graph_diagram
                            .as_ref()
                            .expect("unfiltered diagram remains current")
                            .entity_id(),
                        diagram_entity
                    )
                });
            }
            let diagram_receipt = panel.read_with(cx, |panel, _| {
                panel
                    .graph_diagram_selection
                    .clone()
                    .expect("current diagram receipt")
            });
            for changed in 0..3 {
                panel.update(cx, |panel, cx| {
                    let mut retired = diagram_receipt.clone();
                    match changed {
                        0 => retired.server_id = lsp::LanguageServerId(999_999),
                        1 => retired.generation += 1,
                        _ => match &mut retired.target {
                            GraphQueryTarget::Maven(pom) => {
                                pom.worktree_id = WorktreeId::from_usize(999)
                            }
                            GraphQueryTarget::Gradle { module, .. } => {
                                module.worktree_id = WorktreeId::from_usize(999)
                            }
                        },
                    }
                    panel.graph_diagram_selection = Some(retired);
                    panel.rebuild(cx);
                    assert!(
                        panel.graph_diagram_selection.is_none() && panel.graph_diagram.is_none(),
                        "revoked diagram does not remain visible"
                    );
                    panel.host.select(inspection_target.clone());
                    panel.show_dependency_diagram(cx);
                    assert!(panel.graph_diagram.is_some());
                });
            }
            focus_diagram_control("Close dependency diagram; retain current tree facts", cx);
            press_button(cx, "enter");
            panel.read_with(cx, |panel, _| {
                assert!(panel.graph_diagram_selection.is_none() && panel.graph_diagram.is_none())
            });
            assert_eq!(
                commands.load(Ordering::SeqCst),
                3,
                "show/close/current diagram must never query or execute provider"
            );
            panel.update(cx, |panel, cx| {
                panel.host.select(inspection_target.clone());
                panel.show_dependency_diagram(cx);
            });

            lsp_store.read_with(cx, |store, _| {
                let (snapshot, current) =
                    store.jdtls_project_snapshot(worktree_id).expect("snapshot");
                assert!(receipt.selection.matches(snapshot, current));
                assert!(!receipt.selection.matches(snapshot, false));
                let mut partial = snapshot.clone();
                for module in &mut partial.projects {
                    if let Some(maven) = &mut module.maven
                        && let Some(graph) = &mut maven.dependency_graph
                    {
                        let graph = Arc::make_mut(graph);
                        graph.collection_complete = false;
                        graph.structural_coverage_known = false;
                    }
                    if let Some(gradle) = &mut module.gradle
                        && let Some(graph) = &mut gradle.dependency_graph
                    {
                        let graph = Arc::make_mut(graph);
                        graph.resolution.resolution_complete = false;
                        graph.resolution.structural_coverage_known = false;
                    }
                }
                let source = dependency_diagram_source(&partial, true, view, &diagram_receipt)
                    .expect("partial graph projection")
                    .expect("partial graph diagram");
                assert!(source.contains("coverage false") && source.contains("complete false"));
                rendered_diagram(&source);
                let (_, row) = graph_inspection_row(&partial, view, &receipt)
                    .expect("partial graph remains inspectable");
                assert!(
                    row.accessibility_label
                        .contains("structural coverage false")
                );
                assert!(row.accessibility_label.contains("complete false"));
                assert!(
                    row.accessibility_label
                        .contains("source origins and artifacts unavailable")
                );
            });
            for changed in 0..4 {
                panel.update(cx, |panel, cx| {
                    let mut retired = receipt.clone();
                    match changed {
                        0 => retired.selection.server_id = lsp::LanguageServerId(999_999),
                        1 => retired.selection.generation += 1,
                        2 => match &mut retired.selection.target {
                            GraphQueryTarget::Maven(pom) => {
                                pom.worktree_id = WorktreeId::from_usize(999)
                            }
                            GraphQueryTarget::Gradle { module, .. } => {
                                module.worktree_id = WorktreeId::from_usize(999)
                            }
                        },
                        _ => retired.node = "unavailable-node-private-canary".into(),
                    }
                    panel.graph_inspection = Some(retired);
                    panel.rebuild(cx);
                    assert!(
                        panel.graph_inspection.is_none(),
                        "retired inspection binding clears"
                    );
                    assert!(
                        panel
                            .host
                            .visible_rows()
                            .iter()
                            .all(|row| row.label != GraphInspectionMode::ShortestRootPaths.label())
                    );
                });
            }
            panel.update_in(cx, |panel, window, cx| {
                panel.graph_inspection = Some(receipt.clone());
                panel.set_view(BuildToolView::Dependencies, window, cx);
                assert!(
                    panel.graph_inspection.is_none(),
                    "view change clears ephemeral inspection"
                );
                assert!(panel.graph_diagram.is_none() && panel.graph_diagram_selection.is_none());
                panel.set_view(view, window, cx);
            });
            cx.dispatch_action(ClearGraphInspection);
            panel.read_with(cx, |panel, _| assert!(panel.graph_inspection.is_none()));
            let before_filter = commands.load(Ordering::SeqCst);
            for _ in 0..32 {
                let focused = cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value =
                        serde_json::from_str(&window.debug_a11y_tree_json().expect("filter focus"))
                            .expect("a11y JSON");
                    let focused = tree["gpui_focus"]
                        .as_str()
                        .and_then(|identity| tree["nodes"][identity]["aria"]["label"].as_str());
                    if focused
                        .is_some_and(|label| label.starts_with("Dependency graph fact filter:"))
                    {
                        true
                    } else {
                        window.focus_next(cx);
                        false
                    }
                });
                if focused {
                    break;
                }
            }
            cx.update(|window, cx| {
                let keystroke = gpui::Keystroke::parse("enter").expect("Enter");
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
            panel.read_with(cx, |panel, _| {
                assert_eq!(panel.preferences.graph_fact_filter, GraphFactFilter::Direct)
            });
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value = serde_json::from_str(
                        &window.debug_a11y_tree_json().expect("filtered controls"),
                    )
                    .expect("a11y JSON");
                    assert!(tree["nodes"].as_object().expect("nodes").values().any(
                        |node| node["aria"]["role"] == "Button"
                            && node["aria"]["value"] == "Direct dependencies"
                            && node["aria"]["label"].as_str().is_some_and(|label| {
                                label.starts_with(
                                    "Dependency graph fact filter: Direct dependencies;",
                                )
                            })
                    ));
                });
            }
            for _ in 0..3 {
                cx.dispatch_action(CycleGraphFactFilter);
            }
            panel.read_with(cx, |panel, _| {
                assert_eq!(panel.preferences.graph_fact_filter, GraphFactFilter::All)
            });
            assert_eq!(
                commands.load(Ordering::SeqCst),
                before_filter,
                "filter changes must not dispatch provider queries"
            );
            panel.update(cx, |panel, cx| {
                panel.host.expand_all();
                cx.notify();
            });
            panel.update(cx, |panel, cx| {
                panel.host.collapse_all();
                cx.notify();
            });
            if provider == GraphFixtureProvider::Maven {
                for _ in 0..40 {
                    let focused = cx.update(|window, cx| {
                        window.draw(cx).clear(cx);
                        let tree: serde_json::Value = serde_json::from_str(
                            &window.debug_a11y_tree_json().expect("scope focus"),
                        )
                        .expect("scope tree");
                        let focused = tree["gpui_focus"]
                            .as_str()
                            .and_then(|identity| tree["nodes"][identity]["aria"]["label"].as_str());
                        if focused
                            .is_some_and(|label| label.starts_with("Maven dependency graph scope:"))
                        {
                            true
                        } else {
                            window.focus_next(cx);
                            false
                        }
                    });
                    if focused {
                        break;
                    }
                }
                press_button(cx, "enter");
                panel.read_with(cx, |panel, _| {
                    assert_eq!(
                        panel.preferences.maven_graph_scope,
                        MavenGraphScope::Compile
                    )
                });
                for _ in 0..2 {
                    cx.update(|window, cx| {
                        window.draw(cx).clear(cx);
                        let tree: serde_json::Value = serde_json::from_str(
                            &window.debug_a11y_tree_json().expect("scope metadata"),
                        )
                        .expect("scope tree");
                        assert!(tree["nodes"].as_object().expect("nodes").values().any(
                            |node| node["aria"]["role"] == "Button"
                                && node["aria"]["value"] == "compile"
                                && node["aria"]["label"].as_str().is_some_and(|label| {
                                    label.starts_with("Maven dependency graph scope: compile;")
                                })
                        ));
                    });
                }
                for _ in 0..7 {
                    cx.dispatch_action(CycleMavenGraphScope);
                }
                panel.read_with(cx, |panel, _| {
                    assert_eq!(panel.preferences.maven_graph_scope, MavenGraphScope::All)
                });
                assert_eq!(
                    commands.load(Ordering::SeqCst),
                    before_filter,
                    "scope controls must not query the provider"
                );
            }
            let before_search = commands.load(Ordering::SeqCst);
            let search_editor = panel.read_with(cx, |panel, _| panel.graph_search.clone());
            cx.update(|window, cx| window.focus(&search_editor.read(cx).focus_handle(cx), cx));
            cx.simulate_keystrokes("S H A R E D");
            cx.run_until_parked();
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value =
                        serde_json::from_str(&window.debug_a11y_tree_json().expect("graph search"))
                            .expect("a11y JSON");
                    let nodes = tree["nodes"].as_object().expect("nodes");
                    let (identity, search) = nodes
                        .iter()
                        .find(|(_, node)| {
                            node["aria"]["role"] == "SearchInput"
                                && node["aria"]["label"] == "Search admitted dependency graph facts"
                        })
                        .expect("named shared editor search");
                    assert_eq!(search["aria"]["value"], "SHARED");
                    assert_eq!(tree["gpui_focus"].as_str(), Some(identity.as_str()));
                    assert!(nodes.values().any(|node| node["aria"]["label"]
                        == "Clear dependency graph fact search"
                        && node["aria"]["disabled"] == false));
                });
            }
            panel.read_with(cx, |panel, _| {
                assert!(
                    panel
                        .host
                        .visible_rows()
                        .iter()
                        .any(|row| row.label.contains("shared"))
                )
            });
            search_editor.update_in(cx, |editor, window, cx| {
                editor.set_text("never-present-private-canary", window, cx)
            });
            cx.run_until_parked();
            panel.read_with(cx, |panel, _| {
                assert!(panel.host.visible_rows().iter().any(|row| {
                    row.accessibility_label
                        .contains("No matching admitted graph facts")
                }))
            });
            search_editor.update_in(cx, |editor, window, cx| {
                editor.set_text("α".repeat(MAX_GRAPH_SEARCH_BYTES / 2 + 1), window, cx)
            });
            cx.run_until_parked();
            panel.read_with(cx, |panel, _| {
                assert!(matches!(
                    panel.host.status(),
                    language_tool_tree::LanguageToolTreeStatus::Error(_)
                        | language_tool_tree::LanguageToolTreeStatus::StaleError(_)
                ));
                assert!(
                    panel
                        .host
                        .visible_rows()
                        .iter()
                        .all(|row| !row.id.0.contains("-edge:"))
                );
            });
            cx.dispatch_action(ClearGraphFactSearch);
            cx.run_until_parked();
            assert!(search_editor.read_with(cx, |editor, cx| editor.text(cx).is_empty()));
            assert_eq!(
                commands.load(Ordering::SeqCst),
                before_search,
                "search and clear never query provider"
            );
            panel.update(cx, |panel, cx| {
                panel.host.expand_all();
                cx.notify();
            });
            let items = workspace.read_with(cx, |workspace, cx| {
                workspace.active_pane().read(cx).items().count()
            });
            panel.update(cx, |panel, cx| {
                let edge = panel
                    .host
                    .visible_rows()
                    .iter()
                    .find(|row| row.id.0.contains("-edge:"))
                    .expect("graph edge")
                    .id
                    .clone();
                panel.host.select(edge);
                cx.notify();
            });
            cx.dispatch_action(language_tool_tree::ActivateSelected);
            assert_eq!(
                workspace.read_with(cx, |workspace, cx| workspace
                    .active_pane()
                    .read(cx)
                    .items()
                    .count()),
                items,
                "missing origin cannot open another file"
            );
            panel.update(cx, |panel, cx| {
                panel.host.select(inspection_target);
                panel.inspect_selected_graph(GraphInspectionMode::ShortestRootPaths, cx);
                assert!(panel.graph_inspection.is_some());
                panel.show_dependency_diagram(cx);
                assert!(panel.graph_diagram.is_some());
            });
            select(cx);
            let captured = panel.read_with(cx, |panel, _| {
                (
                    panel.host.selected().cloned().expect("selected target"),
                    panel.selected_graph().expect("capture target"),
                )
            });
            fs.insert_file(
                std::path::Path::new(if provider == GraphFixtureProvider::Maven {
                    "/groovy-graph/application/pom.xml"
                } else {
                    "/groovy-graph/application/build.gradle"
                }),
                b"changed externally".to_vec(),
            )
            .await;
            cx.run_until_parked();
            panel.read_with(cx, |panel, _| {
                assert!(
                    panel.graph_inspection.is_none(),
                    "external input invalidates inspection"
                );
                assert!(
                    panel.graph_diagram.is_none() && panel.graph_diagram_selection.is_none(),
                    "external build changes invalidate the diagram receipt"
                )
            });
            panel.update(cx, |panel, cx| {
                panel
                    .graph_selections
                    .insert(captured.0.clone(), captured.1.clone());
                panel.host.select(captured.0);
                panel.collect_selected_graph(cx);
            });
            cx.run_until_parked();
            assert_eq!(
                commands.load(Ordering::SeqCst),
                3,
                "stale captured target must not dispatch"
            );
            trust.update(cx, |trusted, cx| {
                trusted.restrict(
                    worktree_store.downgrade(),
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                )
            });
            cx.run_until_parked();
            panel.read_with(cx, |panel, _| {
                assert!(panel.graph_selections.is_empty());
                assert!(panel.graph_inspection.is_none());
                assert!(panel.graph_diagram.is_none() && panel.graph_diagram_selection.is_none());
                assert!(panel.host.visible_rows().is_empty());
            });
            cx.dispatch_action(CollectSelectedGraph);
            cx.run_until_parked();
            assert_eq!(
                commands.load(Ordering::SeqCst),
                3,
                "revoked trust cannot resolve metadata"
            );
            return;
        }
        let generation = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .expect("snapshot")
                .0
                .generation
        });
        let query = lsp_store
            .update(cx, |store, cx| {
                query_fixture_graph(
                    store,
                    worktree_id,
                    selected_project.clone(),
                    cx,
                    provider,
                    "parityGraph",
                )
            })
            .expect("selected query");
        cx.run_until_parked();
        assert_eq!(commands.load(Ordering::SeqCst), 1);
        assert!(lsp_store.read_with(cx, |store, _| {
            store.jdtls_project_query_pending(worktree_id)
        }));
        assert!(lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(_, current)| !current)
        }));
        let first_query_token = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_query_token(worktree_id)
                .expect("owned current query token")
        });
        for (server_id, token_worktree) in [
            (original_server_id, WorktreeId::from_usize(99)),
            (lsp::LanguageServerId(999_999), worktree_id),
        ] {
            let mut unrelated = project::jvm_project::JdtlsProjectState::default();
            let token = unrelated
                .begin(server_id, token_worktree)
                .expect("unrelated token");
            assert!(
                !lsp_store
                    .update(cx, |store, cx| {
                        store.cancel_local_jdtls_project_query_for_token(token, cx)
                    })
                    .expect("foreign token must not cancel owned work")
            );
            assert_eq!(
                lsp_store.read_with(cx, |store, _| store.jdtls_project_query_token(worktree_id)),
                Some(first_query_token)
            );
        }
        match outcome {
            MavenGraphOutcome::Cancellation => {
                assert!(
                    lsp_store
                        .update(cx, |store, cx| {
                            store.cancel_local_jdtls_project_query_for_token(first_query_token, cx)
                        })
                        .expect("cancel query")
                );
                assert!(
                    !lsp_store
                        .update(cx, |store, cx| {
                            store.cancel_local_jdtls_project_query_for_token(first_query_token, cx)
                        })
                        .expect("repeated cancellation must be harmless")
                );
            }
            MavenGraphOutcome::InputChange => {
                fs.insert_file(
                    std::path::Path::new(if provider == GraphFixtureProvider::Maven {
                        "/groovy-graph/application/pom.xml"
                    } else {
                        "/groovy-graph/application/build.gradle"
                    }),
                    b"<project>changed</project>".to_vec(),
                )
                .await;
                cx.run_until_parked();
            }
            MavenGraphOutcome::Restart => {
                let stop = lsp_store.update(cx, |store, cx| {
                    store.stop_language_servers_for_buffers(
                        vec![buffer.clone()],
                        HashSet::default(),
                        cx,
                    )
                });
                cx.run_until_parked();
                stop.await.expect("stop old provider");
            }
            MavenGraphOutcome::Revocation => {
                trust.update(cx, |trusted, cx| {
                    trusted.restrict(
                        worktree_store.downgrade(),
                        HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                        cx,
                    )
                });
                cx.run_until_parked();
            }
            MavenGraphOutcome::OtherScope => {
                let folder = "file:///unrelated-worktree/".parse().expect("folder");
                client_server.add_workspace_folder(folder);
                assert!(
                    client_server
                        .workspace_folders()
                        .iter()
                        .any(|folder| folder.as_str() == "file:///unrelated-worktree/")
                );
            }
            MavenGraphOutcome::ScopeChange => {
                let folder = "file:///groovy-graph/additional/".parse().expect("folder");
                client_server.add_workspace_folder(folder);
                assert!(
                    client_server
                        .workspace_folders()
                        .iter()
                        .any(|folder| folder.as_str() == "file:///groovy-graph/additional/"),
                    "fixture must actually change initialized scope"
                );
            }
            _ => {}
        }
        let cancelled = matches!(
            outcome,
            MavenGraphOutcome::Cancellation
                | MavenGraphOutcome::InputChange
                | MavenGraphOutcome::Restart
                | MavenGraphOutcome::Revocation
        );
        if cancelled {
            assert!(query.await.is_err());
            release.cancellation().await;
            assert!(release.is_canceled());
        } else {
            release.send(()).expect("release graph");
            if matches!(
                outcome,
                MavenGraphOutcome::Success | MavenGraphOutcome::OtherScope
            ) {
                query.await.expect("admitted graph");
                lsp_store.read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("graph snapshot");
                    assert!(current && snapshot.generation > generation);
                    if provider == GraphFixtureProvider::Maven
                        && outcome == MavenGraphOutcome::Success
                    {
                        assert_eq!(
                            snapshot
                                .metadata
                                .as_ref()
                                .expect("graph retains current phase facts")
                                .imported_poms
                                .as_slice(),
                            std::slice::from_ref(&selected_project)
                        );
                    }
                    let view = if provider == GraphFixtureProvider::Maven {
                        BuildToolView::Maven
                    } else {
                        BuildToolView::Gradle
                    };
                    let (roots, navigation) =
                        project_rows_for_view(snapshot, view, &BuildViewPreferences::default());
                    let mut rows = vec![roots.as_slice()];
                    let mut graph_rows = 0;
                    while let Some(children) = rows.pop() {
                        for row in children {
                            if row.id.0.starts_with("graph:") {
                                graph_rows += 1;
                                assert!(!row.enabled && row.activation_label.is_none());
                                assert!(
                                    !navigation.contains_key(&row.id),
                                    "Missing origins must not gain panel actions"
                                );
                            }
                            rows.push(row.children.as_slice());
                        }
                    }
                    assert!(
                        graph_rows > 3,
                        "Admitted provider graph must reach the actual panel projection"
                    );
                    match provider {
                        GraphFixtureProvider::Maven => {
                            let maven = snapshot
                                .projects
                                .iter()
                                .find_map(|project| project.maven.as_ref())
                                .expect("Maven");
                            let graph = maven.dependency_graph.as_ref().expect("admitted graph");
                            assert_eq!(graph.nodes.len(), 15);
                            assert!(
                                !graph.origins_known
                                    && !graph.artifacts_known
                                    && !graph.cycle_occurrences_known
                            );
                            assert!(Arc::ptr_eq(
                                graph,
                                maven
                                    .clone()
                                    .dependency_graph
                                    .as_ref()
                                    .expect("shared immutable graph")
                            ));
                        }
                        GraphFixtureProvider::Gradle => {
                            let gradle = snapshot
                                .projects
                                .iter()
                                .find_map(|project| project.gradle.as_ref())
                                .expect("Gradle");
                            let graph = gradle.dependency_graph.as_ref().expect("admitted graph");
                            assert!(
                                snapshot.partial
                                    && graph.resolution.structural_coverage_known
                                    && !graph.resolution.resolution_complete
                            );
                            assert_eq!(graph.resolution.configuration, "parityGraph");
                            assert_eq!(graph.resolution.components.len(), 8);
                            assert!(
                                !graph.resolution.origins_known
                                    && !graph.resolution.artifacts_known
                            );
                            assert!(Arc::ptr_eq(
                                graph,
                                gradle
                                    .clone()
                                    .dependency_graph
                                    .as_ref()
                                    .expect("shared immutable graph")
                            ));
                        }
                    }
                });
            } else {
                let error = query.await.expect_err("rejected graph");
                let message = format!("{error:#}");
                assert!(
                    message.contains("refresh")
                        && !message.contains("private-canary")
                        && !message.contains("token=value")
                        && !message.contains("provider-cache")
                );
                assert!(!lsp_store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }));
            }
        }
        if matches!(
            outcome,
            MavenGraphOutcome::Success | MavenGraphOutcome::OtherScope
        ) {
            let second = lsp_store
                .update(cx, |store, cx| {
                    query_fixture_graph(
                        store,
                        worktree_id,
                        selected_project,
                        cx,
                        provider,
                        if provider == GraphFixtureProvider::Gradle {
                            "runtime 📦 α"
                        } else {
                            "parityGraph"
                        },
                    )
                })
                .expect("second query with advanced receipt");
            let second_query_token = lsp_store.read_with(cx, |store, _| {
                store
                    .jdtls_project_query_token(worktree_id)
                    .expect("new pending query token")
            });
            assert_ne!(first_query_token, second_query_token);
            assert!(
                !lsp_store
                    .update(cx, |store, cx| {
                        store.cancel_local_jdtls_project_query_for_token(first_query_token, cx)
                    })
                    .expect("completed token must not cancel newer query")
            );
            assert_eq!(
                lsp_store.read_with(cx, |store, _| store.jdtls_project_query_token(worktree_id)),
                Some(second_query_token)
            );
            second.await.expect("second admitted graph");
            assert_eq!(commands.load(Ordering::SeqCst), 2);
            if provider == GraphFixtureProvider::Gradle {
                lsp_store.read_with(cx, |store, _| {
                    let graph = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("current configuration")
                        .0
                        .projects
                        .iter()
                        .find_map(|project| project.gradle.as_ref())
                        .expect("Gradle")
                        .dependency_graph
                        .as_ref()
                        .expect("selected graph");
                    assert_eq!(graph.resolution.configuration, "runtime 📦 α");
                });
                lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
                pump(cx);
                lsp_store.read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("read refreshed metadata");
                    assert!(current);
                    assert!(
                        snapshot
                            .projects
                            .iter()
                            .find_map(|project| project.gradle.as_ref())
                            .expect("Gradle metadata")
                            .dependency_graph
                            .is_none()
                    );
                });
                assert_eq!(
                    commands.load(Ordering::SeqCst),
                    2,
                    "read refresh must discard graph without recollecting"
                );
            }
            return;
        }
        fail.store(false, Ordering::SeqCst);
        version.store(1, Ordering::SeqCst);
        if outcome == MavenGraphOutcome::ScopeChange {
            client_server.remove_workspace_folder(
                "file:///groovy-graph/additional/".parse().expect("folder"),
            );
        }
        if outcome == MavenGraphOutcome::Revocation {
            assert!(
                lsp_store
                    .update(cx, |store, cx| query_fixture_graph(
                        store,
                        worktree_id,
                        selected_project.clone(),
                        cx,
                        provider,
                        "parityGraph"
                    ))
                    .is_err()
            );
            trust.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                )
            });
        }
        if matches!(
            outcome,
            MavenGraphOutcome::Restart | MavenGraphOutcome::Revocation
        ) {
            lsp_store.update(cx, |store, cx| {
                store.restart_language_servers_for_buffers(
                    vec![buffer],
                    HashSet::default(),
                    true,
                    cx,
                )
            });
            cx.run_until_parked();
            provider_connections.push(servers.next().await.expect("replacement provider"));
        }
        refresh(cx);
        pump(cx);
        if matches!(
            outcome,
            MavenGraphOutcome::Restart | MavenGraphOutcome::Revocation
        ) {
            let replacement_id = lsp_store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("replacement metadata")
                    .0
                    .server_id
            });
            assert_ne!(replacement_id, original_server_id);
        }
        assert_eq!(
            commands.load(Ordering::SeqCst),
            1,
            "read/restart must not recollect graph"
        );
        assert!(lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(snapshot, current)| {
                    current
                        && snapshot.projects.iter().all(|project| {
                            project
                                .maven
                                .as_ref()
                                .is_none_or(|maven| maven.dependency_graph.is_none())
                                && project
                                    .gradle
                                    .as_ref()
                                    .is_none_or(|gradle| gradle.dependency_graph.is_none())
                        })
                })
        }));
        let recovery = lsp_store
            .update(cx, |store, cx| {
                query_fixture_graph(
                    store,
                    worktree_id,
                    selected_project,
                    cx,
                    provider,
                    "parityGraph",
                )
            })
            .expect("explicit retry");
        let recovery_token = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_query_token(worktree_id)
                .expect("recovery token")
        });
        assert!(
            !lsp_store
                .update(cx, |store, cx| {
                    store.cancel_local_jdtls_project_query_for_token(first_query_token, cx)
                })
                .expect("old token cannot cancel repaired/replaced provider query")
        );
        assert_eq!(
            lsp_store.read_with(cx, |store, _| store.jdtls_project_query_token(worktree_id)),
            Some(recovery_token)
        );
        recovery.await.expect("recovered graph");
        assert_eq!(commands.load(Ordering::SeqCst), 2);
    }

    #[gpui::test]
    async fn saved_gradle_startup_replays_once_and_restart_uses_current_settings(
        cx: &mut TestAppContext,
    ) {
        saved_gradle_startup_lifecycle(cx, None, StartupFixtureOutcome::Success).await;
    }

    #[gpui::test]
    async fn saved_mixed_startup_serializes_maven_and_gradle_intents(cx: &mut TestAppContext) {
        saved_gradle_startup_lifecycle(
            cx,
            Some(&["maven/pom.xml"]),
            StartupFixtureOutcome::Success,
        )
        .await;
    }

    #[gpui::test]
    async fn saved_empty_maven_startup_does_not_block_gradle(cx: &mut TestAppContext) {
        saved_gradle_startup_lifecycle(cx, Some(&[]), StartupFixtureOutcome::Success).await;
    }

    #[gpui::test]
    async fn saved_mixed_startup_failure_discards_pending_gradle(cx: &mut TestAppContext) {
        saved_gradle_startup_lifecycle(
            cx,
            Some(&["maven/pom.xml"]),
            StartupFixtureOutcome::Failure,
        )
        .await;
    }

    #[gpui::test]
    async fn saved_mixed_startup_cancellation_discards_pending_gradle(cx: &mut TestAppContext) {
        saved_gradle_startup_lifecycle(
            cx,
            Some(&["maven/pom.xml"]),
            StartupFixtureOutcome::Cancellation,
        )
        .await;
    }

    #[gpui::test]
    async fn saved_gradle_startup_revocation_discards_unapplied_intent(cx: &mut TestAppContext) {
        saved_gradle_startup_lifecycle(cx, None, StartupFixtureOutcome::Revocation).await;
    }

    #[gpui::test]
    async fn saved_mixed_startup_read_failure_discards_pending_gradle(cx: &mut TestAppContext) {
        saved_gradle_startup_lifecycle(
            cx,
            Some(&["maven/pom.xml"]),
            StartupFixtureOutcome::ReadFailure,
        )
        .await;
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum StartupFixtureOutcome {
        Success,
        Failure,
        ReadFailure,
        Cancellation,
        Revocation,
    }

    async fn saved_gradle_startup_lifecycle(
        cx: &mut TestAppContext,
        maven_selection: Option<&[&str]>,
        outcome: StartupFixtureOutcome,
    ) {
        use settings::SettingsStore;
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/saved-gradle",
            serde_json::json!({
                "build.gradle": "plugins { id 'java' }", "Main.java": "class Main {}",
                "maven": { "pom.xml": "<project/>" }
            }),
        )
        .await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/saved-gradle".as_ref()], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktree_store.read_with(cx, |store, cx| {
            store.worktrees().next().expect("worktree").read(cx).id()
        });
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        let set_user = |cx: &mut TestAppContext, settings: &str| {
            cx.update(|cx| {
                cx.update_global(|store: &mut SettingsStore, cx| {
                    store
                        .set_user_settings(settings, cx)
                        .result()
                        .expect("startup settings");
                })
            });
        };
        let root_settings = r#"{"jvm_builds":{"gradle":{"selected_build_files":["build.gradle"],"ignored_projects":[]}}}"#;
        let mut initial_settings: serde_json::Value =
            serde_json::from_str(root_settings).expect("root settings");
        if let Some(selection) = maven_selection {
            initial_settings["jvm_builds"]["maven"] =
                serde_json::json!({"selected_poms":selection,"ignored_projects":[]});
        }
        set_user(cx, &initial_settings.to_string());
        cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"))
            .update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                )
            });
        let commands = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let active = Arc::new(AtomicUsize::new(0));
        let fail = Arc::new(AtomicBool::new(false));
        let fail_read = Arc::new(AtomicBool::new(false));
        let (release, held) = oneshot::channel();
        let gate = Arc::new(std::sync::Mutex::new(Some(held)));
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let mut servers = languages.register_fake_lsp("Java", FakeLspAdapter {
            name: "jdtls", capabilities: lsp::ServerCapabilities {
                execute_command_provider: Some(lsp::ExecuteCommandOptions {
                    commands: ["java.project.getAll", "java.project.getSettings", "groovy.maven.importedProjects", "groovy.gradle.importedProjects", "groovy.maven.importReactor", "groovy.gradle.importSelected", "groovy.gradle.rootState", "groovy.gradle.applyRootState", "groovy.gradle.moduleStates", "groovy.gradle.applyModuleStates"].into_iter().map(str::to_owned).collect(),
                    ..Default::default()
                }), ..Default::default()
            }, initializer: Some(Box::new({
                let commands = commands.clone(); let active = active.clone(); let fail = fail.clone(); let fail_read = fail_read.clone(); let gate = gate.clone();
                move |server| {
                    server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                        let commands = commands.clone(); let active = active.clone(); let fail = fail.clone(); let fail_read = fail_read.clone(); let gate = gate.clone();
                        move |parameters, _| {
                            let commands = commands.clone(); let active = active.clone(); let fail = fail.clone(); let fail_read = fail_read.clone(); let gate = gate.clone();
                            async move {
                                match parameters.command.as_str() {
                                    "java.project.getAll" => {
                                        if fail_read.load(Ordering::SeqCst) { anyhow::bail!("private-read-canary at /secret/provider-cache"); }
                                        Ok(Some(serde_json::json!(["file:///saved-gradle/", "file:///saved-gradle/maven/"])))
                                    },
                                    "java.project.getSettings" => Ok(Some(serde_json::json!({"org.eclipse.jdt.ls.core.sourcePaths":[],"org.eclipse.jdt.ls.core.classpathEntries":[]}))),
                                    "groovy.maven.importedProjects" | "groovy.gradle.importedProjects" => {
                                        assert_eq!(parameters.arguments, [serde_json::json!({"rootUri":"file:///saved-gradle/","cachedOnly":true})]);
                                        let uri = if parameters.command == "groovy.maven.importedProjects" {"file:///saved-gradle/maven/"} else {"file:///saved-gradle/"};
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":"file:///saved-gradle/","projects":[{"projectUri":uri,"javaNature":true}],"partial":false,"fullSynchronizationKnown":false})))
                                    }
                                    "groovy.gradle.rootState" => Ok(Some(serde_json::json!({"version":1,"rootUri":parameters.arguments[0]["rootUri"],"workspaceState":"open","descriptionAvailable":true,"fullSynchronizationKnown":false}))),
                                    "groovy.gradle.moduleStates" => Ok(Some(serde_json::json!({"version":1,"rootUri":parameters.arguments[0]["rootUri"],"modules":[],"fullSynchronizationKnown":false}))),
                                    "groovy.maven.importReactor" | "groovy.gradle.importSelected" => {
                                        assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0, "startup providers must share one active phase");
                                        commands.lock().expect("commands").push(parameters.command.clone());
                                        let held = gate.lock().expect("gate").take();
                                        if let Some(held) = held { held.await.expect("release startup phase"); }
                                        assert_eq!(active.fetch_sub(1, Ordering::SeqCst), 1);
                                        if fail.load(Ordering::SeqCst) { anyhow::bail!("private-startup-canary at /secret/provider-cache"); }
                                        assert_eq!(parameters.arguments.len(), 1);
                                        assert_eq!(parameters.arguments[0]["rootUri"], "file:///saved-gradle/");
                                        assert_eq!(parameters.arguments[0]["offline"], true);
                                        let mut receipt = serde_json::json!({"version":1,"rootUri":parameters.arguments[0]["rootUri"],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false});
                                        if parameters.command == "groovy.maven.importReactor" {
                                            assert_eq!(parameters.arguments[0]["pomUri"], "file:///saved-gradle/maven/pom.xml");
                                            receipt["reactorPomUri"] = parameters.arguments[0]["pomUri"].clone();
                                            receipt["importedPomUris"] = serde_json::json!([parameters.arguments[0]["pomUri"]]);
                                            receipt["excludedProjectUris"] = parameters.arguments[0]["excludedProjectUris"].clone();
                                        }
                                        Ok(Some(receipt))
                                    }
                                    _ => anyhow::bail!("Unexpected startup fixture command"),
                                }
                            }
                        }
                    });
                }
            })), ..Default::default()
        });
        let (buffer, _open_handle) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/saved-gradle/Main.java", cx)
            })
            .await
            .expect("Java source");
        let mut server = servers.next().await.expect("startup provider");
        let pump = |cx: &mut TestAppContext| {
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
        };
        if outcome == StartupFixtureOutcome::Revocation {
            cx.run_until_parked();
            let trust = cx.update(|cx| cx.remove_global::<TrustedWorktrees>());
            pump(cx);
            assert!(commands.lock().expect("commands").is_empty());
            cx.update(|cx| cx.set_global(trust));
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            pump(cx);
            pump(cx);
            assert!(
                commands.lock().expect("commands").is_empty(),
                "restored permission plus read must not replay revoked startup intent"
            );
            return;
        }
        pump(cx);
        let first = if maven_selection.is_some_and(|selection| !selection.is_empty()) {
            "groovy.maven.importReactor"
        } else {
            "groovy.gradle.importSelected"
        };
        assert_eq!(*commands.lock().expect("commands"), [first]);
        assert_eq!(active.load(Ordering::SeqCst), 1);
        assert!(lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(_, current)| !current)
        }));
        assert!(!lsp_store.read_with(cx, |store, _| {
            store.jdtls_project_query_pending(worktree_id)
        }));
        lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
        pump(cx);
        assert_eq!(
            commands.lock().expect("commands").len(),
            1,
            "reads cannot replace or repeat startup mutation"
        );
        if outcome == StartupFixtureOutcome::Failure {
            fail.store(true, Ordering::SeqCst);
        } else if outcome == StartupFixtureOutcome::Cancellation {
            assert!(
                lsp_store
                    .update(cx, |store, cx| store
                        .cancel_selected_maven_import(worktree_id, cx))
                    .expect("cancel owned startup phase")
            );
        }
        if outcome == StartupFixtureOutcome::ReadFailure {
            fail_read.store(true, Ordering::SeqCst);
        }
        release.send(()).expect("release owned startup phase");
        cx.run_until_parked();
        pump(cx);
        pump(cx);
        if outcome != StartupFixtureOutcome::Success {
            assert_eq!(*commands.lock().expect("commands"), [first]);
            assert_eq!(active.load(Ordering::SeqCst), 0);
            assert!(lsp_store.read_with(cx, |store, _| {
                if outcome == StartupFixtureOutcome::ReadFailure {
                    store.jdtls_project_query_failed(worktree_id)
                } else {
                    store.jdtls_project_import_failed(worktree_id)
                }
            }));
            fail.store(false, Ordering::SeqCst);
            fail_read.store(false, Ordering::SeqCst);
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            pump(cx);
            pump(cx);
            assert_eq!(
                *commands.lock().expect("commands"),
                [first],
                "failure/cancellation discards the other provider's replay"
            );
            return;
        }
        let expected = if first == "groovy.maven.importReactor" {
            vec![first, "groovy.gradle.importSelected"]
        } else {
            vec![first]
        };
        assert_eq!(*commands.lock().expect("commands"), expected);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert!(!lsp_store.read_with(cx, |store, _| {
            store.jdtls_project_query_pending(worktree_id)
        }));
        for _ in 0..2 {
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            pump(cx);
            assert_eq!(
                *commands.lock().expect("commands"),
                expected,
                "startup intent consumed once"
            );
        }
        if maven_selection.is_some() {
            return;
        }
        for (settings, replacement, expected_calls, expected_failure) in [
            (root_settings, None, 1, false),
            ("{}", None, 0, false),
            (
                root_settings,
                Some(r#"{"jvm_builds":{"gradle":{"selected_build_files":[]}}}"#),
                0,
                true,
            ),
            (
                r#"{"jvm_builds":{"gradle":{"selected_build_files":["settings.gradle.kts"]}}}"#,
                None,
                0,
                true,
            ),
        ] {
            commands.lock().expect("commands").clear();
            set_user(cx, settings);
            lsp_store.update(cx, |store, cx| {
                store.restart_language_servers_for_buffers(
                    vec![buffer.clone()],
                    HashSet::default(),
                    true,
                    cx,
                )
            });
            cx.run_until_parked();
            while let Some(Some(replacement_server)) = servers.next().now_or_never() {
                server = replacement_server;
            }
            if let Some(replacement) = replacement {
                set_user(cx, replacement);
            }
            pump(cx);
            pump(cx);
            assert_eq!(commands.lock().expect("commands").len(), expected_calls);
            assert_eq!(
                lsp_store.read_with(cx, |store, _| store
                    .jdtls_project_import_failed(worktree_id)),
                expected_failure
            );
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            pump(cx);
            assert_eq!(
                commands.lock().expect("commands").len(),
                expected_calls,
                "read never retries completed or rejected startup intent"
            );
        }
        commands.lock().expect("commands").clear();
        set_user(cx, root_settings);
        fail.store(true, Ordering::SeqCst);
        lsp_store.update(cx, |store, cx| {
            store.restart_language_servers_for_buffers(
                vec![buffer.clone()],
                HashSet::default(),
                true,
                cx,
            )
        });
        cx.run_until_parked();
        while let Some(Some(replacement_server)) = servers.next().now_or_never() {
            server = replacement_server;
        }
        pump(cx);
        assert_eq!(commands.lock().expect("commands").len(), 1);
        assert!(lsp_store.read_with(cx, |store, _| {
            store.jdtls_project_import_failed(worktree_id)
        }));
        fail.store(false, Ordering::SeqCst);
        lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
        pump(cx);
        assert_eq!(
            commands.lock().expect("commands").len(),
            1,
            "provider recovery read cannot retry startup mutation"
        );
        lsp_store
            .update(cx, |store, cx| {
                store.import_selected_local_gradle_project(worktree_id, cx)
            })
            .expect("explicit startup recovery")
            .await
            .expect("recovered root phase");
        assert_eq!(commands.lock().expect("commands").len(), 2);
        assert!(!lsp_store.read_with(cx, |store, _| {
            store.jdtls_project_import_failed(worktree_id)
        }));
        assert_eq!(
            server.server.server_id(),
            lsp_store.read_with(cx, |store, _| store
                .language_server_statuses()
                .find(|(_, status)| status.worktree == Some(worktree_id))
                .expect("current provider")
                .0)
        );
    }

    #[gpui::test]
    async fn saved_maven_selection_uses_one_owned_phase_and_rechecks_effective_settings(
        cx: &mut TestAppContext,
    ) {
        saved_maven_policy_lifecycle(cx, false, false, true, false, None, None).await;
    }

    #[gpui::test]
    async fn saved_maven_startup_replays_once_and_restart_uses_current_settings(
        cx: &mut TestAppContext,
    ) {
        saved_maven_policy_lifecycle(cx, true, false, true, false, None, None).await;
    }

    #[gpui::test]
    async fn saved_maven_ignore_requires_current_complete_ownership_not_semantic_metadata(
        cx: &mut TestAppContext,
    ) {
        saved_maven_policy_lifecycle(cx, false, true, true, false, None, None).await;
    }

    #[gpui::test]
    async fn saved_maven_ignore_rejects_missing_ownership_provider(cx: &mut TestAppContext) {
        saved_maven_policy_lifecycle(cx, false, false, false, false, None, None).await;
    }

    async fn exercise_saved_build_controls(
        project: &Entity<Project>,
        lsp_store: &Entity<project::LspStore>,
        worktree_id: WorktreeId,
        maven: bool,
        failure: &Arc<AtomicBool>,
        mut release: oneshot::Sender<()>,
        cx: &mut TestAppContext,
    ) {
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("build controls panel");
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.view = if maven {
                BuildToolView::Maven
            } else {
                BuildToolView::Gradle
            };
            panel.rebuild(cx);
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace
                .right_dock()
                .update(cx, |dock, cx| dock.set_open(true, window, cx));
            workspace.focus_panel::<DependenciesPanel>(window, cx);
        });
        cx.update(|window, _| window.activate_a11y_for_test());
        let focus_button = |label: &str, cx: &mut gpui::VisualTestContext| {
            let mut focused = false;
            for _ in 0..64 {
                focused = cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value = serde_json::from_str(
                        &window.debug_a11y_tree_json().expect("build controls"),
                    )
                    .expect("AX JSON");
                    if tree["gpui_focus"]
                        .as_str()
                        .is_some_and(|id| tree["nodes"][id]["aria"]["label"] == label)
                    {
                        true
                    } else {
                        window.focus_next(cx);
                        false
                    }
                });
                if focused {
                    break;
                }
            }
            assert!(focused, "build control receives shared keyboard focus");
        };
        let press_button = |cx: &mut gpui::VisualTestContext| {
            cx.update(|window, cx| {
                let keystroke = gpui::Keystroke::parse("enter").expect("Enter");
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
            cx.run_until_parked();
        };
        let label = if maven {
            "Apply saved Maven roots and exclusions for the selected local project folder; executes trusted build configuration"
        } else {
            "Import effective Gradle selection for the selected local project folder; executes trusted build configuration"
        };
        focus_button(label, cx);
        press_button(cx);
        let cancelled_receipt = panel.read_with(cx, |panel, cx| {
            assert!(panel.build_phase_is_pending(cx));
            panel.build_phase_token.expect("owned UI receipt")
        });
        for _ in 0..2 {
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let tree: serde_json::Value =
                    serde_json::from_str(&window.debug_a11y_tree_json().expect("pending phase"))
                        .expect("AX JSON");
                let status = tree["nodes"]
                    .as_object()
                    .expect("nodes")
                    .values()
                    .find(|node| node["aria"]["label"] == "Selected JVM build import status")
                    .expect("persistent phase status");
                assert_eq!(status["aria"]["role"], "Status");
                assert_eq!(status["aria"]["live"], "Polite");
                assert!(
                    status["aria"]["value"].as_str().is_some_and(
                        |value| value.contains(if maven { "Maven" } else { "Gradle" })
                    )
                );
            });
        }
        focus_button(
            "Cancel only the current build import started by this panel",
            cx,
        );
        press_button(cx);
        release.cancellation().await;
        assert!(
            release.is_canceled(),
            "UI receipt cancels only its owned provider request"
        );
        panel.read_with(cx, |panel, cx| {
            assert!(!panel.build_phase_is_pending(cx));
            assert!(panel.build_phase_token.is_none());
            assert!(
                panel
                    .build_phase_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("cancelled"))
            );
        });
        let refresh = |cx: &mut TestAppContext| {
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
        };
        refresh(cx);
        panel.update(cx, |panel, cx| panel.apply_saved_build_selection(maven, cx));
        cx.run_until_parked();
        refresh(cx);
        panel.read_with(cx, |panel, _| {
            assert!(panel.build_phase_token.is_none());
            assert!(
                panel
                    .build_phase_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("completed"))
            );
        });
        let replacement = lsp_store
            .update(cx, |store, cx| {
                if maven {
                    store.apply_saved_local_maven_selection(worktree_id, cx)
                } else {
                    store.import_selected_local_gradle_project(worktree_id, cx)
                }
            })
            .expect("replacement host phase");
        let replacement_token = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_import_token(worktree_id)
                .expect("replacement receipt")
        });
        assert_ne!(replacement_token, cancelled_receipt);
        panel.update(cx, |panel, cx| {
            panel.build_phase_token = Some(cancelled_receipt);
            panel.cancel_selected_import(cx);
            assert!(
                panel
                    .build_phase_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("no other phase"))
            );
        });
        assert_eq!(
            lsp_store.read_with(cx, |store, _| store.jdtls_project_import_token(worktree_id)),
            Some(replacement_token),
            "retired UI receipt cannot cancel replacement host work"
        );
        replacement
            .await
            .expect("replacement survives late UI cancellation");
        refresh(cx);
        failure.store(true, Ordering::SeqCst);
        panel.update(cx, |panel, cx| panel.apply_saved_build_selection(maven, cx));
        cx.run_until_parked();
        refresh(cx);
        panel.read_with(cx, |panel, _| {
            assert!(panel.build_phase_token.is_none());
            let notice = panel.build_phase_notice.as_deref().expect("failure status");
            assert!(
                notice.contains("review jvm_builds")
                    && !notice.contains("private-canary")
                    && !notice.contains("secret")
            );
        });
        failure.store(false, Ordering::SeqCst);
        panel.update(cx, |panel, cx| panel.apply_saved_build_selection(maven, cx));
        cx.run_until_parked();
        refresh(cx);
        panel.read_with(cx, |panel, _| {
            assert!(
                panel
                    .build_phase_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("completed"))
            )
        });
        let worktrees = project.read_with(cx, |project, _| project.worktree_store());
        cx.update(|_, cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"))
            .update(cx, |trusted, cx| {
                trusted.restrict(
                    worktrees.downgrade(),
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                )
            });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            panel.apply_saved_build_selection(maven, cx);
            assert!(panel.build_phase_token.is_none());
            assert!(
                panel
                    .build_phase_notice
                    .as_deref()
                    .is_some_and(|notice| notice.contains("Trust the project"))
            );
        });
    }

    #[gpui::test]
    async fn maven_panel_applies_saved_selection_cancels_and_recovers(cx: &mut TestAppContext) {
        saved_maven_policy_lifecycle(cx, false, false, true, true, None, None).await;
    }

    #[gpui::test]
    async fn saved_maven_masked_selection_replays_clears_and_recovers(cx: &mut TestAppContext) {
        saved_maven_policy_lifecycle(cx, false, false, true, false, Some(true), None).await;
    }

    #[gpui::test]
    async fn saved_maven_masked_startup_restarts_with_current_facts(cx: &mut TestAppContext) {
        saved_maven_policy_lifecycle(cx, true, false, true, false, Some(true), None).await;
    }

    #[gpui::test]
    async fn saved_maven_masked_selection_refuses_legacy_capability(cx: &mut TestAppContext) {
        saved_maven_policy_lifecycle(cx, false, false, true, false, Some(false), None).await;
    }

    #[gpui::test]
    async fn saved_maven_default_masked_selection_replays_clears_and_recovers(
        cx: &mut TestAppContext,
    ) {
        saved_maven_policy_lifecycle(
            cx,
            false,
            false,
            true,
            false,
            Some(true),
            Some(SavedMavenDefaultPolicy::MasksAndExclusions),
        )
        .await;
    }

    #[gpui::test]
    async fn saved_maven_default_masked_startup_restarts_with_current_facts(
        cx: &mut TestAppContext,
    ) {
        saved_maven_policy_lifecycle(
            cx,
            true,
            false,
            true,
            false,
            Some(true),
            Some(SavedMavenDefaultPolicy::MasksOnly),
        )
        .await;
    }

    #[gpui::test]
    async fn saved_maven_default_masked_selection_refuses_legacy_capability(
        cx: &mut TestAppContext,
    ) {
        saved_maven_policy_lifecycle(
            cx,
            false,
            false,
            true,
            false,
            Some(false),
            Some(SavedMavenDefaultPolicy::MasksAndExclusions),
        )
        .await;
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum SavedMavenDefaultPolicy {
        MasksOnly,
        MasksAndExclusions,
    }

    async fn saved_maven_policy_lifecycle(
        cx: &mut TestAppContext,
        startup: bool,
        coverage_checks: bool,
        ownership_providers: bool,
        panel_controls: bool,
        mask_capability: Option<bool>,
        default_policy: Option<SavedMavenDefaultPolicy>,
    ) {
        let default_roots = default_policy.is_some();
        use settings::{LocalSettingsKind, LocalSettingsPath, SettingsStore};
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/saved-maven",
            serde_json::json!({
                "pom.xml": "<project/>", "Main.java": "class Main {}",
                "other": {"pom.xml": "<project/>"},
                "ignored": {"pom.xml": "<project/>"},
                "gradle": {"build.gradle": "plugins { id 'java' }"},
                ".env": {"pom.xml": "private"}
            }),
        )
        .await;
        if default_roots {
            fs.insert_tree(
                "/saved-maven",
                serde_json::json!({
                    "pom.xml":"<project><modules><module>child</module><module>other</module><module>ignored</module></modules></project>",
                    "unlinked":{"pom.xml":"<project/>"},
                    "child":{"pom.xml":"<project/>"}
                }),
            )
            .await;
        }
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/saved-maven".as_ref()], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktree_store.read_with(cx, |store, cx| {
            store.worktrees().next().expect("worktree").read(cx).id()
        });
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        let original = if let Some(policy) = default_policy {
            match policy {
                SavedMavenDefaultPolicy::MasksOnly => {
                    r#"{"jvm_builds":{"maven":{"ignored_pom_patterns":["*/other/pom.xml"]}}}"#
                }
                SavedMavenDefaultPolicy::MasksAndExclusions => {
                    r#"{"jvm_builds":{"maven":{"ignored_projects":["ignored"],"ignored_pom_patterns":["*/other/pom.xml"]}}}"#
                }
            }
        } else if mask_capability.is_some() {
            r#"{"jvm_builds":{"maven":{"selected_poms":["pom.xml","other/pom.xml"],"ignored_projects":["ignored"],"ignored_pom_patterns":["*/other/pom.xml"]}}}"#
        } else {
            r#"{"jvm_builds":{"maven":{"selected_poms":["pom.xml","other/pom.xml"],"ignored_projects":["ignored"]}}}"#
        };
        let set_user = |cx: &mut TestAppContext, settings: &str| {
            cx.update(|cx| {
                cx.update_global(|store: &mut SettingsStore, cx| {
                    store
                        .set_user_settings(settings, cx)
                        .result()
                        .expect("user settings");
                })
            });
        };
        let empty = r#"{"jvm_builds":{"maven":{"selected_poms":[],"ignored_projects":[]}}}"#;
        set_user(cx, if startup { original } else { empty });
        assert!(
            lsp_store
                .update(cx, |store, cx| store
                    .apply_saved_local_maven_selection(worktree_id, cx))
                .is_err()
        );
        cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust"))
            .update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                );
            });
        let inventory_empty = Arc::new(AtomicBool::new(false));
        let resources_unavailable = Arc::new(AtomicBool::new(false));
        let private_inventory = Arc::new(AtomicBool::new(false));
        let semantic_partial = Arc::new(AtomicBool::new(false));
        let ignored_present = Arc::new(AtomicBool::new(true));
        let other_present = Arc::new(AtomicBool::new(true));
        let partial = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(AtomicBool::new(false));
        let private_reply = Arc::new(AtomicBool::new(false));
        let commands = Arc::new(std::sync::Mutex::new(
            Vec::<lsp::ExecuteCommandParams>::new(),
        ));
        let gate = Arc::new(std::sync::Mutex::new(None::<oneshot::Receiver<()>>));
        let default_root_reads = Arc::new(AtomicUsize::new(0));
        let discovery_gate = Arc::new(std::sync::Mutex::new(None::<oneshot::Receiver<()>>));
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let mut binary_statuses = languages.language_server_binary_statuses();
        let initializations = Arc::new(std::sync::Mutex::new(
            Vec::<Option<serde_json::Value>>::new(),
        ));
        let initialize_capabilities = lsp::ServerCapabilities {
            execute_command_provider: Some(lsp::ExecuteCommandOptions {
                commands: [
                    "java.project.getAll",
                    "java.project.getSettings",
                    "groovy.maven.importedProjects",
                    "groovy.gradle.importedProjects",
                    "groovy.maven.unlinkSelected",
                    "groovy.maven.importReactor",
                    "groovy.maven.importReactorWithMasks",
                    "groovy.maven.defaultImportRoots",
                ]
                .into_iter()
                .filter(|command| {
                    *command != "groovy.maven.importReactorWithMasks"
                        || mask_capability == Some(true)
                        || default_roots
                })
                .filter(|command| {
                    *command != "groovy.maven.defaultImportRoots"
                        || default_roots && mask_capability == Some(true)
                })
                .filter(|command| {
                    ownership_providers || *command != "groovy.gradle.importedProjects"
                })
                .map(str::to_string)
                .collect(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut servers = languages.register_fake_lsp("Java", FakeLspAdapter {
            name: "jdtls",
            capabilities: initialize_capabilities.clone(),
            initializer: Some(Box::new({
                let initializations = initializations.clone();
                let initialize_capabilities = initialize_capabilities.clone();
                let inventory_empty = inventory_empty.clone(); let resources_unavailable = resources_unavailable.clone(); let private_inventory = private_inventory.clone(); let semantic_partial = semantic_partial.clone(); let ignored_present = ignored_present.clone(); let other_present = other_present.clone(); let partial = partial.clone(); let failure = failure.clone(); let commands = commands.clone(); let gate = gate.clone(); let private_reply = private_reply.clone(); let default_root_reads = default_root_reads.clone(); let discovery_gate = discovery_gate.clone();
                move |server| {
                    if default_roots {
                        server.set_request_handler::<lsp::request::Initialize, _, _>({
                            let initializations = initializations.clone();
                            let capabilities = initialize_capabilities.clone();
                            move |parameters, _| {
                                initializations.lock().expect("Initialization log").push(parameters.initialization_options);
                                let capabilities = capabilities.clone();
                                async move { Ok(lsp::InitializeResult {
                                    capabilities, server_info: Some(lsp::ServerInfo { name:"jdtls".into(), version:None }),
                                }) }
                            }
                        });
                    }
                    server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                        let inventory_empty = inventory_empty.clone(); let resources_unavailable = resources_unavailable.clone(); let private_inventory = private_inventory.clone(); let semantic_partial = semantic_partial.clone(); let ignored_present = ignored_present.clone(); let other_present = other_present.clone(); let partial = partial.clone(); let failure = failure.clone(); let commands = commands.clone(); let gate = gate.clone(); let private_reply = private_reply.clone(); let default_root_reads = default_root_reads.clone(); let discovery_gate = discovery_gate.clone();
                        move |parameters, _| {
                            let inventory_empty = inventory_empty.clone(); let resources_unavailable = resources_unavailable.clone(); let private_inventory = private_inventory.clone(); let semantic_partial = semantic_partial.clone(); let ignored_present = ignored_present.clone(); let other_present = other_present.clone(); let partial = partial.clone(); let failure = failure.clone(); let commands = commands.clone(); let gate = gate.clone(); let private_reply = private_reply.clone(); let default_root_reads = default_root_reads.clone(); let discovery_gate = discovery_gate.clone();
                            async move {
                                match parameters.command.as_str() {
                                    "groovy.maven.defaultImportRoots" => {
                                        default_root_reads.fetch_add(1, Ordering::SeqCst);
                                        assert_eq!(parameters.arguments.len(), 1);
                                        assert_eq!(parameters.arguments[0]["rootUri"], "file:///saved-maven/");
                                        assert_eq!(parameters.arguments[0]["cachedOnly"], true);
                                        let candidates = parameters.arguments[0]["candidatePomUris"].as_array().expect("bounded candidates");
                                        assert!(candidates.len() <= 32 && candidates.iter().all(|uri| uri.as_str().is_some_and(|uri| uri.starts_with("file:///saved-maven/") && uri.ends_with("pom.xml"))));
                                        let mut roots = vec![serde_json::json!("file:///saved-maven/pom.xml")];
                                        let mut retained = vec![serde_json::json!("file:///saved-maven/pom.xml"),serde_json::json!("file:///saved-maven/child/pom.xml")];
                                        if other_present.load(Ordering::SeqCst) {retained.push(serde_json::json!("file:///saved-maven/other/pom.xml"));}
                                        if ignored_present.load(Ordering::SeqCst) {retained.push(serde_json::json!("file:///saved-maven/ignored/pom.xml"));}
                                        retained.retain(|uri| candidates.contains(uri));
                                        roots.extend(retained.iter().filter(|uri| **uri != serde_json::json!("file:///saved-maven/pom.xml")).cloned());
                                        let held=discovery_gate.lock().expect("discovery gate").take();
                                        if let Some(held)=held {held.await.expect("release discovery");}
                                        if failure.load(Ordering::SeqCst) {anyhow::bail!("private-canary discovery /secret/provider-cache");}
                                        if private_reply.load(Ordering::SeqCst) {roots.push(serde_json::json!("file:///saved-maven/.env/pom.xml"));}
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":"file:///saved-maven/","rootPomUris":roots,"retainedPomUris":retained,"cachedOnly":true,"readOnly":true,"partial":false,"fullSynchronizationKnown":false})))
                                    }
                                    "java.project.getAll" => {
                                        if resources_unavailable.load(Ordering::SeqCst) && !parameters.arguments.is_empty() { return Ok(None); }
                                        if inventory_empty.load(Ordering::SeqCst) { return Ok(Some(serde_json::json!([]))); }
                                        let mut projects = vec!["file:///saved-maven/", "file:///saved-maven/gradle/"];
                                        if default_roots { projects.push("file:///saved-maven/child/"); }
                                        if other_present.load(Ordering::SeqCst) { projects.push("file:///saved-maven/other/"); }
                                        if ignored_present.load(Ordering::SeqCst) { projects.push("file:///saved-maven/ignored/"); }
                                        if private_inventory.load(Ordering::SeqCst) { projects.push("file:///saved-maven/.env/"); }
                                        Ok(Some(serde_json::json!(projects)))
                                    }
                                    "java.project.getSettings" => Ok(Some(if semantic_partial.load(Ordering::SeqCst) { serde_json::json!({}) } else { serde_json::json!({"org.eclipse.jdt.ls.core.sourcePaths":[],"org.eclipse.jdt.ls.core.classpathEntries":[]}) })),
                                    "groovy.maven.importedProjects" | "groovy.gradle.importedProjects" => {
                                        assert_eq!(parameters.arguments, [serde_json::json!({"rootUri":"file:///saved-maven/","cachedOnly":true})]);
                                        let mut projects = if parameters.command == "groovy.maven.importedProjects" { if other_present.load(Ordering::SeqCst) { vec!["file:///saved-maven/", "file:///saved-maven/other/"] } else { vec!["file:///saved-maven/"] } } else { vec!["file:///saved-maven/gradle/"] };
                                        if parameters.command == "groovy.maven.importedProjects" && ignored_present.load(Ordering::SeqCst) { projects.push("file:///saved-maven/ignored/"); }
                                        if parameters.command == "groovy.maven.importedProjects" && default_roots { projects.push("file:///saved-maven/child/"); }
                                        if inventory_empty.load(Ordering::SeqCst) { projects.clear(); }
                                        if parameters.command == "groovy.maven.importedProjects" && private_inventory.load(Ordering::SeqCst) { projects.push("file:///saved-maven/.env/"); }
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":"file:///saved-maven/","projects":projects.into_iter().map(|uri| serde_json::json!({"projectUri":uri,"javaNature":true})).collect::<Vec<_>>(),"partial":partial.load(Ordering::SeqCst),"fullSynchronizationKnown":false})))
                                    }
                                    "groovy.maven.unlinkSelected" | "groovy.maven.importReactor" | "groovy.maven.importReactorWithMasks" => {
                                        commands.lock().expect("command log").push(parameters.clone());
                                        let held = gate.lock().expect("gate").take();
                                        if let Some(held) = held { held.await.expect("release phase"); }
                                        if failure.load(Ordering::SeqCst) { anyhow::bail!("private-canary at /secret/provider-cache"); }
                                        if parameters.command == "groovy.maven.unlinkSelected" {
                                            assert_eq!(parameters.arguments[0]["projectUris"], serde_json::json!(["file:///saved-maven/ignored/"]));
                                            ignored_present.store(false, Ordering::SeqCst);
                                            Ok(Some(serde_json::json!({"version":1,"rootUri":parameters.arguments[0]["rootUri"],"unlinkedProjectUris":parameters.arguments[0]["projectUris"],"selectedUnlinkPhaseComplete":true,"fullSynchronizationKnown":false})))
                                        } else {
                                            assert_eq!(parameters.arguments[0]["offline"], true);
                                            let exclusions = &parameters.arguments[0]["excludedProjectUris"];
                                            assert!(exclusions == &serde_json::json!([]) || exclusions == &serde_json::json!(["file:///saved-maven/ignored/"]));
                                            let mut poms = vec![parameters.arguments[0]["pomUri"].clone()];
                                            if default_roots && parameters.arguments[0]["pomUri"] == "file:///saved-maven/pom.xml" {
                                                poms.push(serde_json::json!("file:///saved-maven/child/pom.xml"));
                                                poms.push(serde_json::json!("file:///saved-maven/other/pom.xml"));
                                                if exclusions == &serde_json::json!([]) {
                                                    poms.push(serde_json::json!("file:///saved-maven/ignored/pom.xml"));
                                                    ignored_present.store(true,Ordering::SeqCst);
                                                }
                                            }
                                            if parameters.arguments[0]["pomUri"] == "file:///saved-maven/ignored/pom.xml" {
                                                ignored_present.store(true, Ordering::SeqCst);
                                            }
                                            if private_reply.load(Ordering::SeqCst) { poms.push(serde_json::json!("file:///saved-maven/.env/pom.xml")); }
                                            let masked = parameters.command == "groovy.maven.importReactorWithMasks";
                                            let mut ignored = Vec::<serde_json::Value>::new();
                                            if poms.contains(&serde_json::json!("file:///saved-maven/other/pom.xml")) {
                                                let excluded = masked && parameters.arguments[0]["pomMasks"]["patterns"] == serde_json::json!(["*/other/pom.xml"]);
                                                other_present.store(!excluded, Ordering::SeqCst);
                                                if excluded {
                                                    ignored.push(serde_json::json!("file:///saved-maven/other/pom.xml"));
                                                    poms.retain(|pom| *pom != serde_json::json!("file:///saved-maven/other/pom.xml"));
                                                }
                                            }
                                            let mut receipt = serde_json::json!({"version":if masked {2} else {1},"rootUri":parameters.arguments[0]["rootUri"],"reactorPomUri":parameters.arguments[0]["pomUri"],"importedPomUris":poms,"excludedProjectUris":parameters.arguments[0]["excludedProjectUris"],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false});
                                            if masked {
                                                assert_eq!(parameters.arguments[0]["pomMasks"]["version"], 1);
                                                assert_eq!(parameters.arguments[0]["pomMasks"]["syntax"], "idea-mask-v1");
                                                assert_eq!(parameters.arguments[0]["pomMasks"]["pathFormat"], "portable-absolute-v1");
                                                assert!(parameters.arguments[0]["pomMasks"]["patterns"] == serde_json::json!([]) || parameters.arguments[0]["pomMasks"]["patterns"] == serde_json::json!(["*/other/pom.xml"]));
                                                receipt["ignoredPomUris"] = serde_json::json!(ignored);
                                            }
                                            Ok(Some(receipt))
                                        }
                                    }
                                    _ => anyhow::bail!("Unexpected provider command"),
                                }
                            }
                        }
                    });
                }
            })), ..Default::default()
        });
        let (buffer, _open_handle) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/saved-maven/Main.java", cx)
            })
            .await
            .expect("source");
        cx.run_until_parked();
        let observed_statuses =
            std::iter::from_fn(|| binary_statuses.next().now_or_never().flatten())
                .collect::<Vec<_>>();
        let mut server = servers.next().now_or_never().flatten().unwrap_or_else(|| {
            panic!("Saved Maven fixture server did not start: {observed_statuses:?}")
        });
        let refresh = |cx: &mut TestAppContext| {
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
        };
        refresh(cx);
        if default_roots {
            let initializations = initializations.lock().expect("Initialization log");
            let options = initializations
                .first()
                .expect("Actual initialize request")
                .as_ref()
                .expect("Controlled initialization");
            assert_eq!(
                options.pointer("/settings/java/import/maven/enabled"),
                Some(&serde_json::Value::Bool(false)),
                "Default masks suppress automatic Maven imports before controlled phases"
            );
        }
        if !ownership_providers {
            set_user(cx, original);
            let error = lsp_store
                .update(cx, |store, cx| {
                    store.apply_saved_local_maven_selection(worktree_id, cx)
                })
                .expect_err("missing ownership provider cannot reconcile ignore");
            assert!(error.to_string().contains("ownership is unavailable"));
            assert!(commands.lock().expect("commands").is_empty());
            return;
        }
        if startup {
            assert_eq!(
                commands.lock().expect("startup commands").len(),
                if default_roots { 1 } else { 3 }
            );
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
        }
        lsp_store.read_with(cx, |store, _| {
            let (snapshot, current) = store
                .jdtls_project_snapshot(worktree_id)
                .expect("ownership");
            assert!(current && !snapshot.partial);
        });
        let startup_commands = commands.lock().expect("commands").len();
        refresh(cx);
        assert_eq!(
            commands.lock().expect("commands").len(),
            startup_commands,
            "ordinary reads never replay startup selection"
        );
        if !startup {
            assert_eq!(startup_commands, 0, "empty selection never imports");
        }
        if let Some(mask_capability) = mask_capability {
            commands.lock().expect("commands").clear();
            set_user(cx, original);
            refresh(cx);
            let apply = |cx: &mut TestAppContext| {
                lsp_store.update(cx, |store, cx| {
                    store.apply_saved_local_maven_selection(worktree_id, cx)
                })
            };
            if !mask_capability {
                let error = apply(cx)
                    .expect_err("Legacy reactor capability cannot authorize masks")
                    .to_string();
                assert!(error.contains("provider is unavailable") && error.contains("retry"));
                assert!(commands.lock().expect("No mutation").is_empty());
                assert!(
                    ignored_present.load(Ordering::SeqCst) && other_present.load(Ordering::SeqCst)
                );
                return;
            }
            apply(cx)
                .expect("Masked selection")
                .await
                .expect("Qualified phase");
            refresh(cx);
            let assert_facts = |cx: &mut TestAppContext, ignored: bool| {
                lsp_store.read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("Current JVM snapshot");
                    assert!(current);
                    let facts = snapshot
                        .metadata
                        .as_ref()
                        .expect("Phase facts retained through a current read");
                    assert_eq!(
                        facts
                            .ignored_poms
                            .iter()
                            .map(|pom| pom.path.as_unix_str())
                            .collect::<Vec<_>>(),
                        if ignored {
                            vec!["other/pom.xml"]
                        } else {
                            vec![]
                        }
                    );
                    let additional = usize::from(default_roots)
                        + usize::from(default_policy == Some(SavedMavenDefaultPolicy::MasksOnly));
                    assert_eq!(
                        facts.imported_poms.len(),
                        if ignored {
                            1 + additional
                        } else {
                            2 + additional
                        }
                    );
                    assert_eq!(
                        snapshot
                            .projects
                            .iter()
                            .filter(|project| project.module_root.path.as_unix_str() == "other")
                            .count(),
                        if ignored { 0 } else { 1 }
                    );
                    assert!(
                        snapshot.projects.iter().any(|project| project
                            .module_root
                            .path
                            .as_unix_str()
                            == "gradle")
                    );
                });
            };
            assert_facts(cx, true);
            if default_roots {
                assert!(default_root_reads.load(Ordering::SeqCst) > 0);
                assert!(
                    commands
                        .lock()
                        .expect("No unrequested root")
                        .iter()
                        .all(|command| command.arguments[0].get("pomUri")
                            != Some(&serde_json::json!("file:///saved-maven/unlinked/pom.xml")))
                );
            }
            if default_policy == Some(SavedMavenDefaultPolicy::MasksOnly) {
                commands.lock().expect("Commands").clear();
                set_user(
                    cx,
                    r#"{"jvm_builds":{"maven":{"ignored_pom_patterns":[]}}}"#,
                );
                refresh(cx);
                refresh(cx);
                assert!(commands.lock().expect("No saved policy phase").is_empty());
                assert!(
                    lsp_store.read_with(cx, |store, _| store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(snapshot, current)| current && snapshot.metadata.is_none())),
                    "Cleared default policy cannot fabricate an import receipt"
                );
                set_user(cx, original);
                refresh(cx);
                refresh(cx);
                assert_facts(cx, true);
            } else {
                set_user(
                    cx,
                    r#"{"jvm_builds":{"maven":{"selected_poms":["other/pom.xml"],"ignored_projects":[],"ignored_pom_patterns":["*/other/pom.xml"]}}}"#,
                );
                refresh(cx);
                apply(cx)
                    .expect("All selected POMs masked")
                    .await
                    .expect("Known empty imported set");
                refresh(cx);
                lsp_store.read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("Known empty phase snapshot");
                    assert!(current);
                    let facts = snapshot
                        .metadata
                        .as_ref()
                        .expect("Empty imported set is retained");
                    assert!(facts.imported_poms.is_empty());
                    assert_eq!(facts.ignored_poms.len(), 1);
                    assert!(
                        snapshot
                            .projects
                            .iter()
                            .any(|project| project.module_root.path.is_empty()),
                        "Unmentioned imported root remains"
                    );
                });
                set_user(cx, original);
                refresh(cx);
                apply(cx)
                    .expect("Restore multi-root selection")
                    .await
                    .expect("Current facts");
                refresh(cx);
                assert_facts(cx, true);
                assert_eq!(
                    fs.read_file_sync("/saved-maven/other/pom.xml")
                        .expect("Source preserved"),
                    b"<project/>"
                );
                cx.update(|cx| {
                    cx.update_global(|settings: &mut SettingsStore, cx| {
                        settings
                            .set_local_settings(
                                worktree_id,
                                LocalSettingsPath::InWorktree(RelPath::empty().into()),
                                LocalSettingsKind::Settings,
                                Some(r#"{"jvm_builds":{"maven":{"ignored_pom_patterns":[]}}}"#),
                                cx,
                            )
                            .expect("Explicitly clear inherited masks");
                    })
                });
                cx.run_until_parked();
                assert!(
                    !lsp_store.read_with(cx, |store, _| store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)),
                    "Settings clearing revokes current facts before another phase"
                );
                refresh(cx);
                apply(cx)
                    .expect("Cleared masks")
                    .await
                    .expect("Import recovery");
                refresh(cx);
                assert_facts(cx, false);
                cx.update(|cx| {
                    cx.update_global(|settings: &mut SettingsStore, cx| {
                        settings
                            .set_local_settings(
                                worktree_id,
                                LocalSettingsPath::InWorktree(RelPath::empty().into()),
                                LocalSettingsKind::Settings,
                                None,
                                cx,
                            )
                            .expect("Remove override");
                    })
                });
                cx.run_until_parked();
                refresh(cx);
                apply(cx)
                    .expect("Inherited masks")
                    .await
                    .expect("Replay current selection");
                refresh(cx);
                assert_facts(cx, true);
                let unmasked = serde_json::json!({"jvm_builds":{"maven":{"selected_poms":["pom.xml","other/pom.xml"],"ignored_projects":["ignored"]}}}).to_string();
                commands.lock().expect("Commands").clear();
                set_user(cx, &unmasked);
                refresh(cx);
                apply(cx)
                    .expect("Remove masks")
                    .await
                    .expect("Legacy recovery");
                refresh(cx);
                assert_facts(cx, false);
                assert!(
                    commands
                        .lock()
                        .expect("Legacy compatibility")
                        .iter()
                        .all(|command| command.command == "groovy.maven.importReactor")
                );
                set_user(cx, original);
                refresh(cx);
                apply(cx)
                    .expect("Restore masks")
                    .await
                    .expect("Fresh receipt");
                refresh(cx);
                assert_facts(cx, true);
            }
            if default_roots {
                for outcome in [
                    "settings",
                    "lsp-settings",
                    "cancel",
                    "failure",
                    "private",
                    "visibility",
                    "trust-owner",
                    "new-input",
                ] {
                    commands.lock().expect("Mutation log").clear();
                    let before = default_root_reads.load(Ordering::SeqCst);
                    let (release, receiver) = oneshot::channel();
                    *discovery_gate.lock().expect("Discovery gate") = Some(receiver);
                    let application = apply(cx).expect("Held provider read");
                    cx.run_until_parked();
                    assert!(default_root_reads.load(Ordering::SeqCst) > before);
                    assert!(commands.lock().expect("Discovery is read-only").is_empty());
                    let removed_trust = if outcome == "trust-owner" {
                        Some(cx.update(|cx| cx.remove_global::<TrustedWorktrees>()))
                    } else {
                        None
                    };
                    match outcome {
                        "settings" => set_user(cx, empty),
                        "lsp-settings" => {
                            let mut settings: serde_json::Value =
                                serde_json::from_str(original).expect("Settings");
                            settings["lsp"] = serde_json::json!({"jdtls":{"settings":{"java":{"import":{"exclusions":["**/unlinked"]}}}}});
                            set_user(cx, &settings.to_string());
                            cx.run_until_parked();
                        }
                        "cancel" => assert!(
                            lsp_store
                                .update(cx, |store, cx| store
                                    .cancel_selected_maven_import(worktree_id, cx))
                                .expect("Owned cancellation")
                        ),
                        "failure" => failure.store(true, Ordering::SeqCst),
                        "private" => private_reply.store(true, Ordering::SeqCst),
                        "visibility" => {
                            let mut settings: serde_json::Value =
                                serde_json::from_str(original).expect("Settings");
                            settings["private_files"] = serde_json::json!(["other"]);
                            set_user(cx, &settings.to_string());
                            cx.run_until_parked();
                        }
                        "trust-owner" => {}
                        "new-input" => {
                            fs.insert_tree(
                                "/saved-maven/discovery-new-input",
                                serde_json::json!({"pom.xml":"<project/>"}),
                            )
                            .await;
                            cx.run_until_parked();
                        }
                        _ => unreachable!(),
                    }
                    if release.send(()).is_err() {
                        assert!(matches!(
                            outcome,
                            "settings" | "cancel" | "visibility" | "new-input"
                        ));
                    }
                    let error = format!(
                        "{:#}",
                        application
                            .await
                            .expect_err("Obsolete provider read cannot authorize mutation")
                    );
                    assert!(
                        (error.contains("refresh")
                            || error.contains("retry")
                            || error.contains("trust"))
                            && !error.contains("private-canary")
                            && !error.contains("provider-cache")
                            && !error.contains(".env"),
                        "{outcome}: {error}"
                    );
                    assert!(
                        commands
                            .lock()
                            .expect("No mutation after rejected discovery")
                            .is_empty()
                    );
                    if let Some(trust) = removed_trust {
                        cx.update(|cx| cx.set_global(trust));
                    }
                    if outcome == "new-input" {
                        project::Fs::remove_file(
                            fs.as_ref(),
                            std::path::Path::new("/saved-maven/discovery-new-input/pom.xml"),
                            Default::default(),
                        )
                        .await
                        .expect("Remove owned fake candidate");
                    }
                    failure.store(false, Ordering::SeqCst);
                    private_reply.store(false, Ordering::SeqCst);
                    set_user(cx, original);
                    cx.run_until_parked();
                    refresh(cx);
                    apply(cx)
                        .expect("Discovery recovery")
                        .await
                        .expect("Current provider roots");
                    refresh(cx);
                    assert_facts(cx, true);
                }
            }
            for outcome in [
                "settings",
                "cancel",
                "failure",
                "private",
                "visibility",
                "trust-owner",
            ]
            .into_iter()
            .chain(default_roots.then_some("new-input"))
            {
                commands.lock().expect("Commands").clear();
                let (release, receiver) = oneshot::channel();
                *gate.lock().expect("Gate") = Some(receiver);
                let application = apply(cx).expect("Held masked phase");
                cx.run_until_parked();
                assert_eq!(commands.lock().expect("Held command").len(), 1);
                let removed_trust = if outcome == "trust-owner" {
                    Some(cx.update(|cx| cx.remove_global::<TrustedWorktrees>()))
                } else {
                    None
                };
                match outcome {
                    "settings" => set_user(cx, empty),
                    "cancel" => {
                        assert!(
                            lsp_store
                                .update(cx, |store, cx| store
                                    .cancel_selected_maven_import(worktree_id, cx))
                                .expect("Current cancellation")
                        );
                    }
                    "failure" => failure.store(true, Ordering::SeqCst),
                    "private" => private_reply.store(true, Ordering::SeqCst),
                    "visibility" => {
                        let mut settings: serde_json::Value =
                            serde_json::from_str(original).expect("Settings");
                        settings["private_files"] = serde_json::json!(["other"]);
                        set_user(cx, &settings.to_string());
                        cx.run_until_parked();
                    }
                    "trust-owner" => {}
                    "new-input" => {
                        fs.insert_tree(
                            "/saved-maven/new-candidate",
                            serde_json::json!({"pom.xml":"<project/>"}),
                        )
                        .await;
                        cx.run_until_parked();
                    }
                    _ => unreachable!(),
                }
                if release.send(()).is_err() {
                    assert!(
                        matches!(outcome, "settings" | "cancel" | "visibility" | "new-input"),
                        "Only explicit revocation may retire the held request"
                    );
                }
                let error = application
                    .await
                    .expect_err("Obsolete or failed phase cannot publish")
                    .to_string();
                let recovery_is_actionable = if outcome == "trust-owner" {
                    error.contains("trust")
                } else {
                    error.contains("refresh") || error.contains("retry")
                };
                assert!(
                    recovery_is_actionable
                        && !error.contains("private-canary")
                        && !error.contains("provider-cache")
                        && !error.contains(".env"),
                    "{outcome}: {error}"
                );
                assert!(!lsp_store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }));
                assert_eq!(commands.lock().expect("No later mutation").len(), 1);
                if let Some(trust) = removed_trust {
                    cx.update(|cx| cx.set_global(trust));
                }
                if outcome == "new-input" {
                    project::Fs::remove_file(
                        fs.as_ref(),
                        std::path::Path::new("/saved-maven/new-candidate/pom.xml"),
                        Default::default(),
                    )
                    .await
                    .expect("Retire only the fake new candidate");
                    cx.run_until_parked();
                }
                failure.store(false, Ordering::SeqCst);
                private_reply.store(false, Ordering::SeqCst);
                set_user(cx, original);
                cx.run_until_parked();
                refresh(cx);
                apply(cx)
                    .expect("Explicit recovery")
                    .await
                    .expect("New admitted phase");
                refresh(cx);
                assert_facts(cx, true);
            }
            while let Some(Some(replacement)) = servers.next().now_or_never() {
                server = replacement;
            }
            commands.lock().expect("Commands").clear();
            lsp_store.update(cx, |store, cx| {
                store.restart_language_servers_for_buffers(
                    vec![buffer.clone()],
                    HashSet::default(),
                    true,
                    cx,
                )
            });
            cx.run_until_parked();
            let replacement = servers
                .next()
                .now_or_never()
                .flatten()
                .expect("Replacement server");
            assert_ne!(server.server.server_id(), replacement.server.server_id());
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            assert_facts(cx, true);
            let replayed = commands.lock().expect("Fresh replay").len();
            assert_eq!(replayed, if default_roots { 1 } else { 2 });
            if default_roots {
                let initializations = initializations.lock().expect("Initialization log");
                let options = initializations
                    .last()
                    .expect("Restart initialize")
                    .as_ref()
                    .expect("Controlled restart");
                assert_eq!(
                    options.pointer("/settings/java/import/maven/enabled"),
                    Some(&serde_json::Value::Bool(false))
                );
            }
            refresh(cx);
            assert_facts(cx, true);
            assert_eq!(commands.lock().expect("Only once").len(), replayed);
            return;
        }
        commands.lock().expect("commands").clear();
        ignored_present.store(true, Ordering::SeqCst);
        set_user(cx, original);
        if panel_controls {
            let (release, receiver) = oneshot::channel();
            *gate.lock().expect("phase gate") = Some(receiver);
            exercise_saved_build_controls(
                &project,
                &lsp_store,
                worktree_id,
                true,
                &failure,
                release,
                cx,
            )
            .await;
            return;
        }
        if coverage_checks {
            for unavailable in [false, true] {
                inventory_empty.store(true, Ordering::SeqCst);
                resources_unavailable.store(unavailable, Ordering::SeqCst);
                refresh(cx);
                lsp_store.read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("empty ownership read");
                    assert!(current && snapshot.partial && snapshot.projects.is_empty());
                    assert!(
                        snapshot.metadata.is_none(),
                        "Superseded phase facts cannot classify an empty current inventory"
                    );
                });
                let application = lsp_store.update(cx, |store, cx| {
                    store.apply_saved_local_maven_selection(worktree_id, cx)
                });
                if unavailable {
                    assert!(
                        application.is_err(),
                        "unavailable inventory never certifies empty ownership"
                    );
                    assert!(commands.lock().expect("commands").is_empty());
                } else {
                    application
                        .expect("known empty scope admits import")
                        .await
                        .expect("reactor import");
                    assert_eq!(
                        commands
                            .lock()
                            .expect("commands")
                            .iter()
                            .map(|command| command.command.as_str())
                            .collect::<Vec<_>>(),
                        ["groovy.maven.importReactor", "groovy.maven.importReactor"]
                    );
                }
                commands.lock().expect("commands").clear();
            }
            inventory_empty.store(false, Ordering::SeqCst);
            resources_unavailable.store(false, Ordering::SeqCst);
            private_inventory.store(true, Ordering::SeqCst);
            refresh(cx);
            assert!(
                lsp_store
                    .update(cx, |store, cx| store
                        .apply_saved_local_maven_selection(worktree_id, cx))
                    .is_err(),
                "permission-filtered ownership cannot certify complete scope"
            );
            assert!(commands.lock().expect("commands").is_empty());
            private_inventory.store(false, Ordering::SeqCst);
            refresh(cx);
            let mut hidden_settings: serde_json::Value =
                serde_json::from_str(original).expect("saved settings");
            hidden_settings["private_files"] = serde_json::json!(["gradle"]);
            set_user(cx, &hidden_settings.to_string());
            cx.run_until_parked();
            assert!(worktree_store.read_with(cx, |store, cx| {
                store
                    .worktree_for_id(worktree_id, cx)
                    .expect("worktree")
                    .read(cx)
                    .snapshot()
                    .entry_for_path(RelPath::from_unix_str("gradle").expect("path"))
                    .expect("imported directory")
                    .is_private
            }));
            assert!(
                lsp_store
                    .update(cx, |store, cx| store
                        .apply_saved_local_maven_selection(worktree_id, cx))
                    .is_err(),
                "live permission revocation rejects before refreshing stored ownership"
            );
            assert!(commands.lock().expect("commands").is_empty());
            set_user(cx, original);
            cx.run_until_parked();
            semantic_partial.store(true, Ordering::SeqCst);
            refresh(cx);
            assert!(lsp_store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("partial semantic read")
                    .0
                    .partial
            }));
            lsp_store
                .update(cx, |store, cx| {
                    store.apply_saved_local_maven_selection(worktree_id, cx)
                })
                .expect("known ownership does not require source metadata")
                .await
                .expect("saved reconciliation");
            assert_eq!(commands.lock().expect("commands").len(), 3);
            commands.lock().expect("commands").clear();
            semantic_partial.store(false, Ordering::SeqCst);
            ignored_present.store(true, Ordering::SeqCst);
        }
        partial.store(true, Ordering::SeqCst);
        refresh(cx);
        assert!(
            lsp_store
                .update(cx, |store, cx| store
                    .apply_saved_local_maven_selection(worktree_id, cx))
                .is_err(),
            "partial ownership cannot reconcile ignore"
        );
        partial.store(false, Ordering::SeqCst);
        refresh(cx);
        for settings in [
            r#"{"jvm_builds":{"maven":{"ignored_projects":["ignored"]}}}"#,
            r#"{"jvm_builds":{"maven":{"selected_poms":[".env/pom.xml"],"ignored_projects":["ignored"]}}}"#,
            r#"{"jvm_builds":{"maven":{"selected_poms":["pom.xml"],"ignored_projects":["missing"]}}}"#,
        ] {
            set_user(cx, settings);
            assert!(
                lsp_store
                    .update(cx, |store, cx| store
                        .apply_saved_local_maven_selection(worktree_id, cx))
                    .is_err()
            );
        }
        set_user(cx, original);
        refresh(cx);
        let (release, receiver) = oneshot::channel();
        *gate.lock().expect("gate") = Some(receiver);
        let application = lsp_store
            .update(cx, |store, cx| {
                store.apply_saved_local_maven_selection(worktree_id, cx)
            })
            .expect("application");
        cx.run_until_parked();
        assert_eq!(commands.lock().expect("commands").len(), 1);
        assert!(
            lsp_store
                .update(cx, |store, cx| store
                    .apply_saved_local_maven_selection(worktree_id, cx))
                .is_err(),
            "same phase owner rejects concurrent application"
        );
        set_user(
            cx,
            r#"{"jvm_builds":{"maven":{"selected_poms":[],"ignored_projects":["ignored"]}}}"#,
        );
        release.send(()).expect("release");
        let error = application
            .await
            .expect_err("settings changed while unlink held")
            .to_string();
        assert!(error.contains("refresh") && !error.contains("private-canary"));
        assert_eq!(
            commands.lock().expect("commands").len(),
            1,
            "changed intent must stop before reactor dispatch"
        );
        assert!(!lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)
        }));
        set_user(cx, original);
        ignored_present.store(true, Ordering::SeqCst);
        refresh(cx);
        commands.lock().expect("commands").clear();
        private_reply.store(true, Ordering::SeqCst);
        let error = lsp_store
            .update(cx, |store, cx| {
                store.apply_saved_local_maven_selection(worktree_id, cx)
            })
            .expect("private expanded reply request")
            .await
            .expect_err("private module stops before another reactor root")
            .to_string();
        assert!(
            error.contains("refresh") && !error.contains(".env") && !error.contains("saved-maven")
        );
        assert_eq!(commands.lock().expect("commands").len(), 2);
        assert!(!lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)
        }));
        private_reply.store(false, Ordering::SeqCst);
        ignored_present.store(true, Ordering::SeqCst);
        refresh(cx);
        commands.lock().expect("commands").clear();
        lsp_store
            .update(cx, |store, cx| {
                store.apply_saved_local_maven_selection(worktree_id, cx)
            })
            .expect("retry")
            .await
            .expect("ordered phases");
        let observed = commands.lock().expect("commands").clone();
        assert_eq!(
            observed
                .iter()
                .map(|request| request.command.as_str())
                .collect::<Vec<_>>(),
            [
                "groovy.maven.unlinkSelected",
                "groovy.maven.importReactor",
                "groovy.maven.importReactor"
            ]
        );
        assert_eq!(
            observed[1].arguments[0]["pomUri"],
            "file:///saved-maven/pom.xml"
        );
        assert_eq!(
            observed[2].arguments[0]["pomUri"],
            "file:///saved-maven/other/pom.xml"
        );
        refresh(cx);
        lsp_store.read_with(cx, |store, _| {
            let (snapshot, current) = store
                .jdtls_project_snapshot(worktree_id)
                .expect("recovered snapshot");
            assert!(current);
            assert!(
                snapshot
                    .projects
                    .iter()
                    .any(|project| project.module_root.path.as_unix_str() == "gradle")
            );
            assert!(
                !snapshot
                    .projects
                    .iter()
                    .any(|project| project.module_root.path.as_unix_str() == "ignored")
            );
        });
        assert_eq!(
            fs.read_file_sync("/saved-maven/ignored/pom.xml")
                .expect("preserved source"),
            b"<project/>"
        );
        for attempt in 0..2 {
            let (release, receiver) = oneshot::channel();
            *gate.lock().expect("gate") = Some(receiver);
            let application = lsp_store
                .update(cx, |store, cx| {
                    store.apply_saved_local_maven_selection(worktree_id, cx)
                })
                .expect("held application");
            cx.run_until_parked();
            if attempt == 0 {
                assert!(
                    lsp_store
                        .update(cx, |store, cx| store
                            .cancel_selected_maven_import(worktree_id, cx))
                        .expect("cancel")
                );
            } else {
                failure.store(true, Ordering::SeqCst);
            }
            if release.send(()).is_err() {
                assert_eq!(attempt, 0, "only cancellation can close the held provider");
            }
            let error = application.await.expect_err("cancel/failure").to_string();
            assert!(
                error.contains("refresh")
                    && !error.contains("private-canary")
                    && !error.contains("provider-cache")
            );
            assert!(
                lsp_store.read_with(cx, |store, _| store
                    .jdtls_project_import_failed(worktree_id)),
                "saved phase failure stays actionable"
            );
            failure.store(false, Ordering::SeqCst);
            refresh(cx);
        }
        commands.lock().expect("commands").clear();
        cx.update(|cx| cx.update_global(|store: &mut SettingsStore, cx| {
            store.set_local_settings(worktree_id, LocalSettingsPath::InWorktree(RelPath::empty().into()), LocalSettingsKind::Settings, Some(r#"{"jvm_builds":{"maven":{"selected_poms":[],"ignored_projects":[]}}}"#), cx).expect("local empty override");
        }));
        lsp_store
            .update(cx, |store, cx| {
                store.apply_saved_local_maven_selection(worktree_id, cx)
            })
            .expect("empty override")
            .await
            .expect("no imports");
        assert!(commands.lock().expect("commands").is_empty());
        cx.update(|cx| {
            cx.update_global(|store: &mut SettingsStore, cx| {
                store
                    .set_local_settings(
                        worktree_id,
                        LocalSettingsPath::InWorktree(RelPath::empty().into()),
                        LocalSettingsKind::Settings,
                        None,
                        cx,
                    )
                    .expect("remove override");
            })
        });
        refresh(cx);
        lsp_store
            .update(cx, |store, cx| {
                store.apply_saved_local_maven_selection(worktree_id, cx)
            })
            .expect("inherited selection")
            .await
            .expect("restored imports");
        assert_eq!(commands.lock().expect("commands").len(), 2);
        if coverage_checks {
            for unavailable in [false, true] {
                commands.lock().expect("commands").clear();
                inventory_empty.store(true, Ordering::SeqCst);
                resources_unavailable.store(unavailable, Ordering::SeqCst);
                lsp_store.update(cx, |store, cx| {
                    store.restart_language_servers_for_buffers(
                        vec![buffer.clone()],
                        HashSet::default(),
                        true,
                        cx,
                    )
                });
                cx.run_until_parked();
                let _replacement = servers
                    .next()
                    .now_or_never()
                    .flatten()
                    .expect("empty inventory replacement");
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
                assert_eq!(
                    commands.lock().expect("commands").len(),
                    if unavailable { 0 } else { 2 }
                );
                assert_eq!(
                    lsp_store.read_with(cx, |store, _| store
                        .jdtls_project_import_failed(worktree_id)),
                    unavailable
                );
                resources_unavailable.store(false, Ordering::SeqCst);
                refresh(cx);
                assert_eq!(
                    commands.lock().expect("commands").len(),
                    if unavailable { 0 } else { 2 },
                    "ownership recovery read never retries startup mutation"
                );
                if unavailable {
                    lsp_store
                        .update(cx, |store, cx| {
                            store.apply_saved_local_maven_selection(worktree_id, cx)
                        })
                        .expect("explicit empty ownership recovery")
                        .await
                        .expect("reactor recovery");
                    assert_eq!(commands.lock().expect("commands").len(), 2);
                }
            }
        }
        if startup {
            for (settings, change_after_start, expect_import) in [
                (original, false, true),
                (empty, false, false),
                (original, true, false),
                (
                    r#"{"jvm_builds":{"maven":{"selected_poms":[".env/pom.xml"],"ignored_projects":[]}}}"#,
                    false,
                    false,
                ),
            ] {
                commands.lock().expect("commands").clear();
                set_user(cx, settings);
                lsp_store.update(cx, |store, cx| {
                    store.restart_language_servers_for_buffers(
                        vec![buffer.clone()],
                        HashSet::default(),
                        true,
                        cx,
                    )
                });
                cx.run_until_parked();
                let _replacement = servers
                    .next()
                    .now_or_never()
                    .flatten()
                    .expect("replacement server");
                if change_after_start {
                    set_user(cx, empty);
                }
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
                assert_eq!(
                    commands.lock().expect("commands").len(),
                    if expect_import { 2 } else { 0 }
                );
                if change_after_start || settings.contains(".env") {
                    assert!(
                        lsp_store.read_with(cx, |store, _| {
                            store.jdtls_project_import_failed(worktree_id)
                        }),
                        "invalid or superseded startup intent remains actionable"
                    );
                }
                refresh(cx);
                assert_eq!(
                    commands.lock().expect("commands").len(),
                    if expect_import { 2 } else { 0 },
                    "refresh never retries failed or completed mutation"
                );
            }
            commands.lock().expect("commands").clear();
            set_user(cx, original);
            partial.store(true, Ordering::SeqCst);
            lsp_store.update(cx, |store, cx| {
                store.restart_language_servers_for_buffers(
                    vec![buffer.clone()],
                    HashSet::default(),
                    true,
                    cx,
                )
            });
            cx.run_until_parked();
            let _replacement = servers
                .next()
                .now_or_never()
                .flatten()
                .expect("partial provider");
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            assert!(commands.lock().expect("commands").is_empty());
            assert!(lsp_store.read_with(cx, |store, _| {
                store.jdtls_project_import_failed(worktree_id)
            }));
            partial.store(false, Ordering::SeqCst);
            refresh(cx);
            assert!(commands.lock().expect("commands").is_empty());
            lsp_store
                .update(cx, |store, cx| {
                    store.apply_saved_local_maven_selection(worktree_id, cx)
                })
                .expect("manual startup recovery")
                .await
                .expect("recovered selection");
            assert_eq!(commands.lock().expect("commands").len(), 2);
            commands.lock().expect("commands").clear();
            set_user(
                cx,
                r#"{"jvm_builds":{"maven":{"selected_poms":["pom.xml","other/pom.xml"],"ignored_projects":[]}}}"#,
            );
            lsp_store.update(cx, |store, cx| {
                store.restart_language_servers_for_buffers(
                    vec![buffer.clone()],
                    HashSet::default(),
                    true,
                    cx,
                )
            });
            cx.run_until_parked();
            let _replacement = servers
                .next()
                .now_or_never()
                .flatten()
                .expect("manual supersession server");
            lsp_store
                .update(cx, |store, cx| {
                    store.apply_saved_local_maven_selection(worktree_id, cx)
                })
                .expect("manual application before startup query")
                .await
                .expect("manual phase");
            refresh(cx);
            assert_eq!(
                commands.lock().expect("commands").len(),
                2,
                "admitted manual phase supersedes pending startup intent"
            );
        }
    }

    #[gpui::test]
    async fn selected_maven_import_preserves_progress_and_rejects_changed_bindings(
        cx: &mut TestAppContext,
    ) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, false, true, false, false, false, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn selected_gradle_import_preserves_progress_and_rejects_changed_bindings(
        cx: &mut TestAppContext,
    ) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, true, true, false, false, false, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn selected_gradle_import_requires_guarded_provider(cx: &mut TestAppContext) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, true, false, false, false, false, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn selected_maven_unlink_preserves_progress_and_rejects_changed_bindings(
        cx: &mut TestAppContext,
    ) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, false, true, true, false, false, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn selected_gradle_unlink_preserves_progress_and_rejects_changed_bindings(
        cx: &mut TestAppContext,
    ) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, true, true, true, false, false, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn selected_unlink_requires_guarded_provider(cx: &mut TestAppContext) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, true, false, true, false, false, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn selected_non_java_maven_resource_preserves_unlink_permission_lifecycle(
        cx: &mut TestAppContext,
    ) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, false, true, true, true, false, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn selected_non_java_gradle_resource_preserves_unlink_permission_lifecycle(
        cx: &mut TestAppContext,
    ) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, true, true, true, true, false, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn java_unlink_retains_partial_membership_when_resource_enumeration_is_unsupported(
        cx: &mut TestAppContext,
    ) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, true, true, true, false, true, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn maven_reactor_phase_preserves_shared_permission_progress_and_recovery(
        cx: &mut TestAppContext,
    ) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, false, true, false, false, false, true, false,
        )
        .await;
    }

    #[gpui::test]
    async fn maven_reactor_phase_requires_its_distinct_guarded_provider(cx: &mut TestAppContext) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, false, false, false, false, false, true, false,
        )
        .await;
    }

    #[gpui::test]
    async fn gradle_panel_imports_selection_cancels_and_recovers(cx: &mut TestAppContext) {
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, true, false, false, false, false, false, true,
        )
        .await;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ScopedGradleDefaultCase {
        Supported,
        Empty,
        Duplicate,
        Overlap,
        Foreign,
        Private,
        Partial,
        WrongAuthority,
        UnmatchedIgnore,
        OutsideActive,
        MissingDefault,
        MissingPolicy,
        FailureOnce,
    }

    impl gpui::Global for ScopedGradleDefaultCase {}

    #[gpui::test]
    async fn scoped_gradle_metadata_selected_and_default_builds_use_shared_refresh(
        cx: &mut TestAppContext,
    ) {
        scoped_gradle_metadata_lifecycle(cx, false, true).await;
    }

    #[gpui::test]
    async fn scoped_gradle_metadata_rejects_late_module_and_permission_changes(
        cx: &mut TestAppContext,
    ) {
        scoped_gradle_metadata_lifecycle(cx, true, true).await;
    }

    #[gpui::test]
    async fn scoped_gradle_metadata_v3_only_nested_provider_refuses_before_configuration(
        cx: &mut TestAppContext,
    ) {
        scoped_gradle_metadata_lifecycle(cx, false, false).await;
    }

    async fn scoped_gradle_metadata_lifecycle(
        cx: &mut TestAppContext,
        faults: bool,
        module_fence: bool,
    ) {
        use project::Fs as _;
        use settings::SettingsStore;
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/scoped-metadata", serde_json::json!({
            "Main.java":"class Main {}", "shared":{"Shared.java":"class Shared {}"}, "plain":{"Plain.java":"class Plain {}"},
            "maven":{"pom.xml":"<project/>","M.java":"class M {}"},
            "first":{"build.gradle":"plugins {}","child":{"build.gradle":"plugins {}","C.java":"class C {}"}},
            "second":{"settings.gradle":"rootProject.name='second'"}
        })).await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/scoped-metadata".as_ref()], cx).await;
        let worktrees = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktrees.read_with(cx, |store, cx| {
            store.worktrees().next().expect("worktree").read(cx).id()
        });
        let store = project.read_with(cx, |project, _| project.lsp_store());
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust"));
        trust.update(cx, |trusted, cx| {
            trusted.trust(
                &worktrees,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let namespace_changed = Arc::new(AtomicBool::new(false));
        let models = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let gate = Arc::new(std::sync::Mutex::new(None::<oneshot::Receiver<()>>));
        let (entered, mut entries) = futures::channel::mpsc::unbounded();
        let mut servers = languages.register_fake_lsp("Java", FakeLspAdapter {
            name:"jdtls", capabilities:lsp::ServerCapabilities {
                execute_command_provider:Some(lsp::ExecuteCommandOptions {commands:["java.project.getAll","java.project.getSettings","groovy.maven.importedProjects","groovy.gradle.importedProjects","groovy.gradle.importedBuildRoots","groovy.gradle.projectDetails.v4","groovy.gradle.projectDetails.v3","groovy.gradle.projectDetails"].into_iter().filter(|command|module_fence || *command != "groovy.gradle.projectDetails.v4").map(str::to_owned).collect(),..Default::default()}),..Default::default()
            }, initializer:Some(Box::new({
                let models=models.clone();let gate=gate.clone();let namespace_changed=namespace_changed.clone();
                move |server| {
                    server.set_request_handler::<lsp::request::ExecuteCommand,_,_>({
                        let models=models.clone();let gate=gate.clone();let namespace_changed=namespace_changed.clone();let entered=entered.clone();
                        move |parameters,_| {
                            let models=models.clone();let gate=gate.clone();let namespace_changed=namespace_changed.clone();let entered=entered.clone();
                            async move {
                                let uri = |module:&str| format!("file:///scoped-metadata/{module}/");
                                let gradle = ["first", "first/child", "second"];
                                let scope = parameters.arguments.first().and_then(|value|value.get("rootUri")).and_then(serde_json::Value::as_str).unwrap_or("file:///scoped-metadata/").to_owned();
                                match parameters.command.as_str() {
                                    "java.project.getAll" => Ok(Some(serde_json::json!(gradle.into_iter().chain(["plain","maven"]).map(uri).collect::<Vec<_>>()))),
                                    "java.project.getSettings" => Ok(Some(serde_json::json!({"org.eclipse.jdt.ls.core.sourcePaths":[],"org.eclipse.jdt.ls.core.classpathEntries":[]}))),
                                    "groovy.maven.importedProjects" | "groovy.gradle.importedProjects" => Ok(Some(serde_json::json!({"version":1,"rootUri":scope,"projects":if parameters.command.contains("maven") {if uri("maven").starts_with(&scope) {vec![serde_json::json!({"projectUri":uri("maven"),"javaNature":true})]}else{vec![]}} else {gradle.into_iter().filter(|module|uri(module).starts_with(&scope)).map(|module| serde_json::json!({"projectUri":uri(module),"javaNature":true})).collect()},"partial":false,"fullSynchronizationKnown":false}))),
                                    "groovy.gradle.importedBuildRoots" => Ok(Some(serde_json::json!({"version":1,"rootUri":scope,"cachedOnly":true,"readOnly":true,"partial":false,"fullSynchronizationKnown":false,
                                        "projects":gradle.into_iter().filter(|module| uri(module).starts_with(&scope) && (!namespace_changed.load(Ordering::SeqCst) || *module != "first/child")).map(|module|serde_json::json!({"projectUri":uri(module),"buildRootUri":uri(if module.starts_with("first") {"first"} else {"second"}),"javaNature":true})).collect::<Vec<_>>()}))),
                                    "groovy.gradle.projectDetails.v4" => {
                                        let build = parameters.arguments[0]["buildRootUri"].as_str().expect("build").to_owned();
                                        let workspace=parameters.arguments[0]["workspaceRootUri"].as_str().expect("workspace").to_owned();
                                        assert!(workspace=="file:///scoped-metadata/" || workspace==build);
                                        assert_eq!(parameters.arguments, [serde_json::json!({"workspaceRootUri":workspace,"buildRootUri":build,"offline":true})]);
                                        assert!(build == uri("first") || build == uri("second"));
                                        models.lock().expect("models").push(build.clone());
                                        let held = gate.lock().expect("gate").take();
                                        if let Some(held)=held {entered.unbounded_send(()).expect("observer");held.await.expect("release metadata");}
                                        let modules = if build == uri("first") {vec![("first",":"),("first/child",":child")]} else {vec![("second",":")]};
                                        Ok(Some(serde_json::json!({"version":4,"workspaceRootUri":workspace,"buildRootUri":build,"modelQueryComplete":true,"fullSynchronizationKnown":false,
                                            "model":{"version":2,"rootUri":build,"sourceSetsKnown":false,"projects":modules.into_iter().map(|(module,project_path)| {
                                                let task = if project_path == ":" {":verify".to_owned()} else {format!("{project_path}:verify")};
                                                serde_json::json!({"buildRootUri":build.replacen("file://","file:",1),"moduleUri":uri(module).replacen("file://","file:",1),"projectPath":project_path,"name":module,"tasks":[{"path":task,"name":"verify","group":"verification","description":null,"optionsKnown":false}],
                                                    "detailsJson":serde_json::json!({"javaSourceSetsKnown":true,"sourceSets":[{"name":"main","roots":[uri(module),uri("shared")],"generatednessKnown":false}],"tasks":[{"path":task,"optionsComplete":false,"options":[]}]}).to_string()})
                                            }).collect::<Vec<_>>()}})))
                                    }
                                    _ => anyhow::bail!("Unexpected metadata fixture command"),
                                }
                            }
                        }
                    });
                }
            })),..Default::default()
        });
        let (buffer, _open) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/scoped-metadata/Main.java", cx)
            })
            .await
            .expect("Java source");
        let _server = servers.next().await.expect("server");
        let pump = |cx: &mut TestAppContext| {
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
        };
        let settings = |cx: &mut TestAppContext, selected: Option<Vec<&str>>, host: bool| {
            let mut value = serde_json::json!({"jvm_builds":{"gradle":{}}});
            if let Some(selected) = selected {
                value["jvm_builds"]["gradle"]["selected_build_files"] = serde_json::json!(selected);
            }
            if host {
                value["lsp"] =
                    serde_json::json!({"jdtls":{"settings":{"fixture_host_option":true}}});
            }
            cx.update(|cx| {
                cx.update_global(|store: &mut SettingsStore, cx| {
                    store
                        .set_user_settings(&value.to_string(), cx)
                        .result()
                        .expect("settings")
                })
            });
            pump(cx);
        };
        for _ in 0..20 {
            pump(cx);
            if store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)
            }) {
                break;
            }
        }
        assert!(
            models.lock().expect("models").is_empty(),
            "startup must not configure models"
        );
        if !module_fence {
            let error = store
                .update(cx, |store, cx| {
                    store.synchronize_local_gradle_project(Some(worktree_id), cx)
                })
                .expect_err("v3-only nested provider must refuse synchronously");
            assert!(
                error.to_string().contains("Module-fenced") && error.to_string().contains("update")
            );
            assert!(models.lock().expect("models").is_empty());
            return;
        }
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("shared panel");
        for (selected, expected) in [
            (
                Some(vec!["first/build.gradle"]),
                vec!["file:///scoped-metadata/first/"],
            ),
            (
                Some(vec!["first/build.gradle", "second/settings.gradle"]),
                vec![
                    "file:///scoped-metadata/first/",
                    "file:///scoped-metadata/second/",
                ],
            ),
            (
                None,
                vec![
                    "file:///scoped-metadata/first/",
                    "file:///scoped-metadata/second/",
                ],
            ),
            (Some(vec![]), vec![]),
        ] {
            settings(cx, selected, false);
            models.lock().expect("models").clear();
            store
                .update(cx, |store, cx| {
                    store.synchronize_local_gradle_project(Some(worktree_id), cx)
                })
                .expect("explicit metadata");
            for _ in 0..20 {
                pump(cx);
                if store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }) {
                    break;
                }
            }
            assert_eq!(*models.lock().expect("models"), expected);
            store.read_with(cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("metadata snapshot");
                assert!(current);
                assert_eq!(
                    snapshot
                        .projects
                        .iter()
                        .filter(|project| project.gradle.is_some())
                        .count(),
                    if expected.len() == 1 {
                        2
                    } else {
                        expected.len() + usize::from(expected.len() == 2)
                    }
                );
                for project in snapshot
                    .projects
                    .iter()
                    .filter(|project| project.gradle.is_some())
                {
                    let gradle = project.gradle.as_ref().expect("model");
                    assert!(gradle.source_sets_known);
                    assert_eq!(gradle.source_sets.len(), 1);
                    assert_eq!(gradle.source_sets[0].roots.len(), 2);
                    assert_eq!(
                        gradle.source_sets[0].roots[1].path,
                        path(worktree_id, "shared")
                    );
                    assert_eq!(gradle.source_sets[0].roots[0].path, project.module_root);
                    assert_eq!(gradle.tasks.len(), 1);
                }
                assert!(
                    snapshot
                        .projects
                        .iter()
                        .any(|project| project.module_root.path.as_unix_str() == "maven")
                );
                assert!(
                    snapshot
                        .projects
                        .iter()
                        .any(|project| project.module_root.path.as_unix_str() == "plain")
                );
            });
        }
        let scoped_server = store.read_with(cx, |store, _| {
            store
                .language_server_for_id(
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("snapshot")
                        .0
                        .server_id,
                )
                .expect("provider")
        });
        let original_folders = scoped_server.workspace_folders();
        scoped_server.set_workspace_folders(
            ["first", "second", "maven", "plain"]
                .into_iter()
                .map(|directory| {
                    format!("file:///scoped-metadata/{directory}/")
                        .parse()
                        .expect("per-build folder")
                })
                .collect(),
        );
        settings(
            cx,
            Some(vec!["first/build.gradle", "second/settings.gradle"]),
            false,
        );
        models.lock().expect("models").clear();
        store
            .update(cx, |store, cx| {
                store.synchronize_local_gradle_project(Some(worktree_id), cx)
            })
            .expect("per-build authority metadata");
        for _ in 0..20 {
            pump(cx);
            if store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)
            }) {
                break;
            }
        }
        assert_eq!(
            *models.lock().expect("models"),
            [
                "file:///scoped-metadata/first/",
                "file:///scoped-metadata/second/"
            ]
        );
        store.read_with(cx,|store,_|{
            let (snapshot,current)=store.jdtls_project_snapshot(worktree_id).expect("per-build snapshot");assert!(current);
            assert_eq!(snapshot.projects.iter().filter(|project|project.gradle.is_some()).count(),3);
            for gradle in snapshot.projects.iter().filter_map(|project|project.gradle.as_ref()) {
                assert_eq!(gradle.source_sets[0].roots[1].path,path(worktree_id,"shared"),"declared source remains within its trusted worktree, outside the narrower initialization scope");
            }
        });
        scoped_server.set_workspace_folders(original_folders);
        settings(cx, Some(vec!["first/build.gradle"]), false);
        store
            .update(cx, |store, cx| {
                store.synchronize_local_gradle_project(Some(worktree_id), cx)
            })
            .expect("selected metadata restored");
        for _ in 0..20 {
            pump(cx);
            if store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)
            }) {
                break;
            }
        }
        let mut wait_for_model = |cx: &mut TestAppContext| {
            for _ in 0..20 {
                pump(cx);
                if entries.next().now_or_never().flatten().is_some() {
                    return true;
                }
                if store.read_with(cx, |store, _| store.jdtls_project_query_failed(worktree_id)) {
                    return false;
                }
            }
            false
        };
        if faults {
            for mode in 0..13 {
                let client_server = store.read_with(cx, |store, _| {
                    store
                        .language_server_for_id(
                            store
                                .jdtls_project_snapshot(worktree_id)
                                .expect("current snapshot")
                                .0
                                .server_id,
                        )
                        .expect("current provider")
                });
                let original_folders = client_server.workspace_folders();
                let (release, held) = oneshot::channel();
                *gate.lock().expect("gate") = Some(held);
                store
                    .update(cx, |store, cx| {
                        store.synchronize_local_gradle_project(Some(worktree_id), cx)
                    })
                    .expect("held metadata");
                assert!(wait_for_model(cx), "model entered for change mode {mode}");
                match mode {
                    0 => {
                        fs.insert_file(
                            "/scoped-metadata/first/child/build.gradle",
                            b"edited".to_vec(),
                        )
                        .await
                    }
                    1 => {
                        fs.insert_file(
                            "/scoped-metadata/first/child/settings.gradle",
                            b"added".to_vec(),
                        )
                        .await
                    }
                    2 => fs
                        .remove_file(
                            std::path::Path::new("/scoped-metadata/first/child/build.gradle"),
                            Default::default(),
                        )
                        .await
                        .expect("remove"),
                    3 => {
                        fs.rename(
                            std::path::Path::new("/scoped-metadata/first/child"),
                            std::path::Path::new("/scoped-metadata/first/old-child"),
                            Default::default(),
                        )
                        .await
                        .expect("move module");
                        fs.insert_tree(
                            "/scoped-metadata/first/child",
                            serde_json::json!({"C.java":"class C {}"}),
                        )
                        .await;
                    }
                    4 => settings(cx, Some(vec!["second/settings.gradle"]), false),
                    5 => settings(cx, Some(vec!["first/build.gradle"]), true),
                    6 => trust.update(cx, |trusted, cx| {
                        trusted.restrict(
                            worktrees.downgrade(),
                            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                            cx,
                        )
                    }),
                    7 => store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx)),
                    8 => client_server.set_workspace_folders(
                        ["file:///scoped-metadata/first/"
                            .parse()
                            .expect("changed folder")]
                        .into(),
                    ),
                    9 => assert!(
                        store
                            .update(cx, |store, cx| store
                                .cancel_local_jdtls_project_query(worktree_id, cx))
                            .expect("cancel exact current query")
                    ),
                    10 => cx
                        .executor()
                        .advance_clock(std::time::Duration::from_secs(120)),
                    11 => project.update(cx, |project, cx| {
                        project.stop_language_servers_for_buffers(
                            vec![buffer.clone()],
                            HashSet::default(),
                            cx,
                        )
                    }),
                    12 => namespace_changed.store(true, Ordering::SeqCst),
                    _ => unreachable!(),
                }
                cx.run_until_parked();
                if release.send(()).is_err() {
                    assert!(!store.read_with(cx, |store, _| {
                        store.jdtls_project_snapshot(worktree_id).is_some_and(
                            |(snapshot, current)| {
                                current
                                    && snapshot
                                        .projects
                                        .iter()
                                        .any(|project| project.gradle.is_some())
                            },
                        )
                    }));
                }
                pump(cx);
                assert!(
                    !store.read_with(cx, |store, _| store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(snapshot, current)| current
                            && snapshot
                                .projects
                                .iter()
                                .any(|project| project.gradle.is_some()))),
                    "late details cannot publish for mode {mode}"
                );
                match mode {
                    0 | 2 => {
                        fs.insert_file(
                            "/scoped-metadata/first/child/build.gradle",
                            b"plugins {}".to_vec(),
                        )
                        .await
                    }
                    1 => fs
                        .remove_file(
                            std::path::Path::new("/scoped-metadata/first/child/settings.gradle"),
                            Default::default(),
                        )
                        .await
                        .expect("restore absent descriptor"),
                    3 => {
                        fs.rename(
                            std::path::Path::new("/scoped-metadata/first/child"),
                            std::path::Path::new("/scoped-metadata/first/discarded-child"),
                            Default::default(),
                        )
                        .await
                        .expect("retire replacement");
                        fs.rename(
                            std::path::Path::new("/scoped-metadata/first/old-child"),
                            std::path::Path::new("/scoped-metadata/first/child"),
                            Default::default(),
                        )
                        .await
                        .expect("restore original module");
                    }
                    _ => {}
                }
                namespace_changed.store(false, Ordering::SeqCst);
                if mode == 6 {
                    trust.update(cx, |trusted, cx| {
                        trusted.trust(
                            &worktrees,
                            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                            cx,
                        )
                    });
                }
                if mode == 8 {
                    client_server.set_workspace_folders(original_folders.clone());
                }
                if mode == 11 {
                    project.update(cx, |project, cx| {
                        project.restart_language_servers_for_buffers(
                            vec![buffer.clone()],
                            HashSet::default(),
                            true,
                            cx,
                        )
                    });
                    let _replacement = servers.next().await.expect("replacement provider");
                    for _ in 0..20 {
                        pump(cx);
                        if store.read_with(cx, |store, _| {
                            store
                                .jdtls_project_snapshot(worktree_id)
                                .is_some_and(|(_, current)| current)
                        }) {
                            break;
                        }
                    }
                }
                settings(cx, Some(vec!["first/build.gradle"]), false);
                store
                    .update(cx, |store, cx| {
                        store.synchronize_local_gradle_project(Some(worktree_id), cx)
                    })
                    .expect("recovery");
                for _ in 0..20 {
                    pump(cx);
                    if store.read_with(cx, |store, _| {
                        store
                            .jdtls_project_snapshot(worktree_id)
                            .is_some_and(|(_, current)| current)
                    }) {
                        break;
                    }
                }
                assert!(
                    store.read_with(cx, |store, _| store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(snapshot, current)| current
                            && snapshot
                                .projects
                                .iter()
                                .any(|project| project.gradle.is_some()))),
                    "recovery {mode}"
                );
            }
        }
        let before = models.lock().expect("models").len();
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.view = BuildToolView::Gradle;
            panel.refresh(cx);
        });
        pump(cx);
        panel.update(cx, |panel, cx| {
            panel.rebuild(cx);
            panel.host.expand_all();
        });
        panel.read_with(cx, |panel, _| {
            assert!(
                panel
                    .host
                    .visible_rows()
                    .iter()
                    .any(|row| row.label.contains("verify")),
                "current provider tasks reach the shared tree"
            );
        });
        assert_eq!(
            models.lock().expect("models").len(),
            before,
            "render/read-only refresh does not configure"
        );
        if faults {
            let (release, held) = oneshot::channel();
            *gate.lock().expect("gate") = Some(held);
            store
                .update(cx, |store, cx| {
                    store.synchronize_local_gradle_project(Some(worktree_id), cx)
                })
                .expect("query before removal");
            assert!(wait_for_model(cx), "model entered before worktree removal");
            project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx));
            cx.run_until_parked();
            if release.send(()).is_err() {
                assert!(!store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }));
            }
            pump(cx);
            assert!(!store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)
            }));
            panel.read_with(cx, |panel, _| {
                assert!(
                    panel.navigation.is_empty(),
                    "removed worktree cannot retain actionable rows"
                )
            });
        }
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn scoped_gradle_default_policy_applies_and_restores_nested_roots(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| cx.set_global(ScopedGradleDefaultCase::Supported));
        for case in [
            ScopedGradleCase::PolicyModule,
            ScopedGradleCase::PolicyRoot,
            ScopedGradleCase::PolicyParent,
            ScopedGradleCase::PolicyRestore,
            ScopedGradleCase::PolicyFresh,
            ScopedGradleCase::PolicyTwo,
            ScopedGradleCase::PolicyNoop,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_default_policy_refuses_ambiguous_roots_and_namespace(
        cx: &mut TestAppContext,
    ) {
        for mode in [
            ScopedGradleDefaultCase::Empty,
            ScopedGradleDefaultCase::Duplicate,
            ScopedGradleDefaultCase::Overlap,
            ScopedGradleDefaultCase::Foreign,
            ScopedGradleDefaultCase::Private,
            ScopedGradleDefaultCase::Partial,
            ScopedGradleDefaultCase::WrongAuthority,
            ScopedGradleDefaultCase::UnmatchedIgnore,
            ScopedGradleDefaultCase::OutsideActive,
            ScopedGradleDefaultCase::MissingDefault,
            ScopedGradleDefaultCase::MissingPolicy,
        ] {
            cx.update(|cx| cx.set_global(mode));
            scoped_gradle_caller_lifecycle(cx, ScopedGradleCase::PolicyModule).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_default_policy_rechecks_discovery_authority(cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_global(ScopedGradleDefaultCase::Supported));
        for case in [
            ScopedGradleCase::PolicyInputRootRead,
            ScopedGradleCase::PolicyRevokeModuleRead,
            ScopedGradleCase::PolicySettingsModuleMutation,
            ScopedGradleCase::PolicyFolderRootMutation,
            ScopedGradleCase::PolicyServerModuleRead,
            ScopedGradleCase::PolicyRemoveWorktreeModuleMutation,
            ScopedGradleCase::PolicyCancelRootRead,
            ScopedGradleCase::PolicyTimeoutModuleRead,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_default_policy_recovers_from_failed_discovery(cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_global(ScopedGradleDefaultCase::FailureOnce));
        scoped_gradle_caller_lifecycle(cx, ScopedGradleCase::PolicyFailedImportOnce).await;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ScopedGradleTransitionCase {
        First,
        Second,
        Both,
        PerBuild,
        Ignored,
        Legacy,
        Mismatch,
        Overlap,
        FailedUnlink,
        ChangedAssociation,
        CancelUnlink,
        SettingsUnlink,
    }
    impl gpui::Global for ScopedGradleTransitionCase {}

    #[gpui::test]
    async fn scoped_gradle_transition_unlinks_only_deselected_roots_before_import(
        cx: &mut TestAppContext,
    ) {
        for mode in [
            ScopedGradleTransitionCase::First,
            ScopedGradleTransitionCase::Second,
            ScopedGradleTransitionCase::Both,
            ScopedGradleTransitionCase::PerBuild,
            ScopedGradleTransitionCase::Ignored,
        ] {
            cx.update(|cx| cx.set_global(mode));
            scoped_gradle_caller_lifecycle(cx, ScopedGradleCase::PolicyTwo).await;
        }
    }
    #[gpui::test]
    async fn scoped_gradle_transition_refuses_incomplete_or_ambiguous_ownership(
        cx: &mut TestAppContext,
    ) {
        for mode in [
            ScopedGradleTransitionCase::Legacy,
            ScopedGradleTransitionCase::Mismatch,
            ScopedGradleTransitionCase::Overlap,
            ScopedGradleTransitionCase::ChangedAssociation,
        ] {
            cx.update(|cx| cx.set_global(mode));
            scoped_gradle_caller_lifecycle(cx, ScopedGradleCase::PolicyTwo).await;
        }
    }
    #[gpui::test]
    async fn scoped_gradle_transition_stops_after_unlink_failure_cancel_or_settings_change(
        cx: &mut TestAppContext,
    ) {
        for mode in [
            ScopedGradleTransitionCase::FailedUnlink,
            ScopedGradleTransitionCase::CancelUnlink,
            ScopedGradleTransitionCase::SettingsUnlink,
        ] {
            cx.update(|cx| cx.set_global(mode));
            scoped_gradle_caller_lifecycle(cx, ScopedGradleCase::PolicyTwo).await;
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ScopedGradleEmptyCase {
        Clear,
        PerBuild,
        PerBuildFailure,
        Noop,
        Restore,
        MissingCapability,
        Partial,
        Duplicate,
        Foreign,
        Private,
        Mismatch,
        ChangedAssociation,
        FailedUnlinkOnce,
        CancelRead,
        CancelUnlink,
        RevokeRead,
        SettingsRead,
        InputRead,
        FoldersRead,
        ServerRead,
        RemoveWorktree,
        TimeoutRead,
        TimeoutUnlink,
    }
    impl gpui::Global for ScopedGradleEmptyCase {}

    #[gpui::test]
    async fn scoped_gradle_empty_selection_clears_without_model_and_restores(
        cx: &mut TestAppContext,
    ) {
        for mode in [
            ScopedGradleEmptyCase::Clear,
            ScopedGradleEmptyCase::PerBuild,
            ScopedGradleEmptyCase::Noop,
            ScopedGradleEmptyCase::Restore,
        ] {
            cx.update(|cx| cx.set_global(mode));
            scoped_gradle_caller_lifecycle(cx, ScopedGradleCase::PolicyTwo).await;
        }
    }
    #[gpui::test]
    async fn scoped_gradle_empty_selection_refuses_incomplete_cached_ownership(
        cx: &mut TestAppContext,
    ) {
        for mode in [
            ScopedGradleEmptyCase::MissingCapability,
            ScopedGradleEmptyCase::Partial,
            ScopedGradleEmptyCase::Duplicate,
            ScopedGradleEmptyCase::Foreign,
            ScopedGradleEmptyCase::Private,
            ScopedGradleEmptyCase::Mismatch,
            ScopedGradleEmptyCase::ChangedAssociation,
            ScopedGradleEmptyCase::FailedUnlinkOnce,
            ScopedGradleEmptyCase::PerBuildFailure,
        ] {
            cx.update(|cx| cx.set_global(mode));
            scoped_gradle_caller_lifecycle(cx, ScopedGradleCase::PolicyTwo).await;
        }
    }
    #[gpui::test]
    async fn scoped_gradle_empty_selection_rechecks_permission_inputs_and_host(
        cx: &mut TestAppContext,
    ) {
        for mode in [
            ScopedGradleEmptyCase::RevokeRead,
            ScopedGradleEmptyCase::SettingsRead,
            ScopedGradleEmptyCase::InputRead,
            ScopedGradleEmptyCase::FoldersRead,
            ScopedGradleEmptyCase::ServerRead,
            ScopedGradleEmptyCase::RemoveWorktree,
        ] {
            cx.update(|cx| cx.set_global(mode));
            scoped_gradle_caller_lifecycle(cx, ScopedGradleCase::PolicyTwo).await;
        }
    }
    #[gpui::test]
    async fn scoped_gradle_empty_selection_cancels_and_times_out_owned_phases(
        cx: &mut TestAppContext,
    ) {
        for mode in [
            ScopedGradleEmptyCase::CancelRead,
            ScopedGradleEmptyCase::CancelUnlink,
            ScopedGradleEmptyCase::TimeoutRead,
            ScopedGradleEmptyCase::TimeoutUnlink,
        ] {
            cx.update(|cx| cx.set_global(mode));
            scoped_gradle_caller_lifecycle(cx, ScopedGradleCase::PolicyTwo).await;
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ScopedGradleCase {
        One,
        PerBuildAuthority,
        MissingScope,
        FoldersRead,
        ServerRead,
        Two,
        MissingCapability,
        OutsideSelection,
        Partial,
        Duplicate,
        Foreign,
        Private,
        FailedImport,
        BadReceipt,
        PostImportOutside,
        TimeoutRead,
        TimeoutImport,
        CancelRead,
        CancelImport,
        RevokeRead,
        SettingsRead,
        HostSettingsImport,
        InputRead,
        RemoveWorktree,
        MetadataWrite,
        Overlap,
        Ignore,
        PolicyModule,
        PolicyRoot,
        PolicyParent,
        PolicyRestore,
        PolicyFresh,
        PolicyTwo,
        PolicyPerBuild,
        PolicyNoop,
        PolicyMetadata,
        PolicyMissingCapability,
        PolicyBadRoot,
        PolicyBadModule,
        PolicyMissingDescription,
        PolicyReopened,
        PolicyOrphan,
        PolicyFailedImportOnce,
        PolicyCancelRootRead,
        PolicyCancelModuleRead,
        PolicyCancelModuleMutation,
        PolicyCancelRootMutation,
        PolicyCancelImport,
        PolicyRevokeModuleRead,
        PolicySettingsModuleMutation,
        PolicyInputRootRead,
        PolicyFolderRootMutation,
        PolicyServerModuleRead,
        PolicyRemoveWorktreeModuleMutation,
        PolicyIgnoredDirectoryReplacement,
        PolicyTimeoutModuleRead,
        PolicyTimeoutRootMutation,
    }

    impl ScopedGradleCase {
        fn is_scoped_policy(self) -> bool {
            matches!(
                self,
                Self::PolicyModule
                    | Self::PolicyRoot
                    | Self::PolicyParent
                    | Self::PolicyRestore
                    | Self::PolicyFresh
                    | Self::PolicyTwo
                    | Self::PolicyPerBuild
                    | Self::PolicyNoop
                    | Self::PolicyMetadata
                    | Self::PolicyMissingCapability
                    | Self::PolicyBadRoot
                    | Self::PolicyBadModule
                    | Self::PolicyMissingDescription
                    | Self::PolicyReopened
                    | Self::PolicyOrphan
                    | Self::PolicyFailedImportOnce
                    | Self::PolicyCancelRootRead
                    | Self::PolicyCancelModuleRead
                    | Self::PolicyCancelModuleMutation
                    | Self::PolicyCancelRootMutation
                    | Self::PolicyCancelImport
                    | Self::PolicyRevokeModuleRead
                    | Self::PolicySettingsModuleMutation
                    | Self::PolicyInputRootRead
                    | Self::PolicyFolderRootMutation
                    | Self::PolicyServerModuleRead
                    | Self::PolicyRemoveWorktreeModuleMutation
                    | Self::PolicyIgnoredDirectoryReplacement
                    | Self::PolicyTimeoutModuleRead
                    | Self::PolicyTimeoutRootMutation
            )
        }
    }

    #[gpui::test]
    async fn scoped_gradle_caller_imports_one_and_two_nested_roots(cx: &mut TestAppContext) {
        for case in [
            ScopedGradleCase::One,
            ScopedGradleCase::Two,
            ScopedGradleCase::PerBuildAuthority,
            ScopedGradleCase::MetadataWrite,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_caller_refuses_missing_capability_and_unsupported_policy(
        cx: &mut TestAppContext,
    ) {
        for case in [
            ScopedGradleCase::MissingCapability,
            ScopedGradleCase::MissingScope,
            ScopedGradleCase::Overlap,
            ScopedGradleCase::Ignore,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_caller_rejects_incomplete_or_unselected_namespace(
        cx: &mut TestAppContext,
    ) {
        for case in [
            ScopedGradleCase::OutsideSelection,
            ScopedGradleCase::Partial,
            ScopedGradleCase::Duplicate,
            ScopedGradleCase::Foreign,
            ScopedGradleCase::Private,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_caller_stops_after_failed_phase_or_changed_namespace(
        cx: &mut TestAppContext,
    ) {
        for case in [
            ScopedGradleCase::FailedImport,
            ScopedGradleCase::BadReceipt,
            ScopedGradleCase::PostImportOutside,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_caller_cancels_owned_reads_and_imports(cx: &mut TestAppContext) {
        for case in [
            ScopedGradleCase::CancelRead,
            ScopedGradleCase::CancelImport,
            ScopedGradleCase::TimeoutRead,
            ScopedGradleCase::TimeoutImport,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_caller_rechecks_trust_settings_inputs_and_worktree(
        cx: &mut TestAppContext,
    ) {
        for case in [
            ScopedGradleCase::RevokeRead,
            ScopedGradleCase::SettingsRead,
            ScopedGradleCase::HostSettingsImport,
            ScopedGradleCase::InputRead,
            ScopedGradleCase::RemoveWorktree,
            ScopedGradleCase::FoldersRead,
            ScopedGradleCase::ServerRead,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_policy_applies_and_restores_nested_policy(cx: &mut TestAppContext) {
        for case in [
            ScopedGradleCase::PolicyModule,
            ScopedGradleCase::PolicyRoot,
            ScopedGradleCase::PolicyParent,
            ScopedGradleCase::PolicyRestore,
            ScopedGradleCase::PolicyFresh,
            ScopedGradleCase::PolicyTwo,
            ScopedGradleCase::PolicyPerBuild,
            ScopedGradleCase::PolicyNoop,
            ScopedGradleCase::PolicyMetadata,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_policy_refuses_invalid_scoped_policy_data(cx: &mut TestAppContext) {
        for case in [
            ScopedGradleCase::PolicyMissingCapability,
            ScopedGradleCase::PolicyBadRoot,
            ScopedGradleCase::PolicyBadModule,
            ScopedGradleCase::PolicyMissingDescription,
            ScopedGradleCase::PolicyReopened,
            ScopedGradleCase::PolicyOrphan,
            ScopedGradleCase::PolicyFailedImportOnce,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_policy_cancels_scoped_policy_phases(cx: &mut TestAppContext) {
        for case in [
            ScopedGradleCase::PolicyCancelRootRead,
            ScopedGradleCase::PolicyCancelModuleRead,
            ScopedGradleCase::PolicyCancelModuleMutation,
            ScopedGradleCase::PolicyCancelRootMutation,
            ScopedGradleCase::PolicyCancelImport,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_policy_rechecks_scoped_policy_authority(cx: &mut TestAppContext) {
        for case in [
            ScopedGradleCase::PolicyRevokeModuleRead,
            ScopedGradleCase::PolicySettingsModuleMutation,
            ScopedGradleCase::PolicyInputRootRead,
            ScopedGradleCase::PolicyFolderRootMutation,
            ScopedGradleCase::PolicyServerModuleRead,
            ScopedGradleCase::PolicyRemoveWorktreeModuleMutation,
            ScopedGradleCase::PolicyIgnoredDirectoryReplacement,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn scoped_gradle_policy_times_out_scoped_policy_phases(cx: &mut TestAppContext) {
        for case in [
            ScopedGradleCase::PolicyTimeoutModuleRead,
            ScopedGradleCase::PolicyTimeoutRootMutation,
        ] {
            scoped_gradle_caller_lifecycle(cx, case).await;
        }
    }

    async fn scoped_gradle_caller_lifecycle(cx: &mut TestAppContext, case: ScopedGradleCase) {
        use project::Fs as _;
        use settings::SettingsStore;
        let transition_mode =
            cx.update(|cx| cx.try_global::<ScopedGradleTransitionCase>().copied());
        let empty_mode = cx.update(|cx| cx.try_global::<ScopedGradleEmptyCase>().copied());
        let default_mode = cx.update(|cx| cx.try_global::<ScopedGradleDefaultCase>().copied());
        let invalid_defaults = default_mode.is_some_and(|mode| {
            !matches!(
                mode,
                ScopedGradleDefaultCase::Supported | ScopedGradleDefaultCase::FailureOnce
            )
        });
        let policy = case.is_scoped_policy();
        let ignored_root = matches!(
            case,
            ScopedGradleCase::PolicyRoot
                | ScopedGradleCase::PolicyParent
                | ScopedGradleCase::PolicyOrphan
                | ScopedGradleCase::PolicyCancelRootMutation
                | ScopedGradleCase::PolicyFolderRootMutation
                | ScopedGradleCase::PolicyTimeoutRootMutation
        );
        let policy_success = !invalid_defaults
            && matches!(
                case,
                ScopedGradleCase::PolicyModule
                    | ScopedGradleCase::PolicyRoot
                    | ScopedGradleCase::PolicyParent
                    | ScopedGradleCase::PolicyRestore
                    | ScopedGradleCase::PolicyFresh
                    | ScopedGradleCase::PolicyTwo
                    | ScopedGradleCase::PolicyPerBuild
                    | ScopedGradleCase::PolicyNoop
                    | ScopedGradleCase::PolicyMetadata
            );
        let initial_policy_state = std::collections::BTreeMap::from_iter([
            (
                "first".to_owned(),
                (
                    (if matches!(case, ScopedGradleCase::PolicyRestore) {
                        "closed"
                    } else if matches!(
                        case,
                        ScopedGradleCase::PolicyFresh | ScopedGradleCase::PolicyOrphan
                    ) {
                        "absent"
                    } else {
                        "open"
                    })
                    .to_owned(),
                    !matches!(
                        case,
                        ScopedGradleCase::PolicyFresh
                            | ScopedGradleCase::PolicyOrphan
                            | ScopedGradleCase::PolicyMissingDescription
                    ),
                ),
            ),
            (
                "first/child".to_owned(),
                (
                    (if matches!(
                        case,
                        ScopedGradleCase::PolicyRestore | ScopedGradleCase::PolicyNoop
                    ) {
                        "closed"
                    } else if matches!(
                        case,
                        ScopedGradleCase::PolicyFresh | ScopedGradleCase::PolicyOrphan
                    ) {
                        "absent"
                    } else {
                        "open"
                    })
                    .to_owned(),
                    !matches!(
                        case,
                        ScopedGradleCase::PolicyFresh | ScopedGradleCase::PolicyOrphan
                    ),
                ),
            ),
            ("second".to_owned(), ("absent".to_owned(), false)),
        ]);
        let mut initial_policy_state = initial_policy_state;
        if case == ScopedGradleCase::PolicyOrphan {
            initial_policy_state.insert("first/orphan".into(), ("open".into(), true));
        }
        if empty_mode.is_some() || transition_mode.is_some() {
            initial_policy_state.insert("second".into(), ("open".into(), true));
            if empty_mode == Some(ScopedGradleEmptyCase::Noop) {
                for (state, _) in initial_policy_state.values_mut() {
                    *state = "absent".into();
                }
            }
        }
        let policy_states = Arc::new(std::sync::Mutex::new(initial_policy_state.clone()));
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/scoped-gradle", serde_json::json!({
            "Main.java":"class Main {}", "plain":{"Plain.java":"class Plain {}"},
            "maven":{"pom.xml":"<project/>","M.java":"class M {}"},
            "first":{"build.gradle":"plugins { id 'java' }","A.java":"class A {}","child":{"build.gradle":"","C.java":"class C {}"},"orphan":{"build.gradle":""},".env":{"private-module":{"build.gradle":""}}},
            "second":{"settings.gradle":"rootProject.name='second'","B.java":"class B {}"},
            "outside":{"build.gradle":""}, ".env":{"private-module":{"build.gradle":""}}
        })).await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/scoped-gradle".as_ref()], cx).await;
        let worktrees = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktrees.read_with(cx, |store, cx| {
            store.worktrees().next().expect("worktree").read(cx).id()
        });
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust"));
        trust.update(cx, |trusted, cx| {
            trusted.trust(
                &worktrees,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let store = project.read_with(cx, |project, _| project.lsp_store());
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let armed = Arc::new(AtomicBool::new(false));
        let unlinks = Arc::new(AtomicUsize::new(0));
        let association_reads = Arc::new(AtomicUsize::new(0));
        let imports = Arc::new(AtomicUsize::new(0));
        let failure_pending = Arc::new(AtomicBool::new(true));
        let second_imported = Arc::new(AtomicBool::new(false));
        let commands = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let gate = Arc::new(std::sync::Mutex::new(None::<oneshot::Receiver<()>>));
        let (entered, mut entered_receiver) = futures::channel::mpsc::unbounded();
        let held = match case {
            ScopedGradleCase::PolicyCancelRootRead | ScopedGradleCase::PolicyInputRootRead => {
                Some("groovy.gradle.rootState.v2")
            }
            ScopedGradleCase::PolicyCancelModuleRead
            | ScopedGradleCase::PolicyRevokeModuleRead
            | ScopedGradleCase::PolicyServerModuleRead => Some("groovy.gradle.moduleStates.v2"),
            ScopedGradleCase::PolicyCancelModuleMutation
            | ScopedGradleCase::PolicySettingsModuleMutation
            | ScopedGradleCase::PolicyRemoveWorktreeModuleMutation
            | ScopedGradleCase::PolicyIgnoredDirectoryReplacement => {
                Some("groovy.gradle.applyModuleStates.v2")
            }
            ScopedGradleCase::PolicyCancelRootMutation
            | ScopedGradleCase::PolicyFolderRootMutation => Some("groovy.gradle.applyRootState.v2"),
            ScopedGradleCase::PolicyCancelImport => Some("groovy.gradle.importSelected.v2"),
            ScopedGradleCase::CancelImport | ScopedGradleCase::HostSettingsImport => {
                Some("groovy.gradle.importSelected.v2")
            }
            ScopedGradleCase::CancelRead
            | ScopedGradleCase::RevokeRead
            | ScopedGradleCase::SettingsRead
            | ScopedGradleCase::InputRead
            | ScopedGradleCase::RemoveWorktree
            | ScopedGradleCase::FoldersRead
            | ScopedGradleCase::ServerRead => Some("groovy.gradle.importedProjects"),
            _ => None,
        };
        let held = if let Some(mode) = empty_mode {
            match mode {
                ScopedGradleEmptyCase::CancelUnlink => Some("groovy.gradle.unlinkSelected"),
                ScopedGradleEmptyCase::CancelRead
                | ScopedGradleEmptyCase::RevokeRead
                | ScopedGradleEmptyCase::SettingsRead
                | ScopedGradleEmptyCase::InputRead
                | ScopedGradleEmptyCase::FoldersRead
                | ScopedGradleEmptyCase::ServerRead
                | ScopedGradleEmptyCase::RemoveWorktree => Some("groovy.gradle.importedBuildRoots"),
                _ => None,
            }
        } else {
            held
        };
        let held = if matches!(
            transition_mode,
            Some(
                ScopedGradleTransitionCase::CancelUnlink
                    | ScopedGradleTransitionCase::SettingsUnlink
            )
        ) {
            Some("groovy.gradle.unlinkSelected")
        } else {
            held
        };
        let held = if default_mode.is_some() && held.is_some() {
            Some("groovy.gradle.defaultImportRoots")
        } else {
            held
        };
        let mut capabilities = vec![
            "java.project.getAll",
            "java.project.getSettings",
            "groovy.maven.importedProjects",
            "groovy.gradle.importedProjects",
        ];
        if case != ScopedGradleCase::MissingCapability {
            capabilities.push("groovy.gradle.importSelected.v2");
        }
        if policy {
            capabilities.extend([
                "groovy.gradle.rootState.v2",
                "groovy.gradle.applyRootState.v2",
                "groovy.gradle.moduleStates.v2",
            ]);
            if case != ScopedGradleCase::PolicyMissingCapability {
                capabilities.push("groovy.gradle.applyModuleStates.v2");
            }
        }
        if default_mode.is_some() && default_mode != Some(ScopedGradleDefaultCase::MissingDefault) {
            capabilities.push("groovy.gradle.defaultImportRoots");
        }
        if default_mode == Some(ScopedGradleDefaultCase::MissingPolicy) {
            capabilities.retain(|command| *command != "groovy.gradle.applyModuleStates.v2");
        }
        if empty_mode.is_some() {
            capabilities.retain(|command| {
                !command.contains("State") && !command.contains("importSelected")
            });
            capabilities.push("groovy.gradle.importedBuildRoots");
            if empty_mode != Some(ScopedGradleEmptyCase::MissingCapability) {
                capabilities.push("groovy.gradle.unlinkSelected");
            }
            if empty_mode == Some(ScopedGradleEmptyCase::Restore) {
                capabilities.extend([
                    "groovy.gradle.defaultImportRoots",
                    "groovy.gradle.importSelected.v2",
                    "groovy.gradle.rootState.v2",
                    "groovy.gradle.applyRootState.v2",
                    "groovy.gradle.moduleStates.v2",
                    "groovy.gradle.applyModuleStates.v2",
                ]);
            }
        }
        if transition_mode.is_some() && transition_mode != Some(ScopedGradleTransitionCase::Legacy)
        {
            capabilities.extend([
                "groovy.gradle.importedBuildRoots",
                "groovy.gradle.unlinkSelected",
            ]);
        }
        let mut servers = languages.register_fake_lsp("Java", FakeLspAdapter {
            name:"jdtls", capabilities:lsp::ServerCapabilities {
                execute_command_provider:Some(lsp::ExecuteCommandOptions {commands:capabilities.into_iter().map(str::to_owned).collect(), ..Default::default()}), ..Default::default()
            }, initializer:Some(Box::new({
                let unlinks=unlinks.clone(); let association_reads=association_reads.clone(); let second_imported=second_imported.clone(); let failure_pending=failure_pending.clone(); let armed=armed.clone(); let imports=imports.clone(); let commands=commands.clone(); let gate=gate.clone(); let fs=fs.clone(); let policy_states=policy_states.clone();
                move |server| {
                    server.set_request_handler::<lsp::request::ExecuteCommand,_,_>({
                        let unlinks=unlinks.clone(); let association_reads=association_reads.clone(); let second_imported=second_imported.clone(); let failure_pending=failure_pending.clone(); let armed=armed.clone(); let imports=imports.clone(); let commands=commands.clone(); let gate=gate.clone(); let fs=fs.clone(); let policy_states=policy_states.clone(); let entered=entered.clone();
                        move |params, async_cx| {
                            let unlinks=unlinks.clone(); let association_reads=association_reads.clone(); let second_imported=second_imported.clone(); let failure_pending=failure_pending.clone(); let armed=armed.clone(); let imports=imports.clone(); let commands=commands.clone(); let gate=gate.clone(); let fs=fs.clone(); let policy_states=policy_states.clone(); let entered=entered.clone(); let executor=async_cx.background_executor().clone();
                            async move {
                                let importing=armed.load(Ordering::SeqCst);
                                if importing && params.command.starts_with("groovy.gradle.") {
                                    commands.lock().expect("commands").push(params.command.clone());
                                    if held == Some(params.command.as_str()) {
                                        let receiver=gate.lock().expect("gate").take();
                                        if let Some(receiver)=receiver { entered.unbounded_send(()).expect("observer"); receiver.await.expect("released"); }
                                    }
                                }
                                if importing && ((case==ScopedGradleCase::TimeoutRead && params.command=="groovy.gradle.importedProjects") || (case==ScopedGradleCase::TimeoutImport && params.command=="groovy.gradle.importSelected.v2")) {
                                    futures::future::pending::<()>().await;
                                }
                                if importing && ((case == ScopedGradleCase::PolicyTimeoutModuleRead && params.command == "groovy.gradle.moduleStates.v2")
                                    || (case == ScopedGradleCase::PolicyTimeoutRootMutation && params.command == "groovy.gradle.applyRootState.v2")) {
                                    futures::future::pending::<()>().await;
                                }
                                if importing && default_mode.is_some() && case == ScopedGradleCase::PolicyTimeoutModuleRead && params.command == "groovy.gradle.defaultImportRoots" {
                                    futures::future::pending::<()>().await;
                                }
                                if importing && ((empty_mode == Some(ScopedGradleEmptyCase::TimeoutRead) && params.command == "groovy.gradle.importedBuildRoots")
                                    || (empty_mode == Some(ScopedGradleEmptyCase::TimeoutUnlink) && params.command == "groovy.gradle.unlinkSelected")) {
                                    futures::future::pending::<()>().await;
                                }
                                match params.command.as_str() {
                                    "groovy.gradle.importedBuildRoots" => {
                                        let input=params.arguments.first().expect("cached read");
                                        assert_eq!(input.as_object().expect("request").len(), 2);
                                        assert_eq!(input["cachedOnly"], true);
                                        let scope=input["rootUri"].as_str().expect("scope");
                                        let count=if importing { association_reads.fetch_add(1, Ordering::SeqCst) } else {0};
                                        let mut projects=policy_states.lock().expect("state").iter()
                                            .filter(|(_, (state,_))| state=="open")
                                            .map(|(path,_)| (format!("file:///scoped-gradle/{path}/"),if path.starts_with("first/") {"first"} else {path.as_str()}))
                                            .filter(|(uri,_)| uri.starts_with(scope))
                                            .map(|(uri,build)|serde_json::json!({"projectUri":uri,"buildRootUri":format!("file:///scoped-gradle/{build}/"),"javaNature":true})).collect::<Vec<_>>();
                                        if importing {
                                            match empty_mode {
                                                Some(ScopedGradleEmptyCase::Duplicate) => { projects.push(projects.first().expect("project").clone()); }
                                                Some(ScopedGradleEmptyCase::Mismatch) => {projects.pop();}
                                                Some(ScopedGradleEmptyCase::Foreign) => projects.push(serde_json::json!({"projectUri":"file:///foreign-private-canary/","buildRootUri":"file:///foreign-private-canary/","javaNature":true})),
                                                Some(ScopedGradleEmptyCase::Private) => projects.push(serde_json::json!({"projectUri":"file:///scoped-gradle/first/.env/private-module/","buildRootUri":"file:///scoped-gradle/first/","javaNature":true})),
                                                Some(ScopedGradleEmptyCase::ChangedAssociation) if count > 0 => {projects.pop();}
                                                _ => {}
                                            }
                                            if transition_mode==Some(ScopedGradleTransitionCase::Mismatch)
                                                || transition_mode==Some(ScopedGradleTransitionCase::ChangedAssociation) && count>0 {projects.pop();}
                                        }
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":scope,"projects":projects,"cachedOnly":true,"readOnly":true,"partial":importing && empty_mode==Some(ScopedGradleEmptyCase::Partial),"fullSynchronizationKnown":false})))
                                    }
                                    "groovy.gradle.unlinkSelected" => {
                                        let input=params.arguments.first().expect("unlink");
                                        assert_eq!(input.as_object().expect("request").len(),2);
                                        let scope=input["rootUri"].as_str().expect("scope");
                                        let count=unlinks.fetch_add(1,Ordering::SeqCst);
                                        if (transition_mode==Some(ScopedGradleTransitionCase::FailedUnlink) || empty_mode==Some(ScopedGradleEmptyCase::FailedUnlinkOnce) || empty_mode==Some(ScopedGradleEmptyCase::PerBuildFailure) && count==1) && failure_pending.swap(false, Ordering::SeqCst) {
                                            anyhow::bail!("private-canary at /private/provider");
                                        }
                                        let mut states=policy_states.lock().expect("state");
                                        for uri in input["projectUris"].as_array().expect("unlink IDs") {
                                            let uri=uri.as_str().expect("URI"); assert!(uri.starts_with(scope));
                                            let path=uri.strip_prefix("file:///scoped-gradle/").expect("fixture").trim_end_matches('/');
                                            let (state,description)=states.get_mut(path).expect("current resource");
                                            assert_eq!(state,"open"); assert!(*description); *state="absent".into();
                                        }
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":scope,"unlinkedProjectUris":input["projectUris"],"selectedUnlinkPhaseComplete":true,"fullSynchronizationKnown":false})))
                                    }

                                    "groovy.gradle.defaultImportRoots" => {
                                        let input = params.arguments.first().expect("default query");
                                        assert_eq!(input.as_object().expect("request").len(), 3);
                                        assert_eq!(input["rootUri"], "file:///scoped-gradle/");
                                        assert_eq!(input["cachedOnly"], true);
                                        let candidates=input["candidateBuildFileUris"].as_array().expect("candidates");
                                        assert!(candidates.iter().any(|uri| uri == "file:///scoped-gradle/first/build.gradle"));
                                        assert!(candidates.iter().all(|uri| !uri.as_str().expect("URI").contains(".env")));
                                        if importing && default_mode == Some(ScopedGradleDefaultCase::FailureOnce) && failure_pending.swap(false,Ordering::SeqCst) { anyhow::bail!("private-canary at /private/provider"); }
                                        let roots = match default_mode {
                                            Some(ScopedGradleDefaultCase::Empty) => vec![],
                                            Some(ScopedGradleDefaultCase::Duplicate) => vec!["file:///scoped-gradle/first/", "file:///scoped-gradle/first/"],
                                            Some(ScopedGradleDefaultCase::Overlap) => vec!["file:///scoped-gradle/first/", "file:///scoped-gradle/first/child/"],
                                            Some(ScopedGradleDefaultCase::Foreign) => vec!["file:///foreign-private-canary/"],
                                            Some(ScopedGradleDefaultCase::Private) => vec!["file:///scoped-gradle/first/.env/private-module/"],
                                            Some(ScopedGradleDefaultCase::UnmatchedIgnore | ScopedGradleDefaultCase::OutsideActive) => vec!["file:///scoped-gradle/second/"],
                                            _ if case == ScopedGradleCase::PolicyTwo => vec!["file:///scoped-gradle/first/", "file:///scoped-gradle/second/"],
                                            _ => vec!["file:///scoped-gradle/first/"],
                                        };
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":if default_mode == Some(ScopedGradleDefaultCase::WrongAuthority) {serde_json::json!("file:///private-canary/")}else{input["rootUri"].clone()},"buildRootUris":roots,"cachedOnly":true,"readOnly":true,"partial":default_mode == Some(ScopedGradleDefaultCase::Partial),"fullSynchronizationKnown":false})))
                                    }

                                    "java.project.getAll" => {
                                        if policy {
                                            let include_closed = params.arguments.first().is_some_and(|argument| argument.get("includeNonJava").and_then(serde_json::Value::as_bool) == Some(true));
                                            let mut projects = vec!["file:///scoped-gradle/plain/".to_owned(),"file:///scoped-gradle/maven/".to_owned()];
                                            projects.extend(policy_states.lock().expect("state").iter().filter(|(_, (state,_))| state=="open" || include_closed && state!="absent")
                                                .map(|(path,_)| format!("file:///scoped-gradle/{path}/")));
                                            return Ok(Some(serde_json::json!(projects)));
                                        }
                                        let mut projects=vec!["file:///scoped-gradle/plain/", "file:///scoped-gradle/maven/", "file:///scoped-gradle/first/"];
                                        if second_imported.load(Ordering::SeqCst) {projects.push("file:///scoped-gradle/second/");}
                                        Ok(Some(serde_json::json!(projects)))
                                    },
                                    "java.project.getSettings" => Ok(Some(serde_json::json!({"org.eclipse.jdt.ls.core.sourcePaths":[],"org.eclipse.jdt.ls.core.classpathEntries":[]}))),
                                    "groovy.maven.importedProjects" => {
                                        let input=params.arguments.first().expect("Maven scope"); let scope=input["rootUri"].as_str().expect("URI");
                                        let projects=if "file:///scoped-gradle/maven/".starts_with(scope) {vec![serde_json::json!({"projectUri":"file:///scoped-gradle/maven/","javaNature":true})]}else{vec![]};
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":scope,"projects":projects,"partial":false,"fullSynchronizationKnown":false})))
                                    },
                                    "groovy.gradle.importedProjects" => {
                                        let input=params.arguments.first().expect("scope read");
                                        let scope=input["rootUri"].as_str().expect("root URI");
                                        assert_eq!(input["cachedOnly"],true);
                                        if policy {
                                            let projects = policy_states.lock().expect("state").iter().filter(|(_, (state,_))| state=="open")
                                                .map(|(path,_)| format!("file:///scoped-gradle/{path}/")).filter(|uri| uri.starts_with(scope))
                                                .map(|uri|serde_json::json!({"projectUri":uri,"javaNature":true})).collect::<Vec<_>>();
                                            return Ok(Some(serde_json::json!({"version":1,"rootUri":scope,"projects":projects,"partial":false,"fullSynchronizationKnown":false})));
                                        }
                                        let mut projects=vec![serde_json::json!({"projectUri":"file:///scoped-gradle/first/","javaNature":true})];
                                        if second_imported.load(Ordering::SeqCst) {projects.push(serde_json::json!({"projectUri":"file:///scoped-gradle/second/","javaNature":true}));}
                                        if case==ScopedGradleCase::PerBuildAuthority {projects.retain(|project|project["projectUri"].as_str().expect("project URI").starts_with(scope));}
                                        if importing {
                                            let extra=match case {
                                                ScopedGradleCase::OutsideSelection => Some("file:///scoped-gradle/outside/"),
                                                ScopedGradleCase::Duplicate => Some("file:///scoped-gradle/first/"),
                                                ScopedGradleCase::Foreign => Some("file:///foreign-private-canary/"),
                                                ScopedGradleCase::Private => Some("file:///scoped-gradle/first/.env/private-module/"),
                                                ScopedGradleCase::PostImportOutside if imports.load(Ordering::SeqCst)>0 => Some("file:///scoped-gradle/outside/"),
                                                _ => None,
                                            };
                                            if let Some(uri)=extra {projects.push(serde_json::json!({"projectUri":uri,"javaNature":true}));}
                                        }
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":input["rootUri"],"projects":projects,"partial": importing && case==ScopedGradleCase::Partial,"fullSynchronizationKnown":false})))
                                    }
                                    "groovy.gradle.rootState.v2" | "groovy.gradle.moduleStates.v2"
                                        | "groovy.gradle.applyRootState.v2" | "groovy.gradle.applyModuleStates.v2" => {
                                        assert!(policy);
                                        let input=params.arguments.first().expect("state request");
                                        let build=input["buildRootUri"].as_str().expect("build URI");
                                        let prefix=build.strip_prefix("file:///scoped-gradle/").expect("fixture scope").trim_end_matches('/');
                                        assert_eq!(input["offline"], true);
                                        if importing && (case == ScopedGradleCase::PolicyPerBuild || transition_mode == Some(ScopedGradleTransitionCase::PerBuild)) { assert_eq!(input["workspaceRootUri"],input["buildRootUri"]); }
                                        else { assert_eq!(input["workspaceRootUri"],"file:///scoped-gradle/"); }
                                        let reply = {
                                        let mut states=policy_states.lock().expect("state");
                                        if params.command == "groovy.gradle.rootState.v2" {
                                            let (state,description)=states.get(prefix).expect("root state");
                                            return Ok(Some(serde_json::json!({"version":2,"workspaceRootUri":if importing && case==ScopedGradleCase::PolicyBadRoot {serde_json::json!("file:///private-canary/")}else{input["workspaceRootUri"].clone()},
                                                "buildRootUri":build,"workspaceState":state,"descriptionAvailable":description,"fullSynchronizationKnown":false})));
                                        }
                                        if params.command == "groovy.gradle.moduleStates.v2" {
                                            let mut modules=states.iter().filter(|(path,_)|path.as_str()!=prefix && path.starts_with(&format!("{prefix}/")) && !path.ends_with("/orphan"))
                                                .map(|(path,(state,description))| serde_json::json!({"moduleUri":format!("file:///scoped-gradle/{path}/"),"buildRootUri":build,"workspaceState":state,"descriptionAvailable":description})).collect::<Vec<_>>();
                                            if importing && case==ScopedGradleCase::PolicyBadModule {modules.push(serde_json::json!({"moduleUri":"file:///private-canary/","buildRootUri":build,"workspaceState":"open","descriptionAvailable":true}));}
                                            return Ok(Some(serde_json::json!({"version":2,"workspaceRootUri":input["workspaceRootUri"],"buildRootUri":build,"modules":modules,"fullSynchronizationKnown":false})));
                                        }
                                        if params.command == "groovy.gradle.applyRootState.v2" {
                                            let opening=input["open"].as_bool().expect("open flag");
                                            if !opening { assert!(!states.iter().any(|(path,(state,_))|path.as_str()!=prefix && path.starts_with(&format!("{prefix}/")) && state=="open")); }
                                            states.insert(prefix.to_owned(),(if opening {"open"}else{"closed"}.into(),true));
                                            return Ok(Some(serde_json::json!({"version":2,"workspaceRootUri":input["workspaceRootUri"],"buildRootUri":build,"open":opening,"rootStatePhaseComplete":true,"fullSynchronizationKnown":false})));
                                        }
                                        for (field, state) in [("closedProjectUris","closed"),("restoredProjectUris","open")] {
                                            for uri in input[field].as_array().expect("module identities") {
                                                let path=uri.as_str().expect("module URI").strip_prefix("file:///scoped-gradle/").expect("fixture scope").trim_end_matches('/');
                                                assert!(path.starts_with(&format!("{prefix}/")) && states.contains_key(path));
                                                states.insert(path.to_owned(),(state.into(),true));
                                            }
                                        }
                                        serde_json::json!({"version":2,"workspaceRootUri":input["workspaceRootUri"],"buildRootUri":build,
                                            "closedProjectUris":input["closedProjectUris"],"restoredProjectUris":input["restoredProjectUris"],"moduleStatePhaseComplete":true,"fullSynchronizationKnown":false})
                                        };
                                        if importing && case==ScopedGradleCase::PolicyMetadata {
                                            fs.insert_tree("/scoped-gradle/first/child",serde_json::json!({".project":"SDK metadata"})).await;
                                            executor.timer(std::time::Duration::from_millis(300)).await;
                                        }
                                        Ok(Some(reply))
                                    }
                                    "groovy.gradle.importSelected.v2" => {
                                        let input=params.arguments.first().expect("scoped import");
                                        assert_eq!(input.as_object().expect("request").len(),3);
                                        if importing {if matches!(case,ScopedGradleCase::PerBuildAuthority | ScopedGradleCase::PolicyPerBuild) || transition_mode==Some(ScopedGradleTransitionCase::PerBuild) {assert_eq!(input["workspaceRootUri"],input["buildRootUri"]);}else{assert_eq!(input["workspaceRootUri"],"file:///scoped-gradle/");}} assert_eq!(input["offline"],true);
                                        let count=imports.fetch_add(1,Ordering::SeqCst);
                                        if importing && !policy {assert_eq!(input["buildRootUri"],if count==0 {"file:///scoped-gradle/first/"} else {"file:///scoped-gradle/second/"});}
                                        if importing && matches!(case,ScopedGradleCase::FailedImport | ScopedGradleCase::PolicyFailedImportOnce) && failure_pending.swap(false,Ordering::SeqCst) {anyhow::bail!("private-canary at /private/provider");}
                                        if case==ScopedGradleCase::MetadataWrite {
                                            fs.insert_tree("/scoped-gradle/first",serde_json::json!({".project":"SDK owned metadata"})).await;
                                            executor.timer(std::time::Duration::from_millis(300)).await;
                                        }
                                        if policy {
                                            let build=input["buildRootUri"].as_str().expect("build URI").strip_prefix("file:///scoped-gradle/").expect("fixture scope").trim_end_matches('/');
                                            let mut states=policy_states.lock().expect("state");
                                            states.insert(build.to_owned(),("open".into(),true));
                                            if importing && case==ScopedGradleCase::PolicyReopened {states.insert("first/child".into(),("open".into(),true));}
                                        }
                                        if input["buildRootUri"]=="file:///scoped-gradle/second/" {second_imported.store(true,Ordering::SeqCst);}
                                        Ok(Some(serde_json::json!({"version":2,"workspaceRootUri":input["workspaceRootUri"],"buildRootUri":input["buildRootUri"],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":importing && case==ScopedGradleCase::BadReceipt})))
                                    }
                                    _ => anyhow::bail!("unexpected command {}",params.command),
                                }
                            }
                        }
                    });
                }
            })), ..Default::default()
        });
        let (buffer, _open) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/scoped-gradle/Main.java", cx)
            })
            .await
            .expect("Java source");
        let _server = servers.next().await.expect("server");
        for _ in 0..20 {
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            if store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)
            }) {
                break;
            }
        }
        assert!(
            store.read_with(cx, |store, _| store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)),
            "initial current imported coverage"
        );
        let set_settings = |cx: &mut TestAppContext, changed: bool, host: bool| {
            let files = if let Some(mode) = transition_mode {
                if changed {
                    vec!["first/build.gradle"]
                } else {
                    match mode {
                        ScopedGradleTransitionCase::First | ScopedGradleTransitionCase::Ignored => {
                            vec!["first/build.gradle"]
                        }
                        ScopedGradleTransitionCase::Both => {
                            vec!["first/build.gradle", "second/settings.gradle"]
                        }
                        ScopedGradleTransitionCase::Overlap => vec!["first/child/build.gradle"],
                        _ => vec!["second/settings.gradle"],
                    }
                }
            } else if empty_mode.is_some() && !changed {
                vec![]
            } else if case == ScopedGradleCase::Overlap {
                vec!["first/build.gradle", "first/child/build.gradle"]
            } else if case == ScopedGradleCase::One
                || policy
                    && !changed
                    && !matches!(
                        case,
                        ScopedGradleCase::PolicyTwo | ScopedGradleCase::PolicyPerBuild
                    )
            {
                vec!["first/build.gradle"]
            } else if changed {
                vec!["second/settings.gradle"]
            } else {
                vec!["first/build.gradle", "second/settings.gradle"]
            };
            let ignored = if transition_mode.is_some() {
                if transition_mode == Some(ScopedGradleTransitionCase::Ignored) {
                    vec!["first"]
                } else {
                    vec![]
                }
            } else if empty_mode.is_some() {
                vec![]
            } else if case == ScopedGradleCase::Ignore {
                vec!["outside"]
            } else if policy && case != ScopedGradleCase::PolicyRestore {
                if case == ScopedGradleCase::PolicyParent {
                    vec![""]
                } else if ignored_root {
                    vec!["first"]
                } else {
                    vec!["first/child"]
                }
            } else {
                vec![]
            };
            let mut value = serde_json::json!({"jvm_builds":{"gradle":{"selected_build_files":files,"ignored_projects":ignored}}});
            if default_mode.is_some() {
                value["jvm_builds"]["gradle"]
                    .as_object_mut()
                    .expect("settings")
                    .remove("selected_build_files");
                if changed {
                    value["jvm_builds"]["gradle"]["ignored_projects"] =
                        serde_json::json!(["second"]);
                } else if case == ScopedGradleCase::PolicyRestore {
                    value = serde_json::json!({});
                } else if default_mode == Some(ScopedGradleDefaultCase::OutsideActive) {
                    value["jvm_builds"]["gradle"]["ignored_projects"] = serde_json::json!([]);
                }
            }
            if empty_mode == Some(ScopedGradleEmptyCase::Restore) && changed {
                value = serde_json::json!({});
            }
            if host {
                value["lsp"] =
                    serde_json::json!({"jdtls":{"settings":{"fixture_host_option":true}}});
            }
            cx.update(|cx| {
                cx.update_global(|store: &mut SettingsStore, cx| {
                    store
                        .set_user_settings(&value.to_string(), cx)
                        .result()
                        .expect("settings")
                })
            });
        };
        set_settings(cx, false, false);
        cx.run_until_parked();
        for _ in 0..40 {
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            if store.read_with(cx, |store, _| {
                store.jdtls_project_import_token(worktree_id).is_none()
                    && store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
            }) {
                break;
            }
        }
        let client_server = store.read_with(cx, |store, _| {
            let id = store
                .jdtls_project_snapshot(worktree_id)
                .expect("initial snapshot")
                .0
                .server_id;
            store.language_server_for_id(id).expect("client provider")
        });
        if transition_mode == Some(ScopedGradleTransitionCase::PerBuild)
            || matches!(
                empty_mode,
                Some(ScopedGradleEmptyCase::PerBuild | ScopedGradleEmptyCase::PerBuildFailure)
            )
            || matches!(
                case,
                ScopedGradleCase::PerBuildAuthority
                    | ScopedGradleCase::MissingScope
                    | ScopedGradleCase::PolicyPerBuild
            )
        {
            let mut folders = vec![
                "file:///scoped-gradle/first/",
                "file:///scoped-gradle/plain/",
                "file:///scoped-gradle/maven/",
            ];
            if transition_mode == Some(ScopedGradleTransitionCase::PerBuild)
                || matches!(
                    empty_mode,
                    Some(ScopedGradleEmptyCase::PerBuild | ScopedGradleEmptyCase::PerBuildFailure)
                )
                || matches!(
                    case,
                    ScopedGradleCase::PerBuildAuthority | ScopedGradleCase::PolicyPerBuild
                )
            {
                folders.push("file:///scoped-gradle/second/");
            }
            client_server.set_workspace_folders(
                folders
                    .into_iter()
                    .map(|uri| uri.parse().expect("folder URI"))
                    .collect(),
            );
            store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            for _ in 0..20 {
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
                if store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }) {
                    break;
                }
            }
        }
        if policy {
            *policy_states.lock().expect("state") = initial_policy_state.clone();
            store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            for _ in 0..20 {
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
                if store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }) {
                    break;
                }
            }
        }
        unlinks.store(0, Ordering::SeqCst);
        association_reads.store(0, Ordering::SeqCst);
        imports.store(0, Ordering::SeqCst);
        failure_pending.store(true, Ordering::SeqCst);
        commands.lock().expect("commands").clear();
        let (release, receiver) = oneshot::channel();
        if held.is_some() {
            *gate.lock().expect("gate") = Some(receiver);
        } else {
            drop(receiver);
        }
        armed.store(true, Ordering::SeqCst);
        let operation = store.update(cx, |store, cx| {
            store.import_selected_local_gradle_project(worktree_id, cx)
        });
        if empty_mode == Some(ScopedGradleEmptyCase::MissingCapability)
            || matches!(
                default_mode,
                Some(
                    ScopedGradleDefaultCase::MissingDefault
                        | ScopedGradleDefaultCase::MissingPolicy
                )
            )
            || matches!(
                case,
                ScopedGradleCase::MissingCapability
                    | ScopedGradleCase::MissingScope
                    | ScopedGradleCase::Overlap
                    | ScopedGradleCase::Ignore
                    | ScopedGradleCase::PolicyMissingCapability
            )
        {
            let message = operation.expect_err("unsupported scope").to_string();
            assert!(
                message.contains("refresh")
                    || message.contains("review")
                    || message.contains("open"),
                "{message}"
            );
            assert!(commands.lock().expect("commands").is_empty());
            return;
        }
        let operation = operation.expect("scoped operation");
        if default_mode.is_some() && held.is_some() {
            cx.run_until_parked();
            assert_eq!(
                commands.lock().expect("read-only discovery").as_slice(),
                ["groovy.gradle.defaultImportRoots"]
            );
            assert_eq!(*policy_states.lock().expect("state"), initial_policy_state);
        }

        if held.is_some() {
            entered_receiver.next().await.expect("held scoped reply");
            if let Some(mode) = transition_mode {
                match mode {
                    ScopedGradleTransitionCase::CancelUnlink => assert!(
                        store
                            .update(cx, |store, cx| store
                                .cancel_selected_gradle_import(worktree_id, cx))
                            .expect("cancel")
                    ),
                    ScopedGradleTransitionCase::SettingsUnlink => set_settings(cx, true, false),
                    _ => panic!("unexpected transition hold"),
                }
            } else if let Some(mode) = empty_mode {
                match mode {
                    ScopedGradleEmptyCase::CancelRead | ScopedGradleEmptyCase::CancelUnlink => {
                        assert!(
                            store
                                .update(cx, |store, cx| store
                                    .cancel_selected_gradle_import(worktree_id, cx))
                                .expect("cancel")
                        )
                    }
                    ScopedGradleEmptyCase::RevokeRead => trust.update(cx, |trusted, cx| {
                        trusted.restrict(
                            worktrees.downgrade(),
                            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                            cx,
                        )
                    }),
                    ScopedGradleEmptyCase::SettingsRead => set_settings(cx, true, false),
                    ScopedGradleEmptyCase::InputRead => {
                        fs.insert_tree(
                            "/scoped-gradle/first",
                            serde_json::json!({"build.gradle":"changed"}),
                        )
                        .await
                    }
                    ScopedGradleEmptyCase::FoldersRead => client_server.set_workspace_folders(
                        ["file:///scoped-gradle/first/".parse().expect("URI")].into(),
                    ),
                    ScopedGradleEmptyCase::ServerRead => project.update(cx, |project, cx| {
                        project.stop_language_servers_for_buffers(
                            vec![buffer.clone()],
                            HashSet::default(),
                            cx,
                        )
                    }),
                    ScopedGradleEmptyCase::RemoveWorktree => {
                        project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx))
                    }
                    _ => panic!("unexpected empty hold"),
                }
            } else {
                match case {
                    ScopedGradleCase::CancelRead
                    | ScopedGradleCase::CancelImport
                    | ScopedGradleCase::PolicyCancelRootRead
                    | ScopedGradleCase::PolicyCancelModuleRead
                    | ScopedGradleCase::PolicyCancelModuleMutation
                    | ScopedGradleCase::PolicyCancelRootMutation
                    | ScopedGradleCase::PolicyCancelImport => assert!(
                        store
                            .update(cx, |store, cx| store
                                .cancel_selected_gradle_import(worktree_id, cx))
                            .expect("cancel")
                    ),
                    ScopedGradleCase::RevokeRead | ScopedGradleCase::PolicyRevokeModuleRead => {
                        trust.update(cx, |trusted, cx| {
                            trusted.restrict(
                                worktrees.downgrade(),
                                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                                cx,
                            )
                        })
                    }
                    ScopedGradleCase::SettingsRead
                    | ScopedGradleCase::PolicySettingsModuleMutation => {
                        set_settings(cx, true, false)
                    }
                    ScopedGradleCase::HostSettingsImport => set_settings(cx, false, true),
                    ScopedGradleCase::InputRead | ScopedGradleCase::PolicyInputRootRead => {
                        fs.insert_tree(
                            "/scoped-gradle/first",
                            serde_json::json!({"build.gradle":"changed build"}),
                        )
                        .await
                    }
                    ScopedGradleCase::FoldersRead | ScopedGradleCase::PolicyFolderRootMutation => {
                        client_server.set_workspace_folders(
                            ["file:///scoped-gradle/first/".parse().expect("folder URI")].into(),
                        )
                    }
                    ScopedGradleCase::ServerRead | ScopedGradleCase::PolicyServerModuleRead => {
                        project.update(cx, |project, cx| {
                            project.stop_language_servers_for_buffers(
                                vec![buffer.clone()],
                                HashSet::default(),
                                cx,
                            )
                        })
                    }
                    ScopedGradleCase::RemoveWorktree
                    | ScopedGradleCase::PolicyRemoveWorktreeModuleMutation => {
                        project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx))
                    }
                    ScopedGradleCase::PolicyIgnoredDirectoryReplacement => {
                        fs.rename(
                            std::path::Path::new("/scoped-gradle/first/child"),
                            std::path::Path::new("/scoped-gradle/first/retired-child"),
                            Default::default(),
                        )
                        .await
                        .expect("retire fixture directory");
                        fs.insert_tree(
                            "/scoped-gradle/first/child",
                            serde_json::json!({"build.gradle":"","C.java":"class C {}"}),
                        )
                        .await;
                    }
                    _ => panic!("unexpected hold"),
                }
            }
            cx.run_until_parked();
            if release.send(()).is_err() {
                assert!(held.is_some(), "cancelled owned handler");
            }
        } else {
            drop(release);
        }
        let result = operation.await;
        let observed = commands.lock().expect("commands").clone();
        if let Some(mode) = transition_mode {
            let success = matches!(
                mode,
                ScopedGradleTransitionCase::First
                    | ScopedGradleTransitionCase::Second
                    | ScopedGradleTransitionCase::Both
                    | ScopedGradleTransitionCase::PerBuild
                    | ScopedGradleTransitionCase::Ignored
            );
            if success {
                result.expect("saved selection transition");
                let expected_imports = if mode == ScopedGradleTransitionCase::Both {
                    2
                } else if mode == ScopedGradleTransitionCase::Ignored {
                    0
                } else {
                    1
                };
                assert_eq!(imports.load(Ordering::SeqCst), expected_imports);
                assert_eq!(
                    unlinks.load(Ordering::SeqCst),
                    usize::from(mode != ScopedGradleTransitionCase::Both)
                );
                if expected_imports > 0 && mode != ScopedGradleTransitionCase::Both {
                    assert!(
                        observed
                            .iter()
                            .position(|command| command == "groovy.gradle.unlinkSelected")
                            .expect("unlink")
                            < observed
                                .iter()
                                .position(|command| command == "groovy.gradle.importSelected.v2")
                                .expect("import")
                    );
                }
                let states = policy_states.lock().expect("state");
                let first = if matches!(
                    mode,
                    ScopedGradleTransitionCase::First | ScopedGradleTransitionCase::Both
                ) {
                    "open"
                } else if mode == ScopedGradleTransitionCase::Ignored {
                    "closed"
                } else {
                    "absent"
                };
                let second = if matches!(
                    mode,
                    ScopedGradleTransitionCase::First | ScopedGradleTransitionCase::Ignored
                ) {
                    "absent"
                } else {
                    "open"
                };
                assert_eq!(states.get("first").expect("first").0, first);
                assert_eq!(states.get("first/child").expect("child").0, first);
                assert_eq!(states.get("second").expect("second").0, second);
            } else {
                let message = result.expect_err("guarded transition").to_string();
                assert!(
                    !message.contains("private-canary") && !message.contains("/private/provider"),
                    "{message}"
                );
                assert!(
                    message.contains("refresh")
                        || message.contains("review")
                        || message.contains("cancel"),
                    "{mode:?}: {message}"
                );
                assert_eq!(imports.load(Ordering::SeqCst), 0);
                if mode != ScopedGradleTransitionCase::SettingsUnlink {
                    assert_eq!(*policy_states.lock().expect("state"), initial_policy_state);
                }
            }
            assert!(store.read_with(cx, |store, _| {
                store.jdtls_project_import_token(worktree_id).is_none()
            }));
            if mode == ScopedGradleTransitionCase::FailedUnlink {
                armed.store(false, Ordering::SeqCst);
                store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
                for _ in 0..20 {
                    cx.executor()
                        .advance_clock(std::time::Duration::from_millis(300));
                    cx.run_until_parked();
                }
                armed.store(true, Ordering::SeqCst);
                store
                    .update(cx, |store, cx| {
                        store.import_selected_local_gradle_project(worktree_id, cx)
                    })
                    .expect("explicit retry")
                    .await
                    .expect("recovered transition");
                assert_eq!(imports.load(Ordering::SeqCst), 1);
            }
            return;
        }
        if let Some(mode) = empty_mode {
            assert_eq!(
                imports.load(Ordering::SeqCst),
                0,
                "no model/import: {observed:?}"
            );
            assert!(observed.iter().all(|command| matches!(
                command.as_str(),
                "groovy.gradle.importedBuildRoots"
                    | "groovy.gradle.importedProjects"
                    | "groovy.gradle.unlinkSelected"
            )));
            let success = matches!(
                mode,
                ScopedGradleEmptyCase::Clear
                    | ScopedGradleEmptyCase::PerBuild
                    | ScopedGradleEmptyCase::Noop
                    | ScopedGradleEmptyCase::Restore
            );
            if success {
                result.expect("empty selection");
                assert_eq!(
                    unlinks.load(Ordering::SeqCst),
                    if mode == ScopedGradleEmptyCase::Noop {
                        0
                    } else if mode == ScopedGradleEmptyCase::PerBuild {
                        2
                    } else {
                        1
                    }
                );
                assert!(
                    policy_states
                        .lock()
                        .expect("state")
                        .values()
                        .all(|(state, _)| state == "absent")
                );
            } else {
                let message = result.expect_err("refused empty selection").to_string();
                assert!(
                    !message.contains("private-canary") && !message.contains("/private/provider"),
                    "{message}"
                );
                assert!(
                    message.contains("refresh")
                        || message.contains("trust")
                        || message.contains("reopen"),
                    "{mode:?}: {message}"
                );
                if mode == ScopedGradleEmptyCase::PerBuildFailure {
                    let states = policy_states.lock().expect("state");
                    assert_eq!(states.get("first").expect("root").0, "absent");
                    assert_eq!(states.get("first/child").expect("child").0, "absent");
                    assert_eq!(states.get("second").expect("second").0, "open");
                } else {
                    assert_eq!(*policy_states.lock().expect("state"), initial_policy_state);
                }
            }
            assert!(store.read_with(cx, |store, _| {
                store.jdtls_project_import_token(worktree_id).is_none()
            }));
            if success
                || matches!(
                    mode,
                    ScopedGradleEmptyCase::FailedUnlinkOnce
                        | ScopedGradleEmptyCase::PerBuildFailure
                )
            {
                armed.store(false, Ordering::SeqCst);
                store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
                for _ in 0..20 {
                    cx.executor()
                        .advance_clock(std::time::Duration::from_millis(300));
                    cx.run_until_parked();
                }
                if success {
                    store.read_with(cx, |store, _| {
                        let (snapshot, current) = store
                            .jdtls_project_snapshot(worktree_id)
                            .expect("current facts");
                        assert!(current);
                        assert!(
                            snapshot.projects.iter().any(|project| project
                                .module_root
                                .path
                                .as_unix_str()
                                == "maven")
                        );
                        assert!(
                            snapshot.projects.iter().any(|project| project
                                .module_root
                                .path
                                .as_unix_str()
                                == "plain")
                        );
                        assert!(
                            !snapshot
                                .projects
                                .iter()
                                .any(|project| project.imported_build_kind
                                    == project::jvm_project::JdtlsImportedBuildKind::Gradle)
                        );
                    });
                }
                commands.lock().expect("commands").clear();
                armed.store(true, Ordering::SeqCst);
                store
                    .update(cx, |store, cx| {
                        store.import_selected_local_gradle_project(worktree_id, cx)
                    })
                    .expect("retry")
                    .await
                    .expect("idempotent or recovered unlink");
                if mode == ScopedGradleEmptyCase::Restore {
                    armed.store(false, Ordering::SeqCst);
                    set_settings(cx, true, false);
                    cx.run_until_parked();
                    for _ in 0..20 {
                        cx.executor()
                            .advance_clock(std::time::Duration::from_millis(300));
                        cx.run_until_parked();
                    }
                    armed.store(true, Ordering::SeqCst);
                    store
                        .update(cx, |store, cx| {
                            store.import_selected_local_gradle_project(worktree_id, cx)
                        })
                        .expect("restore explicit selection")
                        .await
                        .expect("restore through scoped owner");
                    assert_eq!(
                        policy_states
                            .lock()
                            .expect("state")
                            .get("second")
                            .expect("second")
                            .0,
                        "open"
                    );
                }
            }
            return;
        }
        if default_mode.is_some() {
            assert_eq!(
                observed.first().map(String::as_str),
                Some("groovy.gradle.defaultImportRoots")
            );
            if invalid_defaults
                || held.is_some()
                || case == ScopedGradleCase::PolicyTimeoutModuleRead
            {
                assert_eq!(
                    imports.load(Ordering::SeqCst),
                    0,
                    "{case:?}: {observed:?}; result={result:?}"
                );
                assert!(
                    !observed
                        .iter()
                        .any(|command| command.starts_with("groovy.gradle.apply"))
                );
                assert_eq!(*policy_states.lock().expect("state"), initial_policy_state);
            }
        }
        if policy {
            if policy_success {
                result.expect("scoped policy success");
                let states = policy_states.lock().expect("state");
                let (root_state, _) = states.get("first").expect("first root");
                let (child_state, _) = states.get("first/child").expect("first child");
                assert_eq!(root_state, if ignored_root { "closed" } else { "open" });
                assert_eq!(
                    child_state,
                    if case == ScopedGradleCase::PolicyRestore {
                        "open"
                    } else {
                        "closed"
                    }
                );
                assert_eq!(
                    imports.load(Ordering::SeqCst),
                    if ignored_root {
                        0
                    } else if matches!(
                        case,
                        ScopedGradleCase::PolicyTwo | ScopedGradleCase::PolicyPerBuild
                    ) {
                        2
                    } else {
                        1
                    }
                );
                assert!(
                    observed
                        .iter()
                        .any(|command| command == "groovy.gradle.rootState.v2")
                        && observed
                            .iter()
                            .any(|command| command == "groovy.gradle.moduleStates.v2")
                );
                if case == ScopedGradleCase::PolicyNoop {
                    assert!(
                        !observed
                            .iter()
                            .any(|command| command == "groovy.gradle.applyModuleStates.v2")
                    );
                }
                if case == ScopedGradleCase::PolicyRestore {
                    let root = observed
                        .iter()
                        .position(|command| command == "groovy.gradle.applyRootState.v2")
                        .expect("root restore");
                    let modules = observed
                        .iter()
                        .position(|command| command == "groovy.gradle.applyModuleStates.v2")
                        .expect("module restore");
                    assert!(root < modules);
                }
                if ignored_root {
                    let modules = observed
                        .iter()
                        .position(|command| command == "groovy.gradle.applyModuleStates.v2")
                        .expect("close children");
                    let root = observed
                        .iter()
                        .position(|command| command == "groovy.gradle.applyRootState.v2")
                        .expect("close root");
                    assert!(modules < root);
                }
            } else {
                let message = result.expect_err("scoped policy rejected").to_string();
                assert!(
                    !message.contains("private-canary")
                        && (message.contains("refresh")
                            || message.contains("trust")
                            || message.contains("reopen")
                            || message.contains("review")),
                    "{case:?}: {message}"
                );
                assert!(
                    imports.load(Ordering::SeqCst) <= 1,
                    "later build cannot run after failure"
                );
            }
            assert!(observed.iter().all(|command| !matches!(
                command.as_str(),
                "groovy.gradle.rootState"
                    | "groovy.gradle.applyRootState"
                    | "groovy.gradle.moduleStates"
                    | "groovy.gradle.applyModuleStates"
                    | "groovy.gradle.importSelected"
            )));
            assert!(store.read_with(cx, |store, _| {
                store.jdtls_project_import_token(worktree_id).is_none()
            }));
            if policy_success {
                assert!(!store.read_with(cx, |store, _| {
                    store.jdtls_project_import_failed(worktree_id)
                }));
            }
            if case == ScopedGradleCase::PolicyFailedImportOnce {
                assert!(!failure_pending.load(Ordering::SeqCst));
                armed.store(false, Ordering::SeqCst);
                imports.store(0, Ordering::SeqCst);
                store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
                for _ in 0..20 {
                    cx.executor()
                        .advance_clock(std::time::Duration::from_millis(300));
                    cx.run_until_parked();
                    if store.read_with(cx, |store, _| {
                        store
                            .jdtls_project_snapshot(worktree_id)
                            .is_some_and(|(_, current)| current)
                    }) {
                        break;
                    }
                }
                commands.lock().expect("commands").clear();
                armed.store(true, Ordering::SeqCst);
                store
                    .update(cx, |store, cx| {
                        store.import_selected_local_gradle_project(worktree_id, cx)
                    })
                    .expect("explicit policy retry")
                    .await
                    .expect("recovered current scoped policy");
                assert_eq!(imports.load(Ordering::SeqCst), 1);
                assert!(!store.read_with(cx, |store, _| {
                    store.jdtls_project_import_failed(worktree_id)
                }));
                let recovered = commands.lock().expect("commands").clone();
                assert!(
                    recovered
                        .iter()
                        .any(|command| command == "groovy.gradle.rootState.v2")
                        && recovered
                            .iter()
                            .any(|command| command == "groovy.gradle.moduleStates.v2")
                        && recovered
                            .iter()
                            .any(|command| command == "groovy.gradle.importSelected.v2")
                );
            }
            return;
        }
        if matches!(
            case,
            ScopedGradleCase::One
                | ScopedGradleCase::Two
                | ScopedGradleCase::PerBuildAuthority
                | ScopedGradleCase::MetadataWrite
        ) {
            result.expect("scoped phase success");
            let count = if case == ScopedGradleCase::One { 1 } else { 2 };
            assert_eq!(imports.load(Ordering::SeqCst), count);
            let scopes = if case == ScopedGradleCase::PerBuildAuthority {
                4
            } else {
                1
            };
            let mut expected = vec!["groovy.gradle.importedProjects"; scopes];
            for _ in 0..count {
                expected.push("groovy.gradle.importSelected.v2");
                expected.extend(vec!["groovy.gradle.importedProjects"; scopes]);
            }
            assert!(
                observed.starts_with(
                    &expected
                        .iter()
                        .map(|value| (*value).to_owned())
                        .collect::<Vec<_>>()
                ),
                "read/import ordering: {observed:?}"
            );
        } else {
            let message = result.expect_err("scoped phase rejected").to_string();
            assert!(
                message.contains("refresh")
                    || message.contains("trust")
                    || message.contains("reopen")
                    || message.contains("review")
                    || message.contains("extension setup"),
                "{case:?}: {message}"
            );
            assert!(!message.contains("private-canary") && !message.contains("/private/provider"));
            let expected = usize::from(matches!(
                case,
                ScopedGradleCase::FailedImport
                    | ScopedGradleCase::BadReceipt
                    | ScopedGradleCase::PostImportOutside
                    | ScopedGradleCase::CancelImport
                    | ScopedGradleCase::HostSettingsImport
            ));
            if matches!(
                case,
                ScopedGradleCase::CancelImport
                    | ScopedGradleCase::HostSettingsImport
                    | ScopedGradleCase::TimeoutImport
            ) {
                assert!(imports.load(Ordering::SeqCst) <= 1);
                assert_eq!(
                    observed
                        .iter()
                        .filter(|command| command.as_str() == "groovy.gradle.importSelected.v2")
                        .count(),
                    1
                );
            } else {
                assert_eq!(
                    imports.load(Ordering::SeqCst),
                    expected,
                    "no later import: {observed:?}"
                );
            }
        }
        if matches!(
            case,
            ScopedGradleCase::One
                | ScopedGradleCase::Two
                | ScopedGradleCase::PerBuildAuthority
                | ScopedGradleCase::MetadataWrite
        ) {
            armed.store(false, Ordering::SeqCst);
            for _ in 0..20 {
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
                if store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }) {
                    break;
                }
            }
            store.read_with(cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("refreshed imported facts");
                assert!(current);
                let roots = snapshot
                    .projects
                    .iter()
                    .map(|project| project.module_root.path.as_unix_str())
                    .collect::<Vec<_>>();
                assert!(
                    roots.contains(&"plain")
                        && roots.contains(&"maven")
                        && roots.contains(&"first")
                );
                assert_eq!(roots.contains(&"second"), case != ScopedGradleCase::One);
                assert!(snapshot.projects.iter().any(|project| {
                    project.module_root.path.as_unix_str() == "maven"
                        && project.imported_build_kind
                            == project::jvm_project::JdtlsImportedBuildKind::Maven
                }));
            });
        }
        if case == ScopedGradleCase::FailedImport {
            assert!(!failure_pending.load(Ordering::SeqCst));
            armed.store(false, Ordering::SeqCst);
            imports.store(0, Ordering::SeqCst);
            store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            for _ in 0..20 {
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
                if store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }) {
                    break;
                }
            }
            commands.lock().expect("commands").clear();
            armed.store(true, Ordering::SeqCst);
            store
                .update(cx, |store, cx| {
                    store.import_selected_local_gradle_project(worktree_id, cx)
                })
                .expect("explicit retry")
                .await
                .expect("recovered scoped import");
            assert_eq!(imports.load(Ordering::SeqCst), 2);
            let recovered = commands.lock().expect("commands").clone();
            assert!(
                recovered.starts_with(
                    &[
                        "groovy.gradle.importedProjects",
                        "groovy.gradle.importSelected.v2",
                        "groovy.gradle.importedProjects",
                        "groovy.gradle.importSelected.v2",
                        "groovy.gradle.importedProjects"
                    ]
                    .map(str::to_owned)
                ),
                "fresh namespace and sequential retry: {recovered:?}"
            );
        }
        assert!(
            !observed
                .iter()
                .any(|command| command == "groovy.gradle.importSelected"
                    || command == "groovy.gradle.applyModuleStates")
        );
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum GradleModuleCase {
        Success,
        Noop,
        BadRead,
        MissingCapability,
        MissingRootReadCapability,
        MissingRootPhaseCapability,
        SettingsDuringRead,
        HostSettingsDuringRead,
        CancelRead,
        CancelDelta,
        RevokeRead,
        BuildDuringRead,
        PrivateTarget,
        MissingTarget,
        ForeignTarget,
        RemoveWorktree,
        FailedDelta,
        BadDelta,
        RootIgnore,
        RootFinalAbsent,
        RootFreshDefault,
        RootColdDefault,
        RootClosed,
        RootRecoverable,
        RootFresh,
        RootFailed,
        RootBadReceipt,
        RootBadRead,
        RootBadFinal,
        RootOrphan,
        CancelRootRead,
        CancelRootDelta,
        SettingsDuringRootRead,
        HostSettingsDuringRootRead,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum GradleDefaultRootFixture {
        Supported,
        MissingCapability,
        Empty,
        Nested,
        Malformed,
        Failure,
        RemovedInput,
        AliasedInput,
        StoppedServer,
    }
    impl gpui::Global for GradleDefaultRootFixture {}

    #[gpui::test]
    async fn gradle_default_root_policy_explicit_import_and_removed_ignore(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| cx.set_global(GradleDefaultRootFixture::Supported));
        gradle_module_policy_lifecycle(cx, GradleModuleCase::Success).await;
    }

    #[gpui::test]
    async fn gradle_default_root_policy_automatic_child_exclusions(cx: &mut TestAppContext) {
        cx.update(|cx| {
            cx.set_global(GradleDefaultRootFixture::Supported);
            cx.set_global(GradleStartupPolicyFixture);
        });
        gradle_module_policy_lifecycle(cx, GradleModuleCase::Success).await;
    }

    #[gpui::test]
    async fn gradle_default_root_policy_automatic_retained_root_ignore(cx: &mut TestAppContext) {
        cx.update(|cx| {
            cx.set_global(GradleDefaultRootFixture::Supported);
            cx.set_global(GradleStartupPolicyFixture);
        });
        gradle_module_policy_lifecycle(cx, GradleModuleCase::RootIgnore).await;
    }

    #[gpui::test]
    async fn gradle_default_root_policy_missing_capability_never_dispatches(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| cx.set_global(GradleDefaultRootFixture::MissingCapability));
        gradle_module_policy_lifecycle(cx, GradleModuleCase::Success).await;
    }

    #[gpui::test]
    async fn gradle_default_root_policy_automatic_missing_capability(cx: &mut TestAppContext) {
        cx.update(|cx| {
            cx.set_global(GradleDefaultRootFixture::MissingCapability);
            cx.set_global(GradleStartupPolicyFixture);
        });
        gradle_module_policy_lifecycle(cx, GradleModuleCase::MissingCapability).await;
    }

    #[gpui::test]
    async fn gradle_default_root_policy_unusable_roots_never_mutate(cx: &mut TestAppContext) {
        for mode in [
            GradleDefaultRootFixture::Empty,
            GradleDefaultRootFixture::Nested,
            GradleDefaultRootFixture::Malformed,
            GradleDefaultRootFixture::Failure,
        ] {
            cx.update(|cx| cx.set_global(mode));
            gradle_module_policy_lifecycle(cx, GradleModuleCase::BadRead).await;
        }
    }

    #[gpui::test]
    async fn gradle_default_root_policy_cancellation_and_removed_server(cx: &mut TestAppContext) {
        for mode in [
            GradleDefaultRootFixture::Supported,
            GradleDefaultRootFixture::StoppedServer,
        ] {
            cx.update(|cx| cx.set_global(mode));
            gradle_module_policy_lifecycle(cx, GradleModuleCase::CancelRead).await;
        }
    }

    #[gpui::test]
    async fn gradle_default_root_policy_rechecks_settings_host_trust_and_worktree(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| cx.set_global(GradleDefaultRootFixture::Supported));
        for case in [
            GradleModuleCase::SettingsDuringRead,
            GradleModuleCase::HostSettingsDuringRead,
            GradleModuleCase::RevokeRead,
            GradleModuleCase::RemoveWorktree,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn gradle_default_root_policy_rechecks_current_candidate_entries(
        cx: &mut TestAppContext,
    ) {
        for mode in [
            GradleDefaultRootFixture::Supported,
            GradleDefaultRootFixture::RemovedInput,
            GradleDefaultRootFixture::AliasedInput,
        ] {
            cx.update(|cx| cx.set_global(mode));
            gradle_module_policy_lifecycle(cx, GradleModuleCase::BuildDuringRead).await;
        }
    }

    #[gpui::test]
    async fn gradle_default_root_policy_keeps_fences_after_default_lookup(cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_global(GradleDefaultRootFixture::Supported));
        for case in [
            GradleModuleCase::SettingsDuringRootRead,
            GradleModuleCase::HostSettingsDuringRootRead,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn gradle_default_root_policy_fresh_ignored_root_stays_unsupported(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            cx.set_global(GradleDefaultRootFixture::Supported);
            cx.set_global(GradleStartupPolicyFixture);
        });
        gradle_module_policy_lifecycle(cx, GradleModuleCase::RootFresh).await;
    }

    #[gpui::test]
    async fn gradle_default_root_policy_retry_uses_fresh_lookup(cx: &mut TestAppContext) {
        cx.update(|cx| {
            cx.set_global(GradleDefaultRootFixture::Supported);
            cx.set_global(GradleStartupPolicyFixture);
        });
        gradle_module_policy_lifecycle(cx, GradleModuleCase::FailedDelta).await;
    }

    struct GradleStartupPolicyFixture;
    impl gpui::Global for GradleStartupPolicyFixture {}

    #[gpui::test]
    async fn gradle_startup_policy_applies_child_exclusions(cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_global(GradleStartupPolicyFixture));
        gradle_module_policy_lifecycle(cx, GradleModuleCase::Success).await;
    }

    #[gpui::test]
    async fn gradle_startup_policy_closes_retained_ignored_root(cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_global(GradleStartupPolicyFixture));
        gradle_module_policy_lifecycle(cx, GradleModuleCase::RootIgnore).await;
    }

    #[gpui::test]
    async fn gradle_startup_policy_rejects_missing_capability_before_dispatch(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| cx.set_global(GradleStartupPolicyFixture));
        gradle_module_policy_lifecycle(cx, GradleModuleCase::MissingCapability).await;
    }

    #[gpui::test]
    async fn gradle_startup_policy_refuses_fresh_ignored_root_without_import(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| cx.set_global(GradleStartupPolicyFixture));
        gradle_module_policy_lifecycle(cx, GradleModuleCase::RootFresh).await;
    }

    #[gpui::test]
    async fn gradle_startup_policy_retains_failed_delta_and_recovers_explicitly(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| cx.set_global(GradleStartupPolicyFixture));
        gradle_module_policy_lifecycle(cx, GradleModuleCase::FailedDelta).await;
    }

    #[gpui::test]
    async fn gradle_module_policy_lifecycle_orders_phases_and_restores_removed_ignore(
        cx: &mut TestAppContext,
    ) {
        gradle_module_policy_lifecycle(cx, GradleModuleCase::Success).await;
    }
    #[gpui::test]
    async fn gradle_module_policy_lifecycle_noop_still_imports_without_fabricated_mutation(
        cx: &mut TestAppContext,
    ) {
        gradle_module_policy_lifecycle(cx, GradleModuleCase::Noop).await;
    }
    #[gpui::test]
    async fn gradle_module_policy_lifecycle_rejects_malformed_and_missing_capabilities(
        cx: &mut TestAppContext,
    ) {
        for case in [
            GradleModuleCase::BadRead,
            GradleModuleCase::MissingCapability,
            GradleModuleCase::MissingRootReadCapability,
            GradleModuleCase::MissingRootPhaseCapability,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }
    #[gpui::test]
    async fn gradle_module_policy_lifecycle_rechecks_settings_and_host_after_read(
        cx: &mut TestAppContext,
    ) {
        for case in [
            GradleModuleCase::SettingsDuringRead,
            GradleModuleCase::HostSettingsDuringRead,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }
    #[gpui::test]
    async fn gradle_module_policy_lifecycle_cancels_reads_and_deltas(cx: &mut TestAppContext) {
        for case in [GradleModuleCase::CancelRead, GradleModuleCase::CancelDelta] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }
    #[gpui::test]
    async fn gradle_module_policy_lifecycle_rejects_revocation_and_changed_build_inputs(
        cx: &mut TestAppContext,
    ) {
        for case in [
            GradleModuleCase::RevokeRead,
            GradleModuleCase::BuildDuringRead,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }
    #[gpui::test]
    async fn gradle_module_policy_lifecycle_rejects_private_and_missing_targets(
        cx: &mut TestAppContext,
    ) {
        for case in [
            GradleModuleCase::PrivateTarget,
            GradleModuleCase::MissingTarget,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }
    #[gpui::test]
    async fn gradle_module_policy_lifecycle_stops_on_failed_delta_and_recovers(
        cx: &mut TestAppContext,
    ) {
        for case in [GradleModuleCase::FailedDelta, GradleModuleCase::BadDelta] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn gradle_module_policy_lifecycle_rejects_foreign_provider_targets(
        cx: &mut TestAppContext,
    ) {
        gradle_module_policy_lifecycle(cx, GradleModuleCase::ForeignTarget).await;
    }

    #[gpui::test]
    async fn gradle_module_policy_lifecycle_retires_removed_worktree(cx: &mut TestAppContext) {
        gradle_module_policy_lifecycle(cx, GradleModuleCase::RemoveWorktree).await;
    }

    #[gpui::test]
    async fn gradle_root_policy_lifecycle_closes_and_restores_retained_roots(
        cx: &mut TestAppContext,
    ) {
        for case in [
            GradleModuleCase::RootIgnore,
            GradleModuleCase::RootFinalAbsent,
            GradleModuleCase::RootClosed,
            GradleModuleCase::RootRecoverable,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn gradle_root_policy_lifecycle_imports_fresh_and_recovers_cold_default_roots(
        cx: &mut TestAppContext,
    ) {
        for case in [
            GradleModuleCase::RootFreshDefault,
            GradleModuleCase::RootColdDefault,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn gradle_root_policy_lifecycle_refuses_fresh_and_unconfirmed_roots(
        cx: &mut TestAppContext,
    ) {
        for case in [
            GradleModuleCase::RootFresh,
            GradleModuleCase::RootBadRead,
            GradleModuleCase::RootBadFinal,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn gradle_root_policy_lifecycle_preserves_orphan_guard_and_recovers(
        cx: &mut TestAppContext,
    ) {
        for case in [
            GradleModuleCase::RootOrphan,
            GradleModuleCase::RootFailed,
            GradleModuleCase::RootBadReceipt,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn gradle_root_policy_lifecycle_cancels_root_reads_and_phases(cx: &mut TestAppContext) {
        for case in [
            GradleModuleCase::CancelRootRead,
            GradleModuleCase::CancelRootDelta,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }

    #[gpui::test]
    async fn gradle_root_policy_lifecycle_rechecks_intent_and_host_after_root_read(
        cx: &mut TestAppContext,
    ) {
        for case in [
            GradleModuleCase::SettingsDuringRootRead,
            GradleModuleCase::HostSettingsDuringRootRead,
        ] {
            gradle_module_policy_lifecycle(cx, case).await;
        }
    }

    async fn gradle_module_policy_lifecycle(cx: &mut TestAppContext, case: GradleModuleCase) {
        use project::Fs as _;
        use settings::SettingsStore;
        let default_roots = cx.update(|cx| cx.try_global::<GradleDefaultRootFixture>().copied());
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/module-policy",serde_json::json!({"build.gradle":"plugins { id 'java' }","Main.java":"class Main {}",
            "application":{"build.gradle":"","Main.java":"class Application {}",".project":"application metadata"},
            "library":{"build.gradle":"","Library.java":"class Library {}",".project":"library metadata"},
            "included":{"build.gradle":"","Other.java":"class Other {}",".project":"included metadata"},
            ".env":{"private-module":{"build.gradle":"private"}}})).await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/module-policy".as_ref()], cx).await;
        let worktrees = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktrees.read_with(cx, |store, cx| {
            store.worktrees().next().expect("worktree").read(cx).id()
        });
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust"));
        trust.update(cx, |trusted, cx| {
            trusted.trust(
                &worktrees,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let store = project.read_with(cx, |project, _| project.lsp_store());
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let commands = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let default_events = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let verify_default_order = || {
            if let Some(mode) = default_roots {
                let events = default_events.lock().expect("default events");
                if mode == GradleDefaultRootFixture::MissingCapability {
                    assert!(
                        events.is_empty(),
                        "missing default capability dispatches nothing"
                    );
                } else {
                    assert_eq!(
                        events.first().map(String::as_str),
                        Some("groovy.gradle.defaultImportRoots")
                    );
                    let lookups = events
                        .iter()
                        .filter(|event| event.as_str() == "groovy.gradle.defaultImportRoots")
                        .count();
                    assert_eq!(
                        lookups,
                        if case == GradleModuleCase::FailedDelta {
                            2
                        } else {
                            1
                        }
                    );
                    if matches!(
                        mode,
                        GradleDefaultRootFixture::Empty
                            | GradleDefaultRootFixture::Nested
                            | GradleDefaultRootFixture::Malformed
                            | GradleDefaultRootFixture::Failure
                    ) {
                        assert_eq!(
                            events.len(),
                            1,
                            "unusable default roots never reach mutation or import"
                        );
                        assert!(commands.lock().expect("commands").is_empty());
                    }
                }
            }
        };
        let ignored_root = matches!(
            case,
            GradleModuleCase::RootIgnore
                | GradleModuleCase::RootFinalAbsent
                | GradleModuleCase::RootClosed
                | GradleModuleCase::RootRecoverable
                | GradleModuleCase::RootFresh
                | GradleModuleCase::RootFailed
                | GradleModuleCase::RootBadReceipt
                | GradleModuleCase::RootBadRead
                | GradleModuleCase::RootBadFinal
                | GradleModuleCase::RootOrphan
                | GradleModuleCase::CancelRootRead
                | GradleModuleCase::CancelRootDelta
                | GradleModuleCase::SettingsDuringRootRead
                | GradleModuleCase::HostSettingsDuringRootRead
        );
        let root_state = Arc::new(AtomicUsize::new(match case {
            GradleModuleCase::RootClosed | GradleModuleCase::RootOrphan => 1,
            GradleModuleCase::RootRecoverable | GradleModuleCase::RootColdDefault => 2,
            GradleModuleCase::RootFresh | GradleModuleCase::RootFreshDefault => 3,
            _ => 0,
        }));
        let root_reads = Arc::new(AtomicUsize::new(0));
        let closed_application = Arc::new(AtomicBool::new(case == GradleModuleCase::RootOrphan));
        let closed_library = Arc::new(AtomicBool::new(matches!(
            case,
            GradleModuleCase::Noop | GradleModuleCase::RootOrphan
        )));
        let open_included = Arc::new(AtomicBool::new(case == GradleModuleCase::Noop));
        let fail_once = Arc::new(AtomicBool::new(matches!(
            case,
            GradleModuleCase::FailedDelta
                | GradleModuleCase::BadDelta
                | GradleModuleCase::RootFailed
                | GradleModuleCase::RootBadReceipt
                | GradleModuleCase::RootOrphan
        )));
        let gate = Arc::new(std::sync::Mutex::new(None::<oneshot::Receiver<()>>));
        let (entered_sender, mut entered_receiver) = futures::channel::mpsc::unbounded::<String>();
        let held = match case {
            GradleModuleCase::SettingsDuringRead
            | GradleModuleCase::HostSettingsDuringRead
            | GradleModuleCase::CancelRead
            | GradleModuleCase::RevokeRead
            | GradleModuleCase::BuildDuringRead
            | GradleModuleCase::RemoveWorktree => Some("groovy.gradle.moduleStates"),
            GradleModuleCase::CancelDelta => Some("groovy.gradle.applyModuleStates"),
            GradleModuleCase::CancelRootRead
            | GradleModuleCase::SettingsDuringRootRead
            | GradleModuleCase::HostSettingsDuringRootRead => Some("groovy.gradle.rootState"),
            GradleModuleCase::CancelRootDelta => Some("groovy.gradle.applyRootState"),
            _ => None,
        };
        let held = if default_roots.is_some()
            && matches!(
                case,
                GradleModuleCase::SettingsDuringRead
                    | GradleModuleCase::HostSettingsDuringRead
                    | GradleModuleCase::CancelRead
                    | GradleModuleCase::RevokeRead
                    | GradleModuleCase::BuildDuringRead
                    | GradleModuleCase::RemoveWorktree
            ) {
            Some("groovy.gradle.defaultImportRoots")
        } else {
            held
        };
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../project/test_data/groovy_jvm_gradle/module-state-snapshots-v1.json"
        ))
        .expect("captured snapshots");
        let response = fixture["snapshots"]
            .as_array()
            .expect("snapshots")
            .iter()
            .find(|item| item["label"] == "open")
            .expect("open snapshot")["response"]
            .to_string()
            .replace("file:///groovy-module-state/", "file:///module-policy/");
        let response: serde_json::Value = serde_json::from_str(&response).expect("root remap");
        let mut capabilities = vec![
            "java.project.getAll",
            "java.project.getSettings",
            "groovy.maven.importedProjects",
            "groovy.gradle.importedProjects",
            "groovy.gradle.rootState",
            "groovy.gradle.applyRootState",
            "groovy.gradle.moduleStates",
            "groovy.gradle.importSelected",
        ];
        if case != GradleModuleCase::MissingCapability
            || default_roots == Some(GradleDefaultRootFixture::MissingCapability)
        {
            capabilities.push("groovy.gradle.applyModuleStates");
        }
        if case == GradleModuleCase::MissingRootReadCapability {
            capabilities.retain(|command| *command != "groovy.gradle.rootState");
        }
        if case == GradleModuleCase::MissingRootPhaseCapability {
            capabilities.retain(|command| *command != "groovy.gradle.applyRootState");
        }
        if default_roots.is_some()
            && default_roots != Some(GradleDefaultRootFixture::MissingCapability)
        {
            capabilities.push("groovy.gradle.defaultImportRoots");
        }
        let mut servers = languages.register_fake_lsp(
            "Java",
            FakeLspAdapter {
                name: "jdtls",
                capabilities: lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: capabilities.into_iter().map(str::to_owned).collect(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                initializer: Some(Box::new({
                    let commands = commands.clone();
                    let default_events = default_events.clone();
                    let root_state = root_state.clone();
                    let root_reads = root_reads.clone();
                    let closed_application = closed_application.clone();
                    let closed_library = closed_library.clone();
                    let open_included = open_included.clone();
                    let gate = gate.clone();
                    let fail_once = fail_once.clone();
                    let fs = fs.clone();
                    move |server| {
                        server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                            let commands = commands.clone();
                    let default_events = default_events.clone();
                            let root_state = root_state.clone();
                            let root_reads = root_reads.clone();
                            let closed_application = closed_application.clone();
                            let closed_library = closed_library.clone();
                            let open_included = open_included.clone();
                            let gate = gate.clone();
                            let fail_once = fail_once.clone();
                            let response = response.clone();
                            let fs = fs.clone();
                            let entered_sender = entered_sender.clone();
                            move |params, async_cx| {
                                let commands = commands.clone();
                    let default_events = default_events.clone();
                                let root_state = root_state.clone();
                                let root_reads = root_reads.clone();
                                let closed_application = closed_application.clone();
                                let closed_library = closed_library.clone();
                                let open_included = open_included.clone();
                                let gate = gate.clone();
                                let fail_once = fail_once.clone();
                                let mut response = response.clone();
                                let fs = fs.clone();
                                let entered_sender = entered_sender.clone();
                                let executor = async_cx.background_executor().clone();
                                async move {
                                    let active = || {
                                        let mut values = Vec::new();
                                        if root_state.load(Ordering::SeqCst) == 0 { values.push("file:///module-policy/"); }
                                        if !closed_application.load(Ordering::SeqCst) { values.push("file:///module-policy/application/"); }
                                        if !closed_library.load(Ordering::SeqCst) {
                                            values.push("file:///module-policy/library/");
                                        }
                                        if open_included.load(Ordering::SeqCst) {
                                            values.push("file:///module-policy/included/");
                                        }
                                        values
                                    };
                                    match params.command.as_str() {
                                        "java.project.getAll" => Ok(Some(serde_json::json!(active()))),
                                        "java.project.getSettings" => Ok(Some(serde_json::json!({
                                            "org.eclipse.jdt.ls.core.sourcePaths": [],
                                            "org.eclipse.jdt.ls.core.classpathEntries": []
                                        }))),
                                        "groovy.maven.importedProjects" | "groovy.gradle.importedProjects" => {
                                            let projects = if params.command.starts_with("groovy.gradle.") {
                                                active().into_iter().map(|uri| serde_json::json!({"projectUri": uri, "javaNature": true})).collect::<Vec<_>>()
                                            } else {
                                                vec![]
                                            };
                                            Ok(Some(serde_json::json!({
                                                "version": 1, "rootUri": "file:///module-policy/", "projects": projects,
                                                "partial": false, "fullSynchronizationKnown": false
                                            })))
                                        }
                                        "groovy.gradle.defaultImportRoots" => {
                                            default_events.lock().expect("default events").push(params.command.clone());
                                            entered_sender.unbounded_send(params.command.clone()).expect("default observer");
                                            if held == Some(params.command.as_str()) {
                                                let receiver = gate.lock().expect("gate").take();
                                                if let Some(receiver) = receiver { receiver.await.expect("release default read"); }
                                            }
                                            let input=params.arguments.first().expect("default request");
                                            assert_eq!(input.as_object().expect("request").len(),3);
                                            assert_eq!(input["rootUri"],"file:///module-policy/");
                                            assert_eq!(input["cachedOnly"],true);
                                            assert_eq!(input["candidateBuildFileUris"],serde_json::json!([
                                                "file:///module-policy/application/build.gradle", "file:///module-policy/build.gradle",
                                                "file:///module-policy/included/build.gradle", "file:///module-policy/library/build.gradle"
                                            ]));
                                            if default_roots == Some(GradleDefaultRootFixture::Failure) { anyhow::bail!("private-canary at /private/provider"); }
                                            if default_roots == Some(GradleDefaultRootFixture::Malformed) { return Ok(Some(serde_json::json!({"private-canary":true}))); }
                                            let roots = match default_roots {
                                                Some(GradleDefaultRootFixture::Empty) => vec![],
                                                Some(GradleDefaultRootFixture::Nested) => vec!["file:///module-policy/application/"],
                                                _ => vec!["file:///module-policy/"],
                                            };
                                            Ok(Some(serde_json::json!({"version":1,"rootUri":input["rootUri"],"buildRootUris":roots,
                                                "cachedOnly":true,"readOnly":true,"partial":false,"fullSynchronizationKnown":false})))
                                        }
                                        "groovy.gradle.rootState" | "groovy.gradle.applyRootState" | "groovy.gradle.moduleStates" | "groovy.gradle.applyModuleStates" | "groovy.gradle.importSelected" => {
                                            commands.lock().expect("commands").push(params.command.clone());
                                            default_events.lock().expect("default events").push(params.command.clone());
                                            entered_sender.unbounded_send(params.command.clone()).expect("observer");
                                            if held == Some(params.command.as_str()) {
                                                let receiver = gate.lock().expect("gate").take();
                                                if let Some(receiver) = receiver {
                                                    receiver.await.expect("release");
                                                }
                                            }
                                            let input = params.arguments.first().expect("single request");
                                            assert_eq!(input["rootUri"], "file:///module-policy/");
                                            assert_eq!(input["offline"], true);
                                            if params.command == "groovy.gradle.rootState" {
                                                let read = root_reads.fetch_add(1, Ordering::SeqCst);
                                                if case == GradleModuleCase::RootBadRead { return Ok(Some(serde_json::json!({"private-canary":"/private/provider"}))); }
                                                let state = root_state.load(Ordering::SeqCst);
                                                let status = if case == GradleModuleCase::RootBadFinal && read > 0 || state == 0 { "open" } else if state == 1 { "closed" } else { "absent" };
                                                return Ok(Some(serde_json::json!({"version":1,"rootUri":input["rootUri"],"workspaceState":status,"descriptionAvailable":state != 3,"fullSynchronizationKnown":false})));
                                            }
                                            if params.command == "groovy.gradle.applyRootState" {
                                                let failing = fail_once.swap(false, Ordering::SeqCst);
                                                if matches!(case, GradleModuleCase::RootFailed | GradleModuleCase::RootOrphan) && failing { anyhow::bail!("private-canary at /private/provider"); }
                                                let open = input["open"].as_bool().expect("root open flag");
                                                if !open { assert!(closed_application.load(Ordering::SeqCst) && closed_library.load(Ordering::SeqCst) && !open_included.load(Ordering::SeqCst), "children closed before root"); }
                                                root_state.store(if open {0} else if case == GradleModuleCase::RootFinalAbsent {2} else {1}, Ordering::SeqCst);
                                                fs.insert_tree("/module-policy",serde_json::json!({".project":format!("root open={open}")})).await;
                                                executor.timer(std::time::Duration::from_millis(300)).await;
                                                return Ok(Some(serde_json::json!({"version":1,"rootUri":input["rootUri"],"open":open,"rootStatePhaseComplete":true,"fullSynchronizationKnown":case == GradleModuleCase::RootBadReceipt && failing})));
                                            }
                                            if params.command == "groovy.gradle.moduleStates" {
                                                if case == GradleModuleCase::BadRead {
                                                    return Ok(Some(serde_json::json!({"private-canary": "private-value"})));
                                                }
                                                for module in response["modules"].as_array_mut().expect("records") {
                                                    if module["moduleUri"] == "file:///module-policy/application/" {
                                                        module["workspaceState"] = serde_json::json!(if closed_application.load(Ordering::SeqCst) { "closed" } else { "open" });
                                                    }
                                                    if module["moduleUri"] == "file:///module-policy/library/" {
                                                        let state = if closed_library.load(Ordering::SeqCst) { "closed" } else { "open" };
                                                        module["workspaceState"] = serde_json::json!(state);
                                                    }
                                                    if module["moduleUri"] == "file:///module-policy/included/" {
                                                        let state = if open_included.load(Ordering::SeqCst) { "open" } else { "closed" };
                                                        module["workspaceState"] = serde_json::json!(state);
                                                        if matches!(case, GradleModuleCase::PrivateTarget | GradleModuleCase::MissingTarget) {
                                                            let uri = if case == GradleModuleCase::PrivateTarget {
                                                                "file:///module-policy/.env/private-module/"
                                                            } else {
                                                                "file:///module-policy/missing/"
                                                            };
                                                            module["moduleUri"] = serde_json::json!(uri);
                                                            module["buildRootUri"] = serde_json::json!("file:///module-policy/");
                                                        }
                                                    }
                                                }
                                                if case == GradleModuleCase::ForeignTarget {
                                                    response["modules"][0]["moduleUri"] = serde_json::json!("file:///other-private-root/");
                                                }
                                                return Ok(Some(response));
                                            }
                                            if params.command == "groovy.gradle.applyModuleStates" {
                                                let failing = matches!(case, GradleModuleCase::FailedDelta | GradleModuleCase::BadDelta) && fail_once.swap(false, Ordering::SeqCst);
                                                if case == GradleModuleCase::FailedDelta && failing {
                                                    anyhow::bail!("private-canary at /private/provider");
                                                }
                                                for (field, closing) in [("closedProjectUris", true), ("restoredProjectUris", false)] {
                                                    for uri in input[field].as_array().expect("mutation paths") {
                                                        if uri == "file:///module-policy/application/" {
                                                            closed_application.store(closing, Ordering::SeqCst);
                                                        } else if uri == "file:///module-policy/library/" {
                                                            closed_library.store(closing, Ordering::SeqCst);
                                                        } else if uri == "file:///module-policy/included/" {
                                                            open_included.store(!closing, Ordering::SeqCst);
                                                        } else {
                                                            panic!("unexpected mutation target");
                                                        }
                                                    }
                                                }
                                                fs.insert_tree("/module-policy/library", serde_json::json!({
                                                    ".project": format!("sdk metadata closed={}", closed_library.load(Ordering::SeqCst))
                                                })).await;
                                                executor.timer(std::time::Duration::from_millis(300)).await;
                                                let invalid_receipt = case == GradleModuleCase::BadDelta && failing;
                                                return Ok(Some(serde_json::json!({
                                                    "version": 1, "rootUri": input["rootUri"],
                                                    "closedProjectUris": input["closedProjectUris"], "restoredProjectUris": input["restoredProjectUris"],
                                                    "moduleStatePhaseComplete": true, "fullSynchronizationKnown": invalid_receipt
                                                })));
                                            }
                                            assert_eq!(params.command, "groovy.gradle.importSelected");
                                            root_state.store(0, Ordering::SeqCst);
                                            Ok(Some(serde_json::json!({
                                                "version": 1, "rootUri": input["rootUri"],
                                                "selectedImportPhaseComplete": true, "fullSynchronizationKnown": false
                                            })))
                                        }
                                        _ => anyhow::bail!("unexpected fixture command")
                                    }
                                }
                            }
                        });
                    }
                })),
                ..Default::default()
            },
        );
        let set_settings = |cx: &mut TestAppContext, ignored: &[&str], host: bool| {
            let mut value = serde_json::json!({"jvm_builds":{"gradle":{"selected_build_files":["build.gradle"],"ignored_projects":ignored}}});
            if default_roots.is_some() {
                assert!(
                    value["jvm_builds"]["gradle"]
                        .as_object_mut()
                        .expect("Gradle settings")
                        .remove("selected_build_files")
                        .is_some()
                );
            }
            if host {
                value["lsp"] =
                    serde_json::json!({"jdtls":{"settings":{"fixture_host_option":true}}});
            }
            cx.update(|cx| {
                cx.update_global(|store: &mut SettingsStore, cx| {
                    store
                        .set_user_settings(&value.to_string(), cx)
                        .result()
                        .expect("settings");
                })
            });
        };
        let automatic = cx.update(|cx| cx.has_global::<GradleStartupPolicyFixture>());
        if automatic {
            set_settings(cx, if ignored_root { &[""] } else { &["library"] }, false);
        }
        let (buffer, _open) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/module-policy/Main.java", cx)
            })
            .await
            .expect("Java buffer");
        let _server = servers.next().await.expect("server");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        if automatic {
            for _ in 0..40 {
                cx.run_until_parked();
                let finished = store.read_with(cx, |store, _| {
                    store.jdtls_project_import_failed(worktree_id)
                        || (!commands.lock().expect("commands").is_empty()
                            && store.jdtls_project_import_token(worktree_id).is_none()
                            && store
                                .jdtls_project_snapshot(worktree_id)
                                .is_some_and(|(_, current)| current))
                });
                if finished {
                    break;
                }
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
            }
            let observed = commands.lock().expect("commands").clone();
            let failed = store.read_with(cx, |store, _| {
                store.jdtls_project_import_failed(worktree_id)
            });
            assert!(
                store.read_with(cx, |store, _| store
                    .jdtls_project_import_token(worktree_id)
                    .is_none()),
                "startup policy reached a terminal phase"
            );
            if matches!(
                case,
                GradleModuleCase::Success | GradleModuleCase::RootIgnore
            ) {
                assert!(!failed, "automatic saved policy succeeds");
                assert_eq!(
                    observed,
                    vec![
                        "groovy.gradle.rootState",
                        "groovy.gradle.moduleStates",
                        "groovy.gradle.applyModuleStates",
                        if ignored_root {
                            "groovy.gradle.applyRootState"
                        } else {
                            "groovy.gradle.importSelected"
                        },
                        "groovy.gradle.rootState"
                    ]
                );
                assert!(closed_library.load(Ordering::SeqCst));
                assert_eq!(root_state.load(Ordering::SeqCst), usize::from(ignored_root));
                store.read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("current startup facts");
                    assert!(current);
                    assert!(
                        !snapshot.projects.iter().any(|project| project
                            .module_root
                            .path
                            .as_unix_str()
                            == "library")
                    );
                });
            } else {
                assert!(
                    failed,
                    "failed automatic policy is retained for explicit recovery"
                );
                assert!(
                    !observed
                        .iter()
                        .any(|command| command == "groovy.gradle.importSelected")
                );
                if case == GradleModuleCase::MissingCapability {
                    assert!(observed.is_empty());
                }
                if case == GradleModuleCase::RootFresh {
                    assert_eq!(observed, vec!["groovy.gradle.rootState"]);
                }
                if case == GradleModuleCase::FailedDelta {
                    assert_eq!(
                        observed,
                        vec![
                            "groovy.gradle.rootState",
                            "groovy.gradle.moduleStates",
                            "groovy.gradle.applyModuleStates"
                        ]
                    );
                    store
                        .update(cx, |store, cx| {
                            store.import_selected_local_gradle_project(worktree_id, cx)
                        })
                        .expect("explicit retry")
                        .await
                        .expect("startup failure recovery");
                    assert!(!store.read_with(cx, |store, _| {
                        store.jdtls_project_import_failed(worktree_id)
                    }));
                    assert_eq!(
                        commands
                            .lock()
                            .expect("commands")
                            .last()
                            .map(String::as_str),
                        Some("groovy.gradle.rootState")
                    );
                }
            }
            verify_default_order();
            return;
        }
        set_settings(cx, if ignored_root { &[""] } else { &["library"] }, false);
        cx.run_until_parked();
        assert!(
            commands.lock().expect("commands").is_empty(),
            "settings and reads do not execute policy"
        );
        let (release, receiver) = oneshot::channel();
        if held.is_some() {
            *gate.lock().expect("gate") = Some(receiver);
        } else {
            drop(receiver);
        }
        let operation = store.update(cx, |store, cx| {
            store.import_selected_local_gradle_project(worktree_id, cx)
        });
        if matches!(
            case,
            GradleModuleCase::MissingCapability
                | GradleModuleCase::MissingRootReadCapability
                | GradleModuleCase::MissingRootPhaseCapability
        ) || default_roots == Some(GradleDefaultRootFixture::MissingCapability)
        {
            let message = operation.expect_err("missing delta capability").to_string();
            assert!(message.contains("extension setup"));
            assert!(commands.lock().expect("commands").is_empty());
            if default_roots == Some(GradleDefaultRootFixture::MissingCapability) {
                assert!(message.contains("Gradle default-root discovery provider"));
            }
            verify_default_order();
            return;
        }
        let operation = operation.expect("explicit module policy");
        if let Some(held) = held {
            loop {
                if entered_receiver.next().await.expect("phase entered") == held {
                    break;
                }
            }
            assert!(store.read_with(cx, |store, _| {
                store.jdtls_project_import_token(worktree_id).is_some()
            }));
            assert!(
                !store.read_with(cx, |store, _| store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)),
                "phase receipt does not publish project data"
            );
            if default_roots == Some(GradleDefaultRootFixture::StoppedServer) {
                project.update(cx, |project, cx| {
                    project.stop_language_servers_for_buffers(
                        vec![buffer.clone()],
                        HashSet::default(),
                        cx,
                    )
                });
            } else {
                match case {
                    GradleModuleCase::SettingsDuringRootRead
                    | GradleModuleCase::SettingsDuringRead => {
                        set_settings(cx, &["included"], false)
                    }
                    GradleModuleCase::HostSettingsDuringRootRead
                    | GradleModuleCase::HostSettingsDuringRead => {
                        set_settings(cx, &["library"], true)
                    }
                    GradleModuleCase::CancelRootRead
                    | GradleModuleCase::CancelRootDelta
                    | GradleModuleCase::CancelRead
                    | GradleModuleCase::CancelDelta => assert!(
                        store
                            .update(cx, |store, cx| store
                                .cancel_selected_gradle_import(worktree_id, cx))
                            .expect("cancel")
                    ),
                    GradleModuleCase::RevokeRead => trust.update(cx, |trusted, cx| {
                        trusted.restrict(
                            worktrees.downgrade(),
                            HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                            cx,
                        )
                    }),
                    GradleModuleCase::BuildDuringRead => {
                        if default_roots == Some(GradleDefaultRootFixture::RemovedInput) {
                            fs.remove_file(
                                std::path::Path::new("/module-policy/build.gradle"),
                                Default::default(),
                            )
                            .await
                            .expect("remove candidate");
                        } else if default_roots == Some(GradleDefaultRootFixture::AliasedInput) {
                            fs.insert_tree(
                                "/default-root-fixture-foreign",
                                serde_json::json!({"build.gradle":"private-canary"}),
                            )
                            .await;
                            fs.insert_symlink(
                                "/module-policy/build.gradle",
                                std::path::PathBuf::from(
                                    "/default-root-fixture-foreign/build.gradle",
                                ),
                            )
                            .await;
                        } else {
                            fs.insert_file(
                                std::path::Path::new("/module-policy/build.gradle"),
                                b"plugins { id 'java' } // changed".to_vec(),
                            )
                            .await;
                        }
                    }
                    GradleModuleCase::RemoveWorktree => {
                        project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx));
                    }
                    _ => panic!("unexpected held case"),
                }
            }
            cx.run_until_parked();
            if release.send(()).is_err() {
                assert!(
                    matches!(
                        case,
                        GradleModuleCase::CancelRootRead
                            | GradleModuleCase::CancelRootDelta
                            | GradleModuleCase::SettingsDuringRootRead
                            | GradleModuleCase::HostSettingsDuringRootRead
                            | GradleModuleCase::CancelRead
                            | GradleModuleCase::CancelDelta
                            | GradleModuleCase::RevokeRead
                            | GradleModuleCase::BuildDuringRead
                            | GradleModuleCase::SettingsDuringRead
                            | GradleModuleCase::HostSettingsDuringRead
                            | GradleModuleCase::RemoveWorktree
                    ),
                    "unexpected provider drop"
                );
            }
        } else {
            drop(release);
        }
        let result = operation.await;
        if matches!(
            case,
            GradleModuleCase::Success
                | GradleModuleCase::Noop
                | GradleModuleCase::RootFinalAbsent
                | GradleModuleCase::RootFreshDefault
                | GradleModuleCase::RootColdDefault
                | GradleModuleCase::RootIgnore
                | GradleModuleCase::RootClosed
                | GradleModuleCase::RootRecoverable
        ) {
            result.expect("policy success");
            let mut expected = vec!["groovy.gradle.rootState"];
            if case == GradleModuleCase::RootColdDefault {
                expected.push("groovy.gradle.applyRootState");
            }
            expected.push("groovy.gradle.moduleStates");
            if case != GradleModuleCase::Noop {
                expected.push("groovy.gradle.applyModuleStates");
            }
            expected.push(if ignored_root {
                "groovy.gradle.applyRootState"
            } else {
                "groovy.gradle.importSelected"
            });
            expected.push("groovy.gradle.rootState");
            assert_eq!(*commands.lock().expect("commands"), expected);
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            if case == GradleModuleCase::Success || ignored_root {
                store.read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("refreshed facts");
                    assert!(current);
                    assert!(
                        !snapshot.projects.iter().any(|project| project
                            .module_root
                            .path
                            .as_unix_str()
                            == "library")
                    );
                });
                set_settings(cx, &[], false);
                cx.run_until_parked();
                commands.lock().expect("commands").clear();
                store
                    .update(cx, |store, cx| {
                        store.import_selected_local_gradle_project(worktree_id, cx)
                    })
                    .expect("removed ignore")
                    .await
                    .expect("restore policy");
                assert!(!closed_library.load(Ordering::SeqCst));
                assert_eq!(
                    *commands.lock().expect("commands"),
                    if ignored_root {
                        vec![
                            "groovy.gradle.rootState",
                            "groovy.gradle.applyRootState",
                            "groovy.gradle.moduleStates",
                            "groovy.gradle.applyModuleStates",
                            "groovy.gradle.importSelected",
                            "groovy.gradle.rootState",
                        ]
                    } else {
                        vec![
                            "groovy.gradle.rootState",
                            "groovy.gradle.moduleStates",
                            "groovy.gradle.applyModuleStates",
                            "groovy.gradle.importSelected",
                            "groovy.gradle.rootState",
                        ]
                    }
                );
            }
        } else {
            let message = result.expect_err("rejected policy").to_string();
            if matches!(
                case,
                GradleModuleCase::RootFresh | GradleModuleCase::RootBadRead
            ) {
                assert_eq!(
                    *commands.lock().expect("commands"),
                    vec!["groovy.gradle.rootState"]
                );
            }
            if case == GradleModuleCase::RootFresh {
                assert!(message.contains("not yet supported"));
                assert!(message.contains("Keep the ignore setting"));
                assert!(message.contains("this action did not request an import"));
            }
            if case == GradleModuleCase::RootBadFinal {
                assert!(message.contains("not confirmed"));
            }
            if case == GradleModuleCase::RootOrphan {
                assert!(
                    commands
                        .lock()
                        .expect("commands")
                        .contains(&"groovy.gradle.applyRootState".into()),
                    "already-closed root still invokes SDK preflight"
                );
            }

            assert!(!message.contains("private-canary") && !message.contains("/private/provider"));
            assert!(
                !commands
                    .lock()
                    .expect("commands")
                    .iter()
                    .any(|command| command == "groovy.gradle.importSelected"),
                "failed read/delta never imports root"
            );
            if matches!(
                case,
                GradleModuleCase::FailedDelta
                    | GradleModuleCase::BadDelta
                    | GradleModuleCase::RootFailed
                    | GradleModuleCase::RootBadReceipt
                    | GradleModuleCase::RootOrphan
            ) {
                commands.lock().expect("commands").clear();
                store
                    .update(cx, |store, cx| {
                        store.import_selected_local_gradle_project(worktree_id, cx)
                    })
                    .expect("retry")
                    .await
                    .expect("recovery");
                assert_eq!(
                    commands
                        .lock()
                        .expect("commands")
                        .last()
                        .map(String::as_str),
                    Some("groovy.gradle.rootState")
                );
            }
        }
        verify_default_order();
    }

    struct ImportMetadataFixture;
    impl gpui::Global for ImportMetadataFixture {}

    #[gpui::test]
    async fn selected_gradle_import_keeps_ownership_during_sdk_metadata_writes(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| cx.set_global(ImportMetadataFixture));
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, true, true, false, false, false, false, false,
        )
        .await;
    }

    #[gpui::test]
    async fn selected_maven_import_keeps_ownership_during_sdk_metadata_writes(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| cx.set_global(ImportMetadataFixture));
        selected_import_preserves_progress_and_rejects_changed_bindings(
            cx, false, true, false, false, false, false, false,
        )
        .await;
    }

    async fn selected_import_preserves_progress_and_rejects_changed_bindings(
        cx: &mut TestAppContext,
        gradle: bool,
        guarded_provider: bool,
        unlink: bool,
        non_java: bool,
        resources_unavailable: bool,
        reactor: bool,
        panel_controls: bool,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        enum ProjectEvent {}
        impl lsp::notification::Notification for ProjectEvent {
            type Params = serde_json::Value;
            const METHOD: &'static str = "language/eventNotification";
        }
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-import",
            serde_json::json!({
                "pom.xml": "<project/>", "build.gradle": "plugins { id 'java' }", "Main.java": "class Main {}",
                "module": {"pom.xml": "<project/>"}, "ignored": {"pom.xml": "<project/>"}, ".env": {"pom.xml": "private"}
            }),
        )
        .await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/groovy-import".as_ref()], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktree_store.read_with(cx, |store, cx| {
            store.worktrees().next().expect("worktree").read(cx).id()
        });
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        let import_selected = |store: &mut project::LspStore,
                               poms: Vec<ProjectPath>,
                               cx: &mut Context<project::LspStore>| {
            if unlink {
                if gradle {
                    store.unlink_selected_local_gradle_projects(worktree_id, poms, cx)
                } else {
                    store.unlink_selected_local_maven_projects(worktree_id, poms, cx)
                }
            } else if gradle {
                store.import_selected_local_gradle_project(worktree_id, cx)
            } else if reactor {
                anyhow::ensure!(poms.len() == 1, "Select one reactor root");
                store.import_local_maven_reactor(
                    worktree_id,
                    poms.first()
                        .ok_or_else(|| anyhow::anyhow!("Missing reactor root"))?
                        .clone(),
                    vec![path(worktree_id, "ignored")],
                    cx,
                )
            } else {
                store.import_selected_local_maven_projects(worktree_id, poms, cx)
            }
        };

        let poms = if unlink {
            vec![path(worktree_id, "")]
        } else if reactor {
            vec![path(worktree_id, "pom.xml")]
        } else {
            vec![
                path(worktree_id, "pom.xml"),
                path(worktree_id, "module/pom.xml"),
            ]
        };
        let missing_trust = cx.update(|cx| cx.remove_global::<TrustedWorktrees>());
        assert!(
            lsp_store
                .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                .is_err()
        );
        cx.update(|cx| cx.set_global(missing_trust));
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"));
        trust.update(cx, |trusted, cx| {
            trusted.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        assert!(
            lsp_store
                .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                .is_err(),
            "missing provider"
        );
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let metadata_outputs = cx.update(|cx| cx.has_global::<ImportMetadataFixture>());
        let calls = Arc::new(AtomicUsize::new(0));
        let imported_kind_mode = Arc::new(AtomicUsize::new(0));
        let reactor_reply_mode = Arc::new(AtomicUsize::new(0));
        let fail_import = Arc::new(AtomicBool::new(false));
        let import_gate = Arc::new(std::sync::Mutex::new(None::<oneshot::Receiver<()>>));
        let mut servers = languages.register_fake_lsp("Java", FakeLspAdapter {
            name: "jdtls", capabilities: lsp::ServerCapabilities {
                execute_command_provider: Some(lsp::ExecuteCommandOptions { commands: if guarded_provider { vec!["java.project.getAll".into(), "java.project.getSettings".into(), "groovy.maven.importSelected".into(), "groovy.maven.importReactor".into(), "groovy.gradle.importSelected".into(), "groovy.gradle.rootState".into(), "groovy.gradle.applyRootState".into(), "groovy.gradle.moduleStates".into(), "groovy.gradle.applyModuleStates".into(), "groovy.maven.unlinkSelected".into(), "groovy.gradle.unlinkSelected".into(), "groovy.maven.importedProjects".into(), "groovy.gradle.importedProjects".into()] } else { vec!["java.project.getAll".into(), "java.project.getSettings".into(), "groovy.maven.importSelected".into(), "java.project.changeImportedProjects".into()] }, ..Default::default() }), ..Default::default()
            }, initializer: Some(Box::new({
                let fs = fs.clone();
                let calls = calls.clone(); let fail_import = fail_import.clone(); let import_gate = import_gate.clone(); let imported_kind_mode = imported_kind_mode.clone(); let reactor_reply_mode = reactor_reply_mode.clone();
                move |server| {
                    server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                        let fs = fs.clone();
                        let calls = calls.clone(); let fail_import = fail_import.clone(); let import_gate = import_gate.clone(); let imported_kind_mode = imported_kind_mode.clone(); let reactor_reply_mode = reactor_reply_mode.clone();
                        move |params, async_cx| {
                            let fs = fs.clone();
                            let executor = async_cx.background_executor().clone();
                            let calls = calls.clone(); let fail_import = fail_import.clone(); let import_gate = import_gate.clone(); let imported_kind_mode = imported_kind_mode.clone(); let reactor_reply_mode = reactor_reply_mode.clone();
                            async move {
                                match params.command.as_str() {
                                    "java.project.getAll" => {
                                        if !params.arguments.is_empty() && resources_unavailable {
                                            anyhow::bail!("imported resource enumeration unavailable");
                                        }
                                        if params.arguments.is_empty() && non_java {
                                            Ok(Some(serde_json::json!([])))
                                        } else {
                                            if !params.arguments.is_empty() {
                                                assert_eq!(params.arguments, [serde_json::json!(r#"{"includeNonJava":true}"#)]);
                                            }
                                            Ok(Some(serde_json::json!(["file:///groovy-import"])))
                                        }
                                    },
                                    "groovy.maven.importedProjects" | "groovy.gradle.importedProjects" => {
                                        assert_eq!(params.arguments, [serde_json::json!({"rootUri":"file:///groovy-import/","cachedOnly":true})]);
                                        let ownership_mode = imported_kind_mode.load(Ordering::SeqCst);
                                        let chosen = if gradle != (ownership_mode == 2) { "groovy.gradle.importedProjects" } else { "groovy.maven.importedProjects" };
                                        let projects = if ownership_mode != 1 && params.command == chosen {
                                            serde_json::json!([{"projectUri":"file:///groovy-import/","javaNature":!non_java}])
                                        } else { serde_json::json!([]) };
                                        Ok(Some(serde_json::json!({"version":1,"rootUri":"file:///groovy-import/","projects":projects,"partial":false,"fullSynchronizationKnown":false})))
                                    }
                                    "java.project.getSettings" => {
                                        assert!(!non_java, "non-Java resources must not request Java semantics");
                                        Ok(Some(serde_json::json!({"org.eclipse.jdt.ls.core.sourcePaths": [], "org.eclipse.jdt.ls.core.classpathEntries": []})))
                                    },
                                    "groovy.gradle.rootState" => Ok(Some(serde_json::json!({"version":1,"rootUri":params.arguments[0]["rootUri"],"workspaceState":"open","descriptionAvailable":true,"fullSynchronizationKnown":false}))),
                                    "groovy.gradle.moduleStates" => Ok(Some(serde_json::json!({"version":1,"rootUri":params.arguments[0]["rootUri"],"modules":[],"fullSynchronizationKnown":false}))),
                                    "groovy.maven.importSelected" | "groovy.maven.importReactor" | "groovy.gradle.importSelected" | "groovy.maven.unlinkSelected" | "groovy.gradle.unlinkSelected" => {
                                        calls.fetch_add(1, Ordering::SeqCst);
                                        let gate = import_gate.lock().expect("gate").take();
                                        if let Some(gate) = gate { gate.await.expect("release provider"); }
                                        if fail_import.load(Ordering::SeqCst) { anyhow::bail!("secret at /private/provider/cache"); }
                                        if metadata_outputs && matches!(params.command.as_str(), "groovy.gradle.importSelected" | "groovy.maven.importSelected" | "groovy.maven.importReactor") {
                                            fs.insert_tree("/groovy-import",serde_json::json!({".project":format!("<projectDescription><name>sdk-owned-{}</name></projectDescription>",calls.load(Ordering::SeqCst))})).await;
                                            executor.timer(std::time::Duration::from_secs(1)).await;
                                        }
                                        assert_eq!(params.arguments.len(), 1);
                                        if params.command.ends_with("unlinkSelected") {
                                            assert_eq!(params.arguments[0].as_object().expect("arguments").len(), 2);
                                            return Ok(Some(serde_json::json!({ "version": 1, "rootUri": params.arguments[0]["rootUri"], "unlinkedProjectUris": params.arguments[0]["projectUris"], "selectedUnlinkPhaseComplete": true, "fullSynchronizationKnown": false })));
                                        }
                                        assert_eq!(params.arguments[0]["offline"], true);
                                        let mut receipt = serde_json::json!({ "version": 1, "rootUri": params.arguments[0]["rootUri"], "selectedImportPhaseComplete": true, "fullSynchronizationKnown": false });
                                        if params.command == "groovy.maven.importReactor" {
                                            assert_eq!(params.arguments[0].as_object().expect("arguments").len(), 4);
                                            assert_eq!(params.arguments[0]["excludedProjectUris"], serde_json::json!(["file:///groovy-import/ignored/"]));
                                            receipt["reactorPomUri"] = params.arguments[0]["pomUri"].clone();
                                            receipt["excludedProjectUris"] = params.arguments[0]["excludedProjectUris"].clone();
                                            let expanded_pom = match reactor_reply_mode.load(Ordering::SeqCst) {
                                                1 => "file:///groovy-import/.env/pom.xml",
                                                2 => "file:///groovy-import/missing/pom.xml",
                                                3 => "file:///groovy-import/ignored/pom.xml",
                                                _ => "file:///groovy-import/module/pom.xml",
                                            };
                                            receipt["importedPomUris"] = serde_json::json!([params.arguments[0]["pomUri"], expanded_pom]);
                                        } else if params.command == "groovy.maven.importSelected" {
                                            assert_eq!(params.arguments[0].as_object().expect("arguments").len(), 3);
                                            receipt["importedPomUris"] = params.arguments[0]["pomUris"].clone();
                                        } else {
                                            assert_eq!(params.arguments[0].as_object().expect("arguments").len(), 2);
                                        }
                                        Ok(Some(receipt))
                                    }
                                    _ => anyhow::bail!("unexpected command"),
                                }
                            }
                        }
                    });
                }
            })), ..Default::default()
        });
        let (buffer, _open_handle) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/groovy-import/Main.java", cx)
            })
            .await
            .expect("Java buffer");
        let mut server = servers.next().await.expect("provider");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("panel");
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.refresh(cx);
        });
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "startup and panel reads never import"
        );
        lsp_store.read_with(cx, |store, _| {
            let (snapshot, current) = store
                .jdtls_project_snapshot(worktree_id)
                .expect("current imported kind snapshot");
            assert!(current);
            let expected = if !guarded_provider {
                project::jvm_project::JdtlsImportedBuildKind::Unknown
            } else if gradle {
                project::jvm_project::JdtlsImportedBuildKind::Gradle
            } else {
                project::jvm_project::JdtlsImportedBuildKind::Maven
            };
            assert_eq!(snapshot.projects[0].imported_build_kind, expected);
        });
        if non_java {
            lsp_store.read_with(cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("resource snapshot");
                assert!(current);
                assert_eq!(snapshot.projects.len(), 1);
                assert_eq!(
                    snapshot.projects[0].membership,
                    JdtlsProjectMembership::ImportedNonJava
                );
                assert!(snapshot.projects[0].source_roots.is_empty());
                assert!(!snapshot.projects[0].project_vm_known);
            });
            panel.read_with(cx, |panel, _| {
                assert!(
                    panel.host.visible_rows()[0]
                        .accessibility_label
                        .contains("imported non-Java project; Java semantics unavailable")
                );
            });
        }
        if resources_unavailable {
            lsp_store.read_with(cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("partial Java membership");
                assert!(current && snapshot.partial);
                assert_eq!(
                    snapshot.projects[0].membership,
                    JdtlsProjectMembership::ImportedJava
                );
            });
        }
        if !guarded_provider {
            let error = lsp_store
                .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                .expect_err("unguarded provider cannot import")
                .to_string();
            assert!(error.contains(if unlink {
                "provider ownership is unavailable"
            } else {
                "provider is unavailable"
            }));
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "never substitute the upstream acknowledgement command"
            );
            return;
        }
        if unlink {
            let other_provider = lsp_store
                .update(cx, |store, cx| {
                    if gradle {
                        store.unlink_selected_local_maven_projects(worktree_id, poms.clone(), cx)
                    } else {
                        store.unlink_selected_local_gradle_projects(worktree_id, poms.clone(), cx)
                    }
                })
                .expect_err("wrong provider must deny before dispatch")
                .to_string();
            assert!(other_provider.contains("provider ownership is unavailable or different"));
            assert!(!other_provider.contains("groovy-import"));
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            for ownership_mode in [1, 2, 0] {
                imported_kind_mode.store(ownership_mode, Ordering::SeqCst);
                lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
                assert!(
                    lsp_store.read_with(cx, |store, _| {
                        store
                            .jdtls_project_snapshot(worktree_id)
                            .is_some_and(|(_, current)| !current)
                            && store.jdtls_project_query_pending(worktree_id)
                    }),
                    "explicit refresh invalidates current ownership before the replacement read"
                );
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
                lsp_store.read_with(cx, |store, _| {
                    let (snapshot, current) = store
                        .jdtls_project_snapshot(worktree_id)
                        .expect("refreshed ownership");
                    assert!(current);
                    let expected = match ownership_mode {
                        1 => project::jvm_project::JdtlsImportedBuildKind::Unknown,
                        2 if gradle => project::jvm_project::JdtlsImportedBuildKind::Maven,
                        2 => project::jvm_project::JdtlsImportedBuildKind::Gradle,
                        _ if gradle => project::jvm_project::JdtlsImportedBuildKind::Gradle,
                        _ => project::jvm_project::JdtlsImportedBuildKind::Maven,
                    };
                    assert_eq!(snapshot.projects[0].imported_build_kind, expected);
                    assert!(!store.jdtls_project_query_pending(worktree_id));
                });
                if ownership_mode != 0 {
                    let error = lsp_store
                        .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                        .expect_err("unknown or replaced provider ownership must deny")
                        .to_string();
                    assert!(error.contains("provider ownership is unavailable or different"));
                    assert!(!error.contains("groovy-import"));
                    assert_eq!(
                        calls.load(Ordering::SeqCst),
                        0,
                        "never dispatch a mutation for unknown or other-provider ownership"
                    );
                    assert!(lsp_store.read_with(cx, |store, _| {
                        store
                            .jdtls_project_snapshot(worktree_id)
                            .is_some_and(|(_, current)| current)
                    }));
                }
            }
        }
        if !gradle || unlink {
            for selection in [
                vec![],
                vec![path(worktree_id, "missing/pom.xml")],
                vec![path(worktree_id, ".env/pom.xml")],
                vec![path(worktree_id, "Main.java")],
                vec![path(WorktreeId::from_usize(999), "pom.xml")],
                vec![poms[0].clone(), poms[0].clone()],
            ] {
                assert!(
                    lsp_store
                        .update(cx, |store, cx| import_selected(store, selection, cx))
                        .is_err()
                );
            }
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        if gradle && !unlink {
            use project::Fs as _;
            let remove = || project::RemoveOptions {
                recursive: false,
                ignore_if_not_exists: false,
            };
            fs.remove_file(
                std::path::Path::new("/groovy-import/build.gradle"),
                remove(),
            )
            .await
            .expect("remove selected build input");
            cx.run_until_parked();
            assert!(
                lsp_store
                    .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                    .is_err(),
                "missing Gradle build input"
            );
            fs.insert_file(
                "/groovy-import/build.gradle",
                b"plugins { id 'java' }".to_vec(),
            )
            .await;
            fs.insert_tree(
                "/outside",
                serde_json::json!({"settings.gradle": "private outside input"}),
            )
            .await;
            fs.insert_symlink(
                "/groovy-import/settings.gradle",
                std::path::PathBuf::from("/outside/settings.gradle"),
            )
            .await;
            cx.run_until_parked();
            assert!(
                lsp_store
                    .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                    .is_err(),
                "unsafe sibling cannot hide behind a valid build input"
            );
            fs.rename(
                std::path::Path::new("/groovy-import/settings.gradle"),
                std::path::Path::new("/outside/rejected-settings.gradle"),
                project::RenameOptions::default(),
            )
            .await
            .expect("move rejected symlink outside the selected fixture");
            cx.run_until_parked();
            assert!(
                lsp_store
                    .update(cx, |store, cx| store.import_selected_local_gradle_project(
                        WorktreeId::from_usize(999),
                        cx
                    ))
                    .is_err(),
                "missing selected host worktree"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }
        let gradle_root_settings = r#"{"jvm_builds":{"gradle":{"selected_build_files":["build.gradle"],"ignored_projects":[]}}}"#;
        let set_gradle_user = |cx: &mut gpui::VisualTestContext, text: &str| {
            cx.update(|_, cx| {
                cx.update_global(|store: &mut settings::SettingsStore, cx| {
                    store
                        .set_user_settings(text, cx)
                        .result()
                        .expect("Gradle user settings");
                });
            });
        };
        if gradle && !unlink {
            for settings in [
                r#"{"jvm_builds":{"gradle":{"selected_build_files":[]}}}"#,
                r#"{"jvm_builds":{"gradle":{"selected_build_files":["private-canary/build.gradle"]}}}"#,
                r#"{"jvm_builds":{"gradle":{"selected_build_files":["nested/build.gradle"],"ignored_projects":[""]}}}"#,
                r#"{"jvm_builds":{"gradle":{"ignored_projects":["private-canary"]}}}"#,
                r#"{"jvm_builds":{"gradle":{"selected_build_files":["settings.gradle.kts"]}}}"#,
            ] {
                set_gradle_user(cx, settings);
                cx.run_until_parked();
                let error = lsp_store
                    .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                    .expect_err("unsupported or unavailable saved Gradle selection")
                    .to_string();
                assert!(error.contains("Gradle"));
                assert!(!error.contains("private-canary"));
                assert_eq!(calls.load(Ordering::SeqCst), 0);
            }
            cx.update(|_, cx| {
                cx.update_global(|store: &mut settings::SettingsStore, cx| {
                    store
                        .set_local_settings(
                            worktree_id,
                            settings::LocalSettingsPath::InWorktree(RelPath::empty().into()),
                            settings::LocalSettingsKind::Settings,
                            Some(gradle_root_settings),
                            cx,
                        )
                        .expect("project Gradle selection");
                });
            });
            cx.run_until_parked();
            let import = lsp_store
                .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                .expect("project selection overrides unavailable user selection");
            cx.update(|_, cx| {
                cx.update_global(|store: &mut settings::SettingsStore, cx| {
                    store
                        .set_local_settings(
                            worktree_id,
                            settings::LocalSettingsPath::InWorktree(RelPath::empty().into()),
                            settings::LocalSettingsKind::Settings,
                            None,
                            cx,
                        )
                        .expect("remove project Gradle selection");
                });
            });
            cx.run_until_parked();
            assert!(
                import.await.is_err(),
                "settings replacement rejects before dispatch"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            set_gradle_user(cx, gradle_root_settings);
            cx.run_until_parked();
        }
        if reactor {
            for exclusions in [
                vec![path(worktree_id, "missing")],
                vec![path(worktree_id, ".env")],
                vec![path(worktree_id, "pom.xml")],
                vec![path(worktree_id, "")],
                vec![path(worktree_id, "ignored"), path(worktree_id, "ignored")],
            ] {
                assert!(
                    lsp_store
                        .update(cx, |store, cx| store.import_local_maven_reactor(
                            worktree_id,
                            path(worktree_id, "pom.xml"),
                            exclusions,
                            cx
                        ))
                        .is_err()
                );
            }
        }
        let import = lsp_store
            .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
            .expect("pre-dispatch request");
        let mut prior_receipt = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_import_token(worktree_id)
                .expect("live pre-dispatch phase receipt")
        });
        let removed_trust = cx.update(|_, cx| cx.remove_global::<TrustedWorktrees>());
        cx.run_until_parked();
        assert!(
            import.await.is_err(),
            "permission must be checked again before dispatch"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        cx.update(|_, cx| cx.set_global(removed_trust));
        let attempts = if gradle && !unlink { 6 } else { 5 };
        for attempt in 0..attempts {
            // Explicit saved selection changes JDTLS initialization import policy.
            // Progress must come from the replacement that owns the current phase.
            while let Some(Some(replacement)) = servers.next().now_or_never() {
                server = replacement;
            }
            if unlink {
                if !lsp_store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }) {
                    assert!(
                        lsp_store
                            .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                            .is_err(),
                        "stale imported identities cannot dispatch another unlink"
                    );
                }
                assert!(
                    lsp_store
                        .update(cx, |store, cx| import_selected(
                            store,
                            vec![path(worktree_id, "module")],
                            cx
                        ))
                        .is_err(),
                    "unobserved module cannot be selected"
                );
                lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
            }
            let mut removed_trust = None;
            let (mut release, receiver) = oneshot::channel();
            *import_gate.lock().expect("gate") = Some(receiver);
            let mut import = lsp_store
                .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                .expect("explicit request");
            let receipt = lsp_store.read_with(cx, |store, _| {
                store
                    .jdtls_project_import_token(worktree_id)
                    .expect("live owned phase receipt")
            });
            assert_ne!(receipt, prior_receipt);
            assert!(
                !lsp_store
                    .update(cx, |store, cx| store
                        .cancel_selected_project_phase_for_token(prior_receipt, cx))
                    .expect("retired phase cannot cancel current work")
            );
            let mut unrelated = project::jvm_project::JdtlsProjectState::default();
            let unrelated_receipt = unrelated
                .begin_import_operation(receipt.server_id(), WorktreeId::from_usize(999))
                .expect("unrelated fixture receipt");
            assert!(
                !lsp_store
                    .update(cx, |store, cx| store
                        .cancel_selected_project_phase_for_token(unrelated_receipt, cx))
                    .expect("other-worktree phase cannot cancel current work")
            );
            assert_eq!(
                lsp_store.read_with(cx, |store, _| store.jdtls_project_import_token(worktree_id)),
                Some(receipt)
            );
            cx.run_until_parked();
            assert_eq!(calls.load(Ordering::SeqCst), attempt + 1);
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            cx.run_until_parked();
            assert_eq!(calls.load(Ordering::SeqCst), attempt + 1);
            assert!(
                (&mut import).now_or_never().is_none(),
                "explicit read refresh preserves the held mutation and its cancellation owner"
            );
            assert!(
                lsp_store
                    .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                    .is_err(),
                "busy import cannot be replaced"
            );
            lsp_store.read_with(cx, |store, _| {
                let current_servers = store
                    .language_server_statuses()
                    .filter(|(_, status)| {
                        status.name == lsp::LanguageServerName::new_static("jdtls")
                            && status.worktree == Some(worktree_id)
                    })
                    .map(|(server_id, _)| server_id)
                    .collect::<Vec<_>>();
                assert_eq!(
                    current_servers,
                    [server.server.server_id()],
                    "progress fixture must use the current project provider"
                );
            });
            server
                .request::<lsp::request::WorkDoneProgressCreate>(
                    lsp::WorkDoneProgressCreateParams {
                        token: lsp::NumberOrString::String("selected-import".into()),
                    },
                    std::time::Duration::from_secs(5),
                )
                .await
                .into_response()
                .expect("progress token");
            server.notify::<lsp::notification::Progress>(lsp::ProgressParams {
                token: lsp::NumberOrString::String("selected-import".into()),
                value: lsp::ProgressParamsValue::WorkDone(lsp::WorkDoneProgress::Begin(
                    lsp::WorkDoneProgressBegin {
                        title: if gradle {
                            "Importing Gradle projects"
                        } else {
                            "Importing Maven projects"
                        }
                        .into(),
                        cancellable: Some(true),
                        message: None,
                        percentage: None,
                    },
                )),
            });
            cx.run_until_parked();
            panel.update(cx, |panel, cx| panel.refresh(cx));
            server.notify::<ProjectEvent>(
                serde_json::json!({"eventType": if unlink { 210 } else { 200 }}),
            );
            server.notify::<lsp::notification::Progress>(lsp::ProgressParams {
                token: lsp::NumberOrString::String("selected-import".into()),
                value: lsp::ProgressParamsValue::WorkDone(lsp::WorkDoneProgress::End(
                    lsp::WorkDoneProgressEnd { message: None },
                )),
            });
            cx.run_until_parked();
            assert!(
                !lsp_store.read_with(cx, |store, _| store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)),
                "held phase cannot publish a model"
            );
            match attempt {
                1 => {
                    assert!(
                        lsp_store
                            .update(cx, |store, cx| store
                                .cancel_selected_project_phase_for_token(receipt, cx))
                            .expect("cancel")
                    );
                }
                2 => {
                    fs.insert_file(
                        std::path::Path::new(if gradle {
                            "/groovy-import/build.gradle"
                        } else {
                            "/groovy-import/pom.xml"
                        }),
                        if gradle {
                            b"plugins { id 'java' } // changed".to_vec()
                        } else {
                            b"<project>changed</project>".to_vec()
                        },
                    )
                    .await;
                    cx.run_until_parked();
                }
                3 => {
                    fail_import.store(true, Ordering::SeqCst);
                }
                4 => {
                    cx.update(|_, cx| {
                        removed_trust = Some(cx.remove_global::<TrustedWorktrees>());
                    });
                }
                5 => set_gradle_user(
                    cx,
                    r#"{"jvm_builds":{"gradle":{"selected_build_files":[]}}}"#,
                ),
                _ => {}
            }
            cx.run_until_parked();
            let result = if attempt == 1 || attempt == 2 {
                let result = import.await;
                release.cancellation().await;
                assert!(
                    release.is_canceled(),
                    "existing LSP cancellation drops the held provider handler"
                );
                result
            } else {
                release.send(()).expect("release current provider handler");
                import.await
            };
            if attempt == 0 {
                result.expect("progress did not cancel explicit phase");
                if metadata_outputs {
                    cx.executor()
                        .advance_clock(std::time::Duration::from_millis(300));
                    cx.run_until_parked();
                    assert!(lsp_store.read_with(cx, |store, _| {
                        store.jdtls_project_import_token(worktree_id).is_none()
                    }));
                    assert!(
                        lsp_store.read_with(cx, |store, _| store
                            .jdtls_project_snapshot(worktree_id)
                            .is_some_and(|(_, current)| current)),
                        "baseline data is current before external metadata change"
                    );
                    fs.insert_file(
                        std::path::Path::new("/groovy-import/.project"),
                        b"external metadata change".to_vec(),
                    )
                    .await;
                    cx.executor()
                        .advance_clock(std::time::Duration::from_millis(300));
                    cx.run_until_parked();
                    assert!(
                        !lsp_store.read_with(cx, |store, _| store
                            .jdtls_project_snapshot(worktree_id)
                            .is_some_and(|(_, current)| current)),
                        "external metadata outside import still invalidates project data"
                    );
                    lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
                    cx.executor()
                        .advance_clock(std::time::Duration::from_millis(300));
                    cx.run_until_parked();
                }
            } else {
                let error = result.expect_err("failed or changed binding").to_string();
                assert!(!error.contains("/private/provider/cache"));
                assert!(!error.contains("secret"));
                assert!(error.contains(if gradle { "Gradle" } else { "Maven" }));
                assert!(!error.contains("{name}"));
                assert!(error.contains("refresh") || error.contains("trust"));
            }
            assert_eq!(
                lsp_store.read_with(cx, |store, _| store.jdtls_project_import_token(worktree_id)),
                None
            );
            assert!(
                !lsp_store
                    .update(cx, |store, cx| store
                        .cancel_selected_project_phase_for_token(receipt, cx))
                    .expect("finished receipt cannot cancel anything")
            );
            prior_receipt = receipt;
            fail_import.store(false, Ordering::SeqCst);
            if attempt == 4 {
                cx.update(|_, cx| cx.set_global(removed_trust.expect("removed trust global")));
            }
            if attempt == 5 {
                set_gradle_user(cx, gradle_root_settings);
            }
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
        }
        if unlink {
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
        }
        lsp_store
            .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
            .expect("recovery request")
            .await
            .expect("recovery receipt");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        assert!(
            lsp_store.read_with(cx, |store, _| store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)),
            "separate read recovers current data"
        );
        if reactor {
            for reply_mode in 1..=3 {
                reactor_reply_mode.store(reply_mode, Ordering::SeqCst);
                let error = lsp_store
                    .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                    .expect("explicit reactor reply check")
                    .await
                    .expect_err("invisible or excluded module must reject phase success")
                    .to_string();
                assert!(error.contains("Maven") && error.contains("refresh"));
                assert!(
                    !error.contains(".env")
                        && !error.contains("groovy-import")
                        && !error.contains("ignored")
                );
                assert!(!lsp_store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }));
            }
            reactor_reply_mode.store(0, Ordering::SeqCst);
            lsp_store
                .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
                .expect("reactor recovery")
                .await
                .expect("visible module receipt");
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            assert!(lsp_store.read_with(cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .is_some_and(|(_, current)| current)
            }));
        }
        if panel_controls {
            let (release, receiver) = oneshot::channel();
            *import_gate.lock().expect("phase gate") = Some(receiver);
            exercise_saved_build_controls(
                &project,
                &lsp_store,
                worktree_id,
                false,
                &fail_import,
                release,
                cx,
            )
            .await;
            return;
        }
        let (mut release, receiver) = oneshot::channel();
        *import_gate.lock().expect("gate") = Some(receiver);
        let import = lsp_store
            .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
            .expect("server teardown request");
        let stopped_receipt = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_import_token(worktree_id)
                .expect("owned teardown receipt")
        });
        cx.run_until_parked();
        let stop = lsp_store.update(cx, |store, cx| {
            store.stop_language_servers_for_buffers(vec![buffer.clone()], HashSet::default(), cx)
        });
        cx.run_until_parked();
        assert!(import.await.is_err());
        release.cancellation().await;
        assert!(
            release.is_canceled(),
            "server stop cancels the held provider handler"
        );
        stop.await.expect("stop provider");
        assert!(!lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)
        }));
        lsp_store.update(cx, |store, cx| {
            store.restart_language_servers_for_buffers(vec![buffer], HashSet::default(), true, cx)
        });
        let _replacement = servers.next().await.expect("replacement provider");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        if metadata_outputs {
            for _ in 0..5 {
                if lsp_store.read_with(cx, |store, _| {
                    store.jdtls_project_import_token(worktree_id).is_none()
                }) {
                    break;
                }
                cx.executor()
                    .advance_clock(std::time::Duration::from_secs(1));
                cx.run_until_parked();
            }
            assert!(
                lsp_store.read_with(cx, |store, _| store
                    .jdtls_project_import_token(worktree_id)
                    .is_none()),
                "deliberately delayed startup finishes before explicit recovery request"
            );
        }
        let (mut release, receiver) = oneshot::channel();
        *import_gate.lock().expect("gate") = Some(receiver);
        let import = lsp_store
            .update(cx, |store, cx| import_selected(store, poms.clone(), cx))
            .expect("revocation request");
        let replacement_receipt = lsp_store.read_with(cx, |store, _| {
            store
                .jdtls_project_import_token(worktree_id)
                .expect("replacement phase receipt")
        });
        assert_ne!(replacement_receipt, stopped_receipt);
        assert!(
            !lsp_store
                .update(cx, |store, cx| store
                    .cancel_selected_project_phase_for_token(stopped_receipt, cx))
                .expect("old server cannot cancel replacement phase")
        );
        assert_eq!(
            lsp_store.read_with(cx, |store, _| store.jdtls_project_import_token(worktree_id)),
            Some(replacement_receipt)
        );
        cx.run_until_parked();
        trust.update(cx, |trusted, cx| {
            trusted.restrict(
                worktree_store.downgrade(),
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        cx.run_until_parked();
        assert!(import.await.is_err(), "revoked trust rejects held receipt");
        release.cancellation().await;
        assert!(
            release.is_canceled(),
            "trust revocation cancels the held provider handler"
        );
        assert!(
            lsp_store
                .update(cx, |store, cx| import_selected(store, poms, cx))
                .is_err(),
            "restricted retry denied"
        );
    }

    #[gpui::test]
    async fn local_panel_uses_current_jdtls_data_and_shared_tree_status(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/groovy-panel",
            serde_json::json!({
                "application": {
                    "pom.xml": "<project/>",
                    "Main.java": "class Main {}",
                    "processor-generated": {"Generated.java":"class Generated {}"},
                    "other-processor-generated": {"Generated.java":"class Generated {}"},
                    "configured-file": "not a directory",
                    ".env": {"present.java": "private fixture"}
                }
            }),
        )
        .await;
        fs.insert_tree(
            "/groovy-panel-secondary",
            serde_json::json!({
                "pom.xml": "<project/>",
                ".zed": {"settings.json": serde_json::json!({
                    "jvm_builds": {"maven": {"selected_poms": [], "ignored_projects": []}}
                }).to_string()}
            }),
        )
        .await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/groovy-panel".as_ref()], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = worktree_store.read_with(cx, |store, cx| {
            store
                .worktrees()
                .next()
                .expect("fixture worktree")
                .read(cx)
                .id()
        });
        let trusted_worktrees =
            cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust global"));
        trusted_worktrees.update(cx, |trusted, cx| {
            trusted.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            );
        });
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        let missing_root = lsp_store
            .update(cx, |store, cx| {
                store.synchronize_local_gradle_project(Some(worktree_id), cx)
            })
            .expect_err("missing Gradle root must deny");
        assert!(missing_root.to_string().contains("No Gradle build"));
        fs.insert_file(
            std::path::Path::new("/groovy-panel/settings.gradle.kts"),
            b"rootProject.name = \"preflight\"".to_vec(),
        )
        .await;
        cx.run_until_parked();
        let missing_provider = lsp_store
            .update(cx, |store, cx| {
                store.synchronize_local_gradle_project(Some(worktree_id), cx)
            })
            .expect_err("missing Java provider must deny");
        assert!(
            missing_provider
                .to_string()
                .contains("Java language server is unavailable")
        );
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let fail_queries = Arc::new(AtomicBool::new(false));
        let project_queries = Arc::new(AtomicUsize::new(0));
        let configuration_mode = Arc::new(AtomicUsize::new(2));
        let compiler_output_mode = Arc::new(AtomicUsize::new(0));
        let configuring_queries = Arc::new(AtomicUsize::new(0));
        let configuring_gate = Arc::new(std::sync::Mutex::new(
            None::<(oneshot::Sender<()>, oneshot::Receiver<()>)>,
        ));
        let (release_first_query, first_query_reply) = oneshot::channel();
        let first_query_reply = Arc::new(std::sync::Mutex::new(Some(first_query_reply)));
        let mut fake_servers = languages.register_fake_lsp(
            "Java",
            FakeLspAdapter {
                name: "jdtls",
                capabilities: lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![
                            "java.project.getAll".to_string(),
                            "java.project.getSettings".to_string(),
                            "groovy.gradle.projects".to_string(),
                            "groovy.gradle.projectDetails".to_string(),
                            "groovy.maven.projects".to_string(),
                            "groovy.maven.projectDetails".to_string(),
                            "groovy.maven.goalDetails".to_string(),
                        ],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                initializer: Some(Box::new({
                    let fail_queries = fail_queries.clone();
                    let project_queries = project_queries.clone();
                    let configuration_mode = configuration_mode.clone();
                                let compiler_output_mode = compiler_output_mode.clone();
                    let configuring_queries = configuring_queries.clone();
                    let configuring_gate = configuring_gate.clone();
                    let first_query_reply = first_query_reply.clone();
                    move |fake_server| {
                        fake_server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                            let fail_queries = fail_queries.clone();
                            let project_queries = project_queries.clone();
                            let configuration_mode = configuration_mode.clone();
                                let compiler_output_mode = compiler_output_mode.clone();
                            let configuring_queries = configuring_queries.clone();
                            let configuring_gate = configuring_gate.clone();
                            let first_query_reply = first_query_reply.clone();
                            move |params, _| {
                                let fail_queries = fail_queries.clone();
                                let project_queries = project_queries.clone();
                                let configuration_mode = configuration_mode.clone();
                                let compiler_output_mode = compiler_output_mode.clone();
                                let configuring_queries = configuring_queries.clone();
                                let configuring_gate = configuring_gate.clone();
                                let first_query_reply = first_query_reply.clone();
                                async move {
                                    if params.command == "java.project.getAll" {
                                        project_queries.fetch_add(1, Ordering::SeqCst);
                                        let reply = first_query_reply
                                            .lock()
                                            .expect("first query lock")
                                            .take();
                                        if let Some(reply) = reply {
                                            reply.await.expect("release initial project query");
                                        }
                                    }
                                    if fail_queries.load(Ordering::SeqCst) {
                                        anyhow::bail!(
                                            "provider query failed at /private/host/path"
                                        );
                                    }
                                    match params.command.as_str() {
                                        "java.project.getAll" => Ok(Some(serde_json::json!([
                                            "file:///groovy-panel/application"
                                        ]))),
                                        "java.project.getSettings" => Ok(Some(serde_json::json!({
                                            "org.eclipse.jdt.ls.core.sourcePaths": [],
                                            "org.eclipse.jdt.ls.core.classpathEntries": [
                                                {"kind": 2, "path": "/library"},
                                                {"kind": 1, "path": "/private/cache/junit.jar"}
                                            ]
                                        }))),
                                        "groovy.gradle.projectDetails" => {
                                            configuring_queries.fetch_add(1, Ordering::SeqCst);
                                            let gate = configuring_gate.lock().expect("configuring gate lock").take();
                                            if let Some((started, release)) = gate {
                                                started.send(()).map_err(|_| anyhow::anyhow!("configuring start receiver closed"))?;
                                                release.await?;
                                            }
                                            let configuration_mode = configuration_mode.load(Ordering::SeqCst);
                                            let compiler_output_mode = compiler_output_mode.load(Ordering::SeqCst);
                                            let configuration_declarations = if configuration_mode == 2 {
                                                serde_json::json!([
                                                    {"name":"implementation","description":"Dependency declarations","resolvable":false,"consumable":false,"extendsFrom":[]},
                                                    {"name":"runtimeClasspath","description":"Runtime declarations","resolvable":true,"consumable":false,"extendsFrom":["implementation"]}
                                                ])
                                            } else { serde_json::json!([]) };
                                            assert_eq!(params.arguments[0]["offline"], true);
                                            Ok(Some(serde_json::json!({
                                                "version": 2, "rootUri": "file:///groovy-panel/", "sourceSetsKnown": false,
                                                "projects": [{
                                                    "buildRootUri": "file:///groovy-panel/",
                                                    "moduleUri": "file:///groovy-panel/application/",
                                                    "projectPath": ":application", "name": "application",
                                                    "tasks": [{"path": ":application:test", "name": "test",
                                                        "group": "verification", "description": "Run tests", "optionsKnown": false}],
                                                    "detailsJson": serde_json::json!({
                                                        "javaSourceSetsKnown": true,
                                                        "configurationsKnown": configuration_mode != 0,
                                                        "configurations": configuration_declarations,
                                                        "dependencyGraphKnown": false,
                                                        "sourceSets": [{"name": "main", "roots": ["file:///groovy-panel/application/src/main/java", "file:///groovy-panel/application/.env/missing-source"], "generatednessKnown": false,
                                                            "annotationProcessorOutputKnown": compiler_output_mode != 0,
                                                            "annotationProcessorOutputUri": (match compiler_output_mode {
                                                                2 => Some("file:///groovy-panel/application/processor-generated"),
                                                                3 => Some("file:///groovy-panel/application/other-processor-generated"),
                                                                4 => Some("file:///groovy-panel/application/missing-processor-output"),
                                                                5 => Some("file:///groovy-panel/application/.env/present.java"),
                                                                6 => Some("file:///groovy-panel/application/configured-file"),
                                                                _ => None,
                                                            })
                                                        }],
                                                        "tasks": [{"path": ":application:test", "optionsComplete": true,
                                                            "options": [
                                                                {"name": "tests", "description": "Selected test filter", "declaredType": "java.util.List<java.lang.String>"},
                                                                {"name": "mode", "description": "Declared mode", "declaredType": "fixture.Mode", "enumConstantsKnown": true, "enumConstants": ["ALPHA", "ZULU"]},
                                                                {"name": "emptyMode", "description": "Empty declaration", "declaredType": "fixture.Empty", "enumConstantsKnown": true, "enumConstants": []}]}]
                                                    }).to_string()
                                                }]
                                            })))
                                        },
                                        "groovy.maven.goalDetails" => {
                                            let parameter_declarations = serde_json::json!([{"name": "skipTests", "declaredType": "boolean", "required": false, "editable": true}]);
                                            assert_eq!(params.arguments[0]["cachedOnly"], true);
                                            Ok(Some(serde_json::json!({
                                                "version": 3, "rootUri": "file:///groovy-panel/", "cachedOnly": true,
                                                "reactorOrderKnown": true, "reactorPomUris": ["file:///groovy-panel/application/pom.xml"], "unavailablePomUris": [],
                                                "lifecycles": [{"id": "default", "phases": ["test", "pre-integration-test", "verify"]}],
                                                "projects": [{"pomUri": "file:///groovy-panel/application/pom.xml",
                                                    "groupId": "evidence", "artifactId": "application", "version": "1.0",
                                                    "packaging": "jar", "moduleUris": ["file:///groovy-panel/application/.env/missing-module"],
                                                    "profiles": {"parity": false}, "inheritedProfileDefinitionsKnown": true,
                                                    "profileDefinitions": [
                                                        {"id": "parity", "sourceGroupId": "evidence", "sourceArtifactId": "application", "sourceVersion": "1.0",
                                                            "sourcePomUri": "file:///groovy-panel/application/pom.xml", "inherited": false, "activeAtSource": false},
                                                        {"id": "private", "sourceGroupId": "evidence", "sourceArtifactId": "parent", "sourceVersion": "1.0",
                                                            "sourcePomUri": "file:///groovy-panel/application/.env/missing-parent/pom.xml", "inherited": true, "activeAtSource": false}
                                                    ],
                                                    "plugins": [{"groupId": "evidence", "artifactId": "uncached-plugin", "version": null,
                                                        "managedOnly": true, "goalsKnown": false, "goals": []},
                                                        {"groupId": "org.apache.maven.plugins", "artifactId": "maven-surefire-plugin", "version": "3.5.2",
                                                            "managedOnly": false, "goalsKnown": true, "goalPrefix": "surefire",
                                                            "goals": [{"name": "test", "phase": "test", "description": "Run selected tests",
                                                                "parametersKnown": true, "parameters": parameter_declarations}]}]}]
                                            })))
                                        },
                                        _ => anyhow::bail!("unexpected project command"),
                                    }
                                }
                            }
                        });
                    }
                })),
                ..FakeLspAdapter::default()
            },
        );
        let _java_buffer = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/groovy-panel/application/Main.java", cx)
            })
            .await
            .expect("open Java source");
        let _fake_server = fake_servers.next().await.expect("JDTLS provider");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();

        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Groovy Dependencies panel");
        panel.update(cx, |panel, cx| {
            panel.active = true;
            panel.rebuild(cx);
        });
        panel.read_with(cx, |panel, _| {
            assert!(panel.host.visible_rows().is_empty());
            assert_eq!(
                panel.host.status(),
                &language_tools::language_tool_tree::LanguageToolTreeStatus::Loading
            );
        });
        release_first_query
            .send(())
            .expect("pending project query remains active");
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel.host.status(),
                &language_tools::language_tool_tree::LanguageToolTreeStatus::Current
            );
            let rows = panel.host.visible_rows();
            assert_eq!(rows.len(), 1);
            assert!(rows[0].accessibility_label.contains("JDTLS module"));
            let navigation = panel
                .navigation
                .get(&rows[0].id)
                .expect("manifest navigation");
            assert_eq!(navigation.path.path.as_unix_str(), "application/pom.xml");
        });
        fs.insert_file(
            std::path::Path::new("/groovy-panel/settings.gradle.kts"),
            b"include(\"application\")".to_vec(),
        )
        .await;
        fs.insert_file(
            std::path::Path::new("/groovy-panel/pom.xml"),
            b"<project/>".to_vec(),
        )
        .await;
        fs.insert_file(
            std::path::Path::new("/groovy-panel/application/build.gradle.kts"),
            b"plugins { java }".to_vec(),
        )
        .await;
        cx.run_until_parked();
        panel.update(cx, |panel, cx| panel.refresh(cx));
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        project.read_with(cx, |project, cx| {
            project.lsp_store().read_with(cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("current composed provider snapshot");
                assert!(current);
                let maven = snapshot.projects[0]
                    .maven
                    .as_ref()
                    .expect("Maven facts through current LSP owner");
                assert_eq!(maven.artifact_id, "application");
                assert_eq!(maven.reactor_position, Some(0));
                assert_eq!(
                    maven.profile_definitions.len(),
                    1,
                    "private profile origin omitted"
                );
                assert!(
                    !maven.profile_definitions_known,
                    "hidden origin downgrades completeness"
                );
                assert!(
                    maven.modules.is_empty(),
                    "private missing module link omitted"
                );
                assert!(!maven.plugins[0].goals_known);
                assert!(maven.plugins[1].goals[0].parameters_known);
                assert_eq!(maven.plugins[1].goals[0].parameters[0].name, "skipTests");
                assert!(
                    snapshot.projects[0].gradle.is_none(),
                    "ordinary reads must leave configuring metadata unavailable"
                );
            });
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace.right_dock().update(cx, |dock, cx| {
                dock.set_open(true, window, cx);
            });
            workspace.focus_panel::<DependenciesPanel>(window, cx);
        });
        cx.run_until_parked();
        cx.update(|window, _| window.activate_a11y_for_test());
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            0,
            "startup and ordinary panel reads must not configure Gradle"
        );
        panel.update(cx, |panel, cx| panel.refresh(cx));
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            0,
            "manual Refresh is also a read"
        );
        panel.update_in(cx, |panel, window, cx| {
            panel.set_view(BuildToolView::Gradle, window, cx)
        });
        panel.read_with(cx, |panel, _| {
            assert!(panel.host.visible_rows().is_empty());
            let language_tools::language_tool_tree::LanguageToolTreeStatus::Empty(message) =
                panel.host.status()
            else {
                panic!("ordinary Gradle read must show unavailable metadata");
            };
            assert!(message.contains("Synchronize metadata"));
        });
        let (preflight_release, preflight_receiver) = oneshot::channel();
        *first_query_reply.lock().expect("preflight query lock") = Some(preflight_receiver);
        let reads_before_held = project_queries.load(Ordering::SeqCst);
        panel.update(cx, |panel, cx| panel.synchronize_gradle_metadata(cx));
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        assert_eq!(
            project_queries.load(Ordering::SeqCst),
            reads_before_held + 1
        );
        let missing_trust_owner = cx.update(|_, cx| cx.remove_global::<TrustedWorktrees>());
        preflight_release
            .send(())
            .expect("release reads after trust owner removal");
        cx.run_until_parked();
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            0,
            "permissions must be rechecked after snapshot reads, before configuration"
        );
        assert!(lsp_store.read_with(cx, |store, _| {
            store.jdtls_project_snapshot(worktree_id).is_none()
        }));
        cx.update(|_, cx| cx.set_global(missing_trust_owner));
        panel.update(cx, |panel, cx| panel.synchronize_gradle_metadata(cx));
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            1,
            "only the explicit operation configures Gradle"
        );
        for position in [
            DockPosition::Bottom,
            DockPosition::Left,
            DockPosition::Right,
        ] {
            panel.update_in(cx, |panel, window, cx| {
                assert!(panel.position_is_valid(position));
                panel.set_position(position, window, cx);
            });
            cx.run_until_parked();
            workspace.update_in(cx, |workspace, window, cx| {
                let dock = match position {
                    DockPosition::Left => workspace.left_dock(),
                    DockPosition::Bottom => workspace.bottom_dock(),
                    DockPosition::Right => workspace.right_dock(),
                }
                .read(cx);
                assert!(dock.is_open());
                assert!(dock.panel_for_id(panel.entity_id()).is_some());
                assert!(panel.read(cx).focus_handle.is_focused(window));
            });
        }
        for view in [BuildToolView::Gradle, BuildToolView::Maven] {
            panel.update_in(cx, |panel, window, cx| {
                panel.set_view(view, window, cx);
                panel.host.expand_all();
                assert!(
                    panel.focus_handle.is_focused(window),
                    "view switch retains tree keyboard focus"
                );
            });
            panel.read_with(cx, |panel, _| {
                let labels = panel
                    .host
                    .visible_rows()
                    .iter()
                    .map(|row| row.label.as_str())
                    .collect::<Vec<_>>();
                assert_eq!(panel.view, view);
                let root = panel.host.visible_rows().first().expect("provider module");
                let navigation = panel
                    .navigation
                    .get(&root.id)
                    .expect("tool-specific manifest navigation");
                match view {
                    BuildToolView::Gradle => {
                        assert!(labels.contains(&"Gradle tasks"));
                        assert!(labels.contains(&"Configurations"));
                        assert!(labels.contains(&"runtimeClasspath"));
                        assert_eq!(
                            labels
                                .iter()
                                .filter(|label| **label == "implementation")
                                .count(),
                            2
                        );
                        assert!(!labels.contains(&"Lifecycle"));
                        assert_eq!(
                            navigation.path.path.as_unix_str(),
                            "application/build.gradle.kts"
                        );
                    }
                    BuildToolView::Maven => {
                        for label in [
                            "Reactor",
                            "Lifecycle",
                            "Plugins",
                            "Profiles",
                            "test",
                            "skipTests",
                            "parity",
                            "Repositories",
                            "Saved configurations",
                        ] {
                            assert!(labels.contains(&label), "Maven row {label}");
                        }
                        assert!(!labels.contains(&"Gradle tasks"));
                        assert!(!labels.contains(&"private"));
                        assert_eq!(navigation.path.path.as_unix_str(), "application/pom.xml");
                        assert!(
                            panel
                                .host
                                .visible_rows()
                                .iter()
                                .filter(|row| row.label == "test")
                                .all(|row| !panel.navigation.contains_key(&row.id)),
                            "goals are not manifest navigation"
                        );
                    }
                    BuildToolView::Dependencies => unreachable!("build view fixture"),
                }
            });
            if view == BuildToolView::Gradle {
                let configuration_id = panel.read_with(cx, |panel, _| {
                    let row = panel
                        .host
                        .visible_rows()
                        .iter()
                        .find(|row| row.label == "runtimeClasspath")
                        .expect("current configuration");
                    assert!(
                        row.secondary_label
                            .as_deref()
                            .is_some_and(|label| label.contains("Resolvable; not consumable"))
                    );
                    assert!(row.accessibility_label.contains("Runtime declarations"));
                    assert!(
                        row.accessibility_label
                            .contains("resolved dependencies unavailable")
                    );
                    assert!(!panel.navigation.contains_key(&row.id));
                    row.id.clone()
                });
                panel.update_in(cx, |panel, window, cx| {
                    let count = workspace.read(cx).active_pane().read(cx).items().count();
                    panel.host.select(configuration_id.clone());
                    panel.host.reveal_selection(ScrollStrategy::Center);
                    panel.activate_selected(window, cx);
                    cx.notify();
                    assert_eq!(
                        workspace.read(cx).active_pane().read(cx).items().count(),
                        count
                    );
                });
                let mut previous_configuration_node = None;
                for _ in 0..2 {
                    cx.update(|window, cx| {
                        window.draw(cx).clear(cx);
                        let tree: serde_json::Value = serde_json::from_str(
                            &window.debug_a11y_tree_json().expect("configuration tree"),
                        )
                        .expect("valid configuration accessibility tree");
                        let nodes = tree["nodes"].as_object().expect("configuration nodes");
                        let (identity, node) = nodes
                            .iter()
                            .find(|(_, node)| {
                                node["aria"]["role"] == "TreeItem"
                                    && node["aria"]["label"].as_str().is_some_and(|label| {
                                        label.starts_with("Gradle configuration runtimeClasspath;")
                                    })
                            })
                            .expect("rendered configuration row");
                        assert_eq!(node["aria"]["selected"], true);
                        assert_eq!(
                            tree["active_descendant_focus"].as_str(),
                            Some(identity.as_str())
                        );
                        assert!(nodes.values().any(|node| node["aria"]["role"] == "Button" && node["aria"]["label"] == "Synchronize local Gradle metadata; executes trusted build configuration scripts"));
                        let focus = tree["gpui_focus"].as_str().expect("panel keyboard focus");
                        assert_eq!(nodes[focus]["aria"]["label"], "Gradle panel");
                        if let Some(previous) = &previous_configuration_node {
                            assert_eq!(&node["accesskit_id"], previous);
                        }
                        previous_configuration_node = Some(node["accesskit_id"].clone());
                    });
                }
                cx.simulate_keystrokes("down");
                panel.read_with(cx, |panel, _| {
                    let selected = panel.host.selected().expect("keyboard-selected parent");
                    let row = panel
                        .host
                        .visible_rows()
                        .iter()
                        .find(|row| &row.id == selected)
                        .expect("parent row");
                    assert!(
                        row.accessibility_label
                            .contains("extends declared parent implementation")
                    );
                    assert!(!panel.navigation.contains_key(selected));
                });
                panel.update_in(cx, |panel, window, cx| {
                    let count = workspace.read(cx).active_pane().read(cx).items().count();
                    panel.activate_selected(window, cx);
                    assert_eq!(
                        workspace.read(cx).active_pane().read(cx).items().count(),
                        count
                    );
                    assert!(panel.focus_handle.is_focused(window));
                });
                panel.update(cx, |panel, cx| {
                    panel.host.select(configuration_id.clone());
                    cx.notify();
                });
                for (iteration, mode) in [2, 0, 1, 2].into_iter().enumerate() {
                    configuration_mode.store(mode, Ordering::SeqCst);
                    let reads_before_refresh = project_queries.load(Ordering::SeqCst);
                    panel.update(cx, |panel, cx| {
                        if iteration == 0 {
                            panel.refresh(cx);
                        } else {
                            panel.synchronize_gradle_metadata(cx);
                        }
                    });
                    cx.executor()
                        .advance_clock(std::time::Duration::from_millis(300));
                    cx.run_until_parked();
                    panel.update(cx, |panel, cx| {
                        panel.host.expand_all();
                        cx.notify();
                    });
                    if iteration == 0 {
                        assert_eq!(
                            project_queries.load(Ordering::SeqCst),
                            reads_before_refresh + 2,
                            "explicit current refresh enumerates Java and non-Java resources without reconfiguring Gradle"
                        );
                        assert_eq!(
                            configuring_queries.load(Ordering::SeqCst),
                            1,
                            "current declaration reads retain their provenance"
                        );
                    }
                    panel.read_with(cx, |panel, _| {
                        let row = panel
                            .host
                            .visible_rows()
                            .iter()
                            .find(|row| row.label == "Configurations")
                            .expect("configuration coverage");
                        match mode {
                            0 => assert!(
                                row.accessibility_label.contains("declarations unavailable")
                            ),
                            1 => assert!(
                                row.accessibility_label
                                    .contains("No declared configurations")
                            ),
                            2 => assert!(row.accessibility_label.contains("2 declarations")),
                            _ => unreachable!("fixture coverage mode"),
                        }
                        let configuration = panel
                            .host
                            .visible_rows()
                            .iter()
                            .find(|row| row.label == "runtimeClasspath");
                        assert_eq!(configuration.is_some(), mode == 2);
                        if let Some(configuration) = configuration {
                            assert_eq!(configuration.id, configuration_id);
                        }
                        if iteration == 0 {
                            assert_eq!(panel.host.selected(), Some(&configuration_id));
                        }
                    });
                }
            }
        }
        panel.update_in(cx, |panel, window, cx| {
            let item_count = workspace.read(cx).active_pane().read(cx).items().count();
            let goals = panel
                .host
                .visible_rows()
                .iter()
                .filter(|row| row.label == "test")
                .map(|row| row.id.clone())
                .collect::<Vec<_>>();
            assert_eq!(
                goals.len(),
                2,
                "phase and plugin goal retain distinct identities"
            );
            for id in goals {
                panel.host.select(id);
                panel.activate_selected(window, cx);
            }
            let parameter = panel
                .host
                .visible_rows()
                .iter()
                .find(|row| row.label == "skipTests")
                .expect("cached parameter");
            assert!(!panel.navigation.contains_key(&parameter.id));
            assert!(
                parameter
                    .accessibility_label
                    .contains("CLI property mapping unknown")
            );
            let parameter_id = parameter.id.clone();
            panel.host.select(parameter_id.clone());
            assert_eq!(panel.host.selected(), Some(&parameter_id));
            panel.activate_selected(window, cx);
            assert_eq!(
                workspace.read(cx).active_pane().read(cx).items().count(),
                item_count,
                "read-only phase/goal activation does not navigate"
            );
            for row in panel
                .host
                .visible_rows()
                .iter()
                .filter(|row| row.label == "test")
            {
                assert!(row.accessibility_label.contains("execution unavailable"));
            }
        });
        panel.update_in(cx, |panel, window, cx| {
            panel.set_view(BuildToolView::Dependencies, window, cx)
        });
        let (secondary, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree("/groovy-panel-secondary", true, cx)
            })
            .await
            .expect("second Maven worktree without provider data");
        let secondary_id = secondary.read_with(cx, |worktree, _| worktree.id());
        assert_ne!(secondary_id, worktree_id);
        trusted_worktrees.update(cx, |trusted, cx| {
            trusted.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(secondary_id)]),
                cx,
            );
        });
        for _ in 0..40 {
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.background_executor
                .timer(std::time::Duration::from_millis(50))
                .await;
            cx.run_until_parked();
            if panel.read_with(cx, |panel, _| {
                matches!(
                    panel.host.status(),
                    language_tools::language_tool_tree::LanguageToolTreeStatus::Partial(_)
                ) && panel.host.visible_rows().len() == 1
                    && panel.navigation.len() == 1
            }) {
                break;
            }
        }
        panel.read_with(cx, |panel, _| {
            let language_tools::language_tool_tree::LanguageToolTreeStatus::Partial(message) =
                panel.host.status()
            else {
                panic!(
                    "the missing second provider snapshot must make coverage partial: {:?}",
                    panel.host.status()
                );
            };
            assert!(message.contains("Some project folders"));
            assert_eq!(panel.host.visible_rows().len(), 1);
            assert_eq!(panel.navigation.len(), 1);
        });
        panel.update(cx, |panel, cx| {
            panel.host.expand_all();
            cx.notify();
        });
        panel.read_with(cx, |panel, _| {
            let library = panel
                .host
                .visible_rows()
                .iter()
                .find(|row| row.label == "junit.jar")
                .expect("provider library file in shared tree");
            assert!(library.accessibility_label.contains("scope unknown"));
            assert!(!panel.navigation.contains_key(&library.id));
            let task = panel
                .host
                .visible_rows()
                .iter()
                .find(|row| row.label == ":application:test")
                .expect("task discovered through existing JDTLS snapshot");
            assert!(task.accessibility_label.contains("group verification"));
            assert!(task.accessibility_label.contains("execution unavailable"));
            assert!(!panel.navigation.contains_key(&task.id));
            let option = panel
                .host
                .visible_rows()
                .iter()
                .find(|row| row.label == "--tests")
                .expect("declared option in shared tree");
            assert!(
                option
                    .accessibility_label
                    .contains("accepted values and requiredness unknown")
            );
            assert!(!panel.navigation.contains_key(&option.id));
            let source = panel
                .host
                .visible_rows()
                .iter()
                .find(|row| row.label == "application/src/main/java")
                .expect("declared root in shared tree");
            assert!(
                source
                    .accessibility_label
                    .contains("not present in project snapshot")
            );
            assert!(!panel.navigation.contains_key(&source.id));
            assert!(
                panel
                    .host
                    .visible_rows()
                    .iter()
                    .all(|row| !row.label.contains(".env"))
            );
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.focus_panel::<DependenciesPanel>(window, cx);
        });
        cx.run_until_parked();
        panel.update_in(cx, |panel, window, cx| {
            assert!(panel.focus_handle(cx).contains_focused(window, cx));
        });
        cx.update(|window, _cx| window.activate_a11y_for_test());
        let enum_reads = project_queries.load(Ordering::SeqCst);
        cx.dispatch_action(ShowGradle);
        panel.update(cx, |panel, cx| {
            panel.host.expand_all();
            cx.notify();
        });
        let (option_id, member_id) = panel.read_with(cx, |panel, _| {
            let rows = panel.host.visible_rows();
            let unknown = rows
                .iter()
                .find(|row| row.label == "--tests")
                .expect("unknown declaration");
            assert!(
                unknown
                    .accessibility_label
                    .contains("enum declarations unknown")
            );
            assert!(!unknown.is_branch);
            let empty = rows
                .iter()
                .find(|row| row.label == "--emptyMode")
                .expect("known-empty declaration");
            assert!(
                empty
                    .accessibility_label
                    .contains("no declared enum members")
            );
            assert!(!empty.is_branch);
            let option = rows
                .iter()
                .find(|row| row.label == "--mode")
                .expect("declared mode");
            assert!(
                option
                    .accessibility_label
                    .contains("2 declared enum members")
            );
            assert!(
                option
                    .accessibility_label
                    .contains("accepted values and requiredness unknown")
            );
            let member = rows
                .iter()
                .find(|row| row.label == "ALPHA")
                .expect("declared member");
            assert_eq!(member.parent_id.as_ref(), Some(&option.id));
            assert!(
                member
                    .accessibility_label
                    .contains("custom setter acceptance unknown")
            );
            assert!(!panel.navigation.contains_key(&member.id));
            (option.id.clone(), member.id.clone())
        });
        panel.update(cx, |panel, cx| {
            panel.host.select(option_id.clone());
            panel.host.reveal_selection(ScrollStrategy::Center);
            cx.notify();
        });
        cx.simulate_keystrokes("right");
        panel.read_with(cx, |panel, _| {
            assert_eq!(panel.host.selected(), Some(&member_id))
        });
        cx.simulate_keystrokes("left");
        panel.read_with(cx, |panel, _| {
            assert_eq!(panel.host.selected(), Some(&option_id))
        });
        cx.simulate_keystrokes("right");
        panel.update(cx, |panel, cx| {
            panel.host.reveal_selection(ScrollStrategy::Center);
            cx.notify();
        });
        let enum_items = workspace.read_with(cx, |workspace, cx| {
            workspace.active_pane().read(cx).items().count()
        });
        cx.simulate_keystrokes("enter");
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| workspace
                .active_pane()
                .read(cx)
                .items()
                .count()),
            enum_items
        );
        assert_eq!(
            project_queries.load(Ordering::SeqCst),
            enum_reads,
            "enum projection and keyboard do not query the provider"
        );
        let mut member_accesskit_id = None;
        for refresh in [false, true] {
            if refresh {
                panel.update(cx, |panel, cx| panel.refresh(cx));
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
                assert_eq!(
                    project_queries.load(Ordering::SeqCst),
                    enum_reads + 2,
                    "only explicit refresh reads current resources"
                );
            }
            panel.update(cx, |panel, cx| {
                panel.host.expand_all();
                panel.host.reveal_selection(ScrollStrategy::Center);
                cx.notify();
            });
            panel.read_with(cx, |panel, _| {
                assert_eq!(panel.host.selected(), Some(&member_id))
            });
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value =
                        serde_json::from_str(&window.debug_a11y_tree_json().expect("enum tree"))
                            .expect("enum accessibility JSON");
                    let nodes = tree["nodes"].as_object().expect("enum nodes");
                    let (identity, node) = nodes
                        .iter()
                        .find(|(_, node)| {
                            node["aria"]["role"] == "TreeItem"
                                && node["aria"]["label"].as_str().is_some_and(|label| {
                                    label.starts_with("Declared enum member ALPHA for option mode;")
                                })
                        })
                        .expect("rendered selected enum member");
                    assert_eq!(node["aria"]["selected"], true);
                    assert_eq!(
                        tree["active_descendant_focus"].as_str(),
                        Some(identity.as_str())
                    );
                    let focus = tree["gpui_focus"].as_str().expect("enum keyboard focus");
                    assert_eq!(nodes[focus]["aria"]["label"], "Gradle panel");
                    if let Some(previous) = &member_accesskit_id {
                        assert_eq!(&node["accesskit_id"], previous);
                    }
                    member_accesskit_id = Some(node["accesskit_id"].clone());
                });
            }
        }
        panel.update_in(cx, |panel, window, cx| {
            panel.set_view(BuildToolView::Dependencies, window, cx)
        });
        for _ in 0..2 {
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let tree: serde_json::Value = serde_json::from_str(
                    &window.debug_a11y_tree_json().expect("active panel tree"),
                )
                .expect("valid accessibility tree");
                let nodes = tree["nodes"].as_object().expect("panel nodes");
                for label in [
                    "Expand All",
                    "Collapse All",
                    "Refresh Dependencies",
                    "Dependencies view",
                    "Gradle view",
                    "Maven view",
                ] {
                    assert!(nodes.values().any(|node| {
                        node["aria"]["label"] == label && node["aria"]["role"] == "Button"
                    }));
                }
            });
        }
        for view in [BuildToolView::Gradle, BuildToolView::Maven] {
            match view {
                BuildToolView::Gradle => cx.dispatch_action(ShowGradle),
                BuildToolView::Maven => cx.dispatch_action(ShowMaven),
                BuildToolView::Dependencies => unreachable!("build view action"),
            }
            panel.read_with(cx, |panel, _| assert_eq!(panel.view, view));
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree: serde_json::Value = serde_json::from_str(
                        &window.debug_a11y_tree_json().expect("build view tree"),
                    )
                    .expect("valid tree");
                    let nodes = tree["nodes"].as_object().expect("build view nodes");
                    for candidate in [
                        BuildToolView::Dependencies,
                        BuildToolView::Gradle,
                        BuildToolView::Maven,
                    ] {
                        let label = format!("{} view", candidate.label());
                        assert!(
                            nodes.values().any(|node| node["aria"]["label"] == label
                                && node["aria"]["role"] == "Button"
                                && node["aria"]["toggled"]
                                    == if candidate == view { "True" } else { "False" }),
                            "named selected view {label}"
                        );
                    }
                });
            }
        }
        let queries_before_options = project_queries.load(Ordering::SeqCst);
        let navigation_before_options = panel.read_with(cx, |panel, _| {
            panel.navigation.keys().cloned().collect::<HashSet<_>>()
        });
        panel.read_with(cx, |panel, _| {
            assert!(panel.preferences.show_basic_maven_phases);
            assert!(
                !panel
                    .host
                    .visible_rows()
                    .iter()
                    .any(|row| row.label == "pre-integration-test")
            );
        });
        panel.update(cx, |panel, cx| {
            panel.show_options = true;
            panel.toggle_option(BuildViewOption::ShowBasicMavenPhases, cx);
            panel.toggle_option(BuildViewOption::GroupModules, cx);
            panel.host.expand_all();
            panel.toggle_option(BuildViewOption::ShowToolbar, cx);
            panel.toggle_option(BuildViewOption::ShowMavenVersion, cx);
            panel.toggle_option(BuildViewOption::ShowMavenArtifactId, cx);
            panel.cycle_graph_fact_filter(cx);
            panel.cycle_maven_graph_scope(cx);
        });
        for _ in 0..2 {
            cx.update(|window, cx| {
                window.draw(cx).clear(cx);
                let tree: serde_json::Value =
                    serde_json::from_str(&window.debug_a11y_tree_json().expect("options tree"))
                        .expect("valid options JSON");
                let nodes = tree["nodes"].as_object().expect("options nodes");
                for label in [
                    "Maven view options",
                    "Group Modules",
                    "Show Basic Phases Only",
                    "Always Show ArtifactId",
                    "Show Version",
                    "Show Toolbar",
                    "Maven view",
                    "Gradle view",
                ] {
                    assert!(
                        nodes.values().any(|node| node["aria"]["label"] == label
                            && node["aria"]["role"] == "Button"),
                        "named option {label}"
                    );
                }
                assert!(
                    !nodes
                        .values()
                        .any(|node| node["aria"]["label"] == "Refresh Dependencies")
                );
                assert!(
                    nodes
                        .values()
                        .any(|node| node["aria"]["label"] == "Show Toolbar"
                            && node["aria"]["toggled"] == "False")
                );
            });
        }
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel.navigation.keys().cloned().collect::<HashSet<_>>(),
                navigation_before_options
            );
            assert!(!panel.preferences.show_maven_version);
            assert!(!panel.preferences.show_maven_artifact_id);
            assert!(panel.preferences.group_maven_modules);
            assert!(!panel.preferences.show_basic_maven_phases);
            assert!(
                panel
                    .host
                    .visible_rows()
                    .iter()
                    .any(|row| row.label == "pre-integration-test")
            );
            assert!(
                panel
                    .host
                    .visible_rows()
                    .iter()
                    .any(|row| row.label == "Directory: Project root")
            );
        });
        assert_eq!(
            project_queries.load(Ordering::SeqCst),
            queries_before_options,
            "presentation changes must not query or reimport"
        );
        panel.update_in(cx, |panel, window, cx| {
            for position in [
                DockPosition::Left,
                DockPosition::Right,
                DockPosition::Bottom,
            ] {
                panel.set_position(position, window, cx);
            }
            let generation = panel.preferences_generation;
            panel.set_position(DockPosition::Bottom, window, cx);
            assert_eq!(panel.preferences_generation, generation);
        });
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, cx| {
            assert!(workspace.bottom_dock().read(cx).is_open());
            assert!(
                workspace
                    .bottom_dock()
                    .read(cx)
                    .panel_for_id(panel.entity_id())
                    .is_some()
            );
        });
        panel.update_in(cx, |panel, window, _| {
            assert!(panel.focus_handle.is_focused(window));
        });
        assert_eq!(
            project_queries.load(Ordering::SeqCst),
            queries_before_options
        );
        let write = panel
            .update(cx, |panel, _| panel.preferences_write.take())
            .expect("ordered preferences write");
        write.await;
        let expected_preferences = panel.read_with(cx, |panel, _| {
            assert!(
                panel.preferences_notice.is_none(),
                "workspace preferences saved"
            );
            panel.preferences.clone()
        });
        let restored = DependenciesPanel::load(
            workspace.downgrade(),
            cx.update(|window, cx| window.to_async(cx)),
        )
        .await
        .expect("restore workspace presentation");
        restored.read_with(cx, |panel, _| {
            assert_eq!(panel.preferences, expected_preferences);
            assert_eq!(panel.position, DockPosition::Bottom);
            assert_eq!(panel.view, BuildToolView::Maven);
            assert!(!panel.active);
            assert!(
                panel.navigation.is_empty(),
                "persistence must not restore provider navigation"
            );
        });
        let preferences_key = panel
            .read_with(cx, |panel, _| panel.preferences_key.clone())
            .expect("workspace preference key");
        let store = cx.read(|cx| KeyValueStore::global(cx));
        let released_key = format!("{preferences_key}:released-panel-fixture");
        let (release_predecessor, predecessor) = oneshot::channel();
        restored.update_in(cx, |panel, window, cx| {
            panel.preferences_write = Some(cx.background_executor().spawn(async move {
                predecessor
                    .await
                    .expect("release queued preference predecessor");
            }));
            panel.preferences_key = Some(released_key.clone());
            panel.set_position(DockPosition::Left, window, cx);
        });
        drop(restored);
        cx.update(|_, _| ());
        cx.run_until_parked();
        assert!(
            release_predecessor.send(()).is_ok(),
            "panel release must preserve the pending preference chain"
        );
        cx.run_until_parked();
        store.write(|_| ()).await;
        let released = store
            .read_kvp(&released_key)
            .expect("released panel preference read")
            .expect("queued position save must survive panel release");
        assert_eq!(
            BuildViewPreferences::restore(Some(&released))
                .expect("released panel preference record")
                .dock_position,
            settings::DockPosition::Left
        );
        store
            .delete_kvp(released_key)
            .await
            .expect("remove released-panel fixture state");
        panel.update_in(cx, |panel, window, cx| {
            panel.preferences_key = None;
            panel.set_position(DockPosition::Left, window, cx);
            assert!(
                panel
                    .preferences_notice
                    .as_ref()
                    .expect("missing identity notice")
                    .contains("persistent identity")
            );
            panel.preferences_key = Some(preferences_key.clone());
        });
        let escaped_key = preferences_key.replace('\'', "''");
        store
            .write(move |connection| {
                connection.exec(&format!(
                    "CREATE TRIGGER groovy_dock_preference_failure BEFORE INSERT ON kv_store WHEN NEW.key = '{escaped_key}' BEGIN SELECT RAISE(ABORT, '/private/preference-canary'); END;"
                ))?()
            })
            .await
            .expect("controlled fixture-key write failure");
        panel.update_in(cx, |panel, window, cx| {
            panel.set_position(DockPosition::Right, window, cx);
        });
        panel
            .update(cx, |panel, _| panel.preferences_write.take())
            .expect("failed preference write")
            .await;
        panel.read_with(cx, |panel, _| {
            let notice = panel.preferences_notice.as_ref().expect("write notice");
            assert!(notice.contains("could not be saved"));
            assert!(!notice.contains("preference-canary"));
        });
        assert_eq!(
            BuildViewPreferences::restore(
                store
                    .read_kvp(&preferences_key)
                    .expect("preserved saved preferences")
                    .as_deref()
            )
            .expect("saved preference record"),
            expected_preferences
        );
        store
            .write(|connection| connection.exec("DROP TRIGGER groovy_dock_preference_failure;")?())
            .await
            .expect("remove controlled write failure");
        panel.update_in(cx, |panel, window, cx| {
            panel.set_position(DockPosition::Bottom, window, cx);
        });
        panel
            .update(cx, |panel, _| panel.preferences_write.take())
            .expect("recovery preference write")
            .await;
        panel.read_with(cx, |panel, _| assert!(panel.preferences_notice.is_none()));
        let unsupported = r#"{"version":2,"project_path":"/private/provider/root"}"#;
        store
            .write_kvp(preferences_key.clone(), unsupported.into())
            .await
            .expect("controlled unsupported preference record");
        let recovered = DependenciesPanel::load(
            workspace.downgrade(),
            cx.update(|window, cx| window.to_async(cx)),
        )
        .await
        .expect("recover unsupported workspace preferences");
        recovered.read_with(cx, |panel, _| {
            assert_eq!(panel.preferences, BuildViewPreferences::default());
            let notice = panel
                .preferences_notice
                .as_ref()
                .expect("visible recovery notice");
            assert!(notice.contains("Defaults are shown"));
            assert!(!notice.contains("/private/provider/root"));
        });
        assert_eq!(
            store
                .read_kvp(&preferences_key)
                .expect("preserved unsupported bytes")
                .as_deref(),
            Some(unsupported)
        );
        panel.update(cx, |panel, cx| {
            panel.toggle_option(BuildViewOption::ShowToolbar, cx);
            panel.show_options = false;
        });
        panel.update_in(cx, |panel, window, cx| {
            panel.set_view(BuildToolView::Dependencies, window, cx);
            panel.host.expand_all();
            cx.notify();
        });
        cx.dispatch_action(language_tool_tree::SelectFirst);
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel.host.selected(),
                Some(&panel.host.visible_rows()[0].id)
            );
        });
        cx.simulate_keystrokes("down");
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel.host.selected(),
                Some(&panel.host.visible_rows()[1].id)
            );
        });
        cx.simulate_keystrokes("up");
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel.host.selected(),
                Some(&panel.host.visible_rows()[0].id)
            );
        });
        cx.simulate_keystrokes("space");
        panel.read_with(cx, |panel, _| {
            assert_eq!(panel.host.visible_rows().len(), 1);
            assert!(
                panel
                    .navigation
                    .contains_key(&panel.host.visible_rows()[0].id)
            );
        });
        let directory_reveals = cx.new(|_| Vec::<project::ProjectEntryId>::new());
        let _directory_reveal_subscription = cx.update(|_, cx| {
            cx.subscribe(&project, {
                let directory_reveals = directory_reveals.clone();
                move |_, event, cx| {
                    if let project::Event::RevealInProjectPanel(entry_id) = event {
                        directory_reveals.update(cx, |reveals, _| reveals.push(*entry_id));
                    }
                }
            })
        });
        panel.update_in(cx, |panel, window, cx| {
            panel.set_view(BuildToolView::Gradle, window, cx);
            panel.host.expand_all();
        });
        let mut compiler_output_identity = None;
        let mut compiler_output_accesskit_id = None;
        for mode in [0, 1, 2, 3, 4, 5, 6, 2] {
            compiler_output_mode.store(mode, Ordering::SeqCst);
            panel.update(cx, |panel, cx| panel.synchronize_gradle_metadata(cx));
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(300));
            cx.run_until_parked();
            let identity = panel.update_in(cx, |panel, window, cx| {
                panel.host.expand_all();
                let row = panel
                    .host
                    .visible_rows()
                    .iter()
                    .find(|row| row.id.0.ends_with(":annotation-processor-output"))
                    .expect("Compiler-output role in actual provider tree");
                assert!(
                    row.accessibility_label
                        .contains("other generated-source coverage unknown")
                );
                let available = matches!(mode, 2 | 3);
                assert_eq!(panel.navigation.contains_key(&row.id), available);
                if mode == 1 {
                    assert!(
                        row.accessibility_label
                            .contains("No output directory configured")
                    );
                }
                if mode == 5 {
                    assert!(
                        !row.label.contains(".env") && !row.accessibility_label.contains(".env")
                    );
                }
                if mode == 6 {
                    assert!(row.accessibility_label.contains("not a directory"));
                }
                let identity = row.id.clone();
                panel.host.select(identity.clone());
                panel.host.reveal_selection(ScrollStrategy::Center);
                panel.focus_handle.focus(window, cx);
                cx.notify();
                identity
            });
            if let Some(previous) = &compiler_output_identity {
                assert_eq!(
                    previous, &identity,
                    "Role identity survives knowledge and path changes"
                );
            }
            compiler_output_identity = Some(identity.clone());
            if matches!(mode, 2 | 3) {
                for _ in 0..2 {
                    cx.update(|window, cx| {
                        window.draw(cx).clear(cx);
                        let tree: serde_json::Value = serde_json::from_str(
                            &window
                                .debug_a11y_tree_json()
                                .expect("Compiler-output accessibility"),
                        )
                        .expect("Accessibility JSON");
                        let nodes = tree["nodes"].as_object().expect("Nodes");
                        let (node_id, node) = nodes
                            .iter()
                            .find(|(_, node)| {
                                node["aria"]["role"] == "TreeItem"
                                    && node["aria"]["label"].as_str().is_some_and(|label| {
                                        label.starts_with(
                                            "Gradle annotation-processor output application/",
                                        )
                                    })
                            })
                            .expect("Rendered output row");
                        assert_eq!(node["aria"]["selected"], true);
                        assert_eq!(
                            tree["active_descendant_focus"].as_str(),
                            Some(node_id.as_str())
                        );
                        let focus = tree["gpui_focus"].as_str().expect("Keyboard focus");
                        assert_eq!(nodes[focus]["aria"]["label"], "Gradle panel");
                        if let Some(previous) = &compiler_output_accesskit_id {
                            assert_eq!(previous, &node["accesskit_id"]);
                        }
                        compiler_output_accesskit_id = Some(node["accesskit_id"].clone());
                    });
                }
            }
            let before_reveals = directory_reveals.read_with(cx, |reveals, _| reveals.len());
            let before_queries = project_queries.load(Ordering::SeqCst);
            let before_configurations = configuring_queries.load(Ordering::SeqCst);
            let before_items = workspace.read_with(cx, |workspace, cx| {
                workspace.active_pane().read(cx).items().count()
            });
            cx.simulate_keystrokes("enter");
            cx.run_until_parked();
            let expected_reveals = before_reveals + usize::from(matches!(mode, 2 | 3));
            assert_eq!(
                directory_reveals.read_with(cx, |reveals, _| reveals.len()),
                expected_reveals
            );
            assert_eq!(project_queries.load(Ordering::SeqCst), before_queries);
            assert_eq!(
                configuring_queries.load(Ordering::SeqCst),
                before_configurations
            );
            assert_eq!(
                workspace.read_with(cx, |workspace, cx| workspace
                    .active_pane()
                    .read(cx)
                    .items()
                    .count()),
                before_items
            );
            if matches!(mode, 2 | 3) {
                let directory = if mode == 2 {
                    "application/processor-generated"
                } else {
                    "application/other-processor-generated"
                };
                let entry_id = worktree_store.read_with(cx, |store, cx| {
                    store
                        .worktree_for_id(worktree_id, cx)
                        .expect("Current worktree")
                        .read(cx)
                        .entry_for_path(RelPath::from_unix_str(directory).expect("Output path"))
                        .expect("Current directory")
                        .id
                });
                assert_eq!(
                    directory_reveals.read_with(cx, |reveals, _| reveals.last().copied()),
                    Some(entry_id)
                );
            }
        }
        let compiler_output_identity = compiler_output_identity.expect("Stable output role");
        let configurations_before_entry_change = configuring_queries.load(Ordering::SeqCst);
        fs.insert_tree(
            "/outside-generated-private-fixture",
            serde_json::json!({"Generated.java":"outside"}),
        )
        .await;
        project::Fs::create_symlink(
            fs.as_ref(),
            std::path::Path::new("/groovy-panel/application/processor-generated"),
            "/outside-generated-private-fixture".into(),
        )
        .await
        .expect("Replace only the fake output directory");
        cx.run_until_parked();
        let reveals_before_external = directory_reveals.read_with(cx, |reveals, _| reveals.len());
        panel.update_in(cx, |panel, window, cx| {
            panel.host.select(compiler_output_identity.clone());
            panel.activate_selected(window, cx);
        });
        cx.run_until_parked();
        assert_eq!(
            directory_reveals.read_with(cx, |reveals, _| reveals.len()),
            reveals_before_external
        );
        panel.read_with(cx, |panel, _| {
            let row = panel
                .host
                .visible_rows()
                .iter()
                .find(|row| row.id == compiler_output_identity)
                .expect("Retained output role");
            assert!(!panel.navigation.contains_key(&compiler_output_identity));
            assert!(
                !row.accessibility_label
                    .contains("outside-generated-private-fixture")
            );
        });
        project::Fs::rename(
            fs.as_ref(),
            std::path::Path::new("/groovy-panel/application/processor-generated"),
            std::path::Path::new("/groovy-panel/application/retired-processor-symlink"),
            project::RenameOptions::default(),
        )
        .await
        .expect("Retire only the in-memory fake symlink");
        fs.insert_tree(
            "/groovy-panel/application/processor-generated",
            serde_json::json!({"Generated.java":"restored fixture"}),
        )
        .await;
        cx.run_until_parked();
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            configurations_before_entry_change
        );
        panel.update(cx, |panel, cx| panel.synchronize_gradle_metadata(cx));
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        panel.update_in(cx, |panel, window, cx| {
            panel.host.select(compiler_output_identity.clone());
            panel.activate_selected(window, cx);
        });
        cx.run_until_parked();
        assert_eq!(
            directory_reveals.read_with(cx, |reveals, _| reveals.len()),
            reveals_before_external + 1
        );
        let (configuring_started, configuring_started_receiver) = oneshot::channel();
        let (configuring_release, configuring_release_receiver) = oneshot::channel();
        *configuring_gate.lock().expect("configuring gate lock") =
            Some((configuring_started, configuring_release_receiver));
        let configuring_before_held = configuring_queries.load(Ordering::SeqCst);
        panel.update(cx, |panel, cx| panel.synchronize_gradle_metadata(cx));
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        assert!(matches!(
            configuring_started_receiver.now_or_never(),
            Some(Ok(()))
        ));
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            configuring_before_held + 1
        );
        fs.insert_file(
            std::path::Path::new("/groovy-panel/application/pom.xml"),
            b"<project><version>2</version></project>".to_vec(),
        )
        .await;
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(matches!(
                panel.host.status(),
                language_tools::language_tool_tree::LanguageToolTreeStatus::Stale(_)
            ));
        });
        match configuring_release.send(()) {
            Ok(()) => {}
            Err(()) => assert!(lsp_store.read_with(cx, |store, _| {
                !store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("cancelled snapshot")
                    .1
            })),
        }
        cx.run_until_parked();
        assert!(
            lsp_store.read_with(cx, |store, _| !store
                .jdtls_project_snapshot(worktree_id)
                .expect("late configuring snapshot")
                .1),
            "late configuration reply cannot restore current provenance"
        );
        let reveals_before_stale = directory_reveals.read_with(cx, |reveals, _| reveals.len());
        panel.update_in(cx, |panel, window, cx| {
            panel.host.select(compiler_output_identity.clone());
            panel.activate_selected(window, cx);
        });
        assert_eq!(
            directory_reveals.read_with(cx, |reveals, _| reveals.len()),
            reveals_before_stale
        );
        fail_queries.store(true, Ordering::SeqCst);
        panel.update(cx, |panel, cx| panel.refresh(cx));
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            let language_tools::language_tool_tree::LanguageToolTreeStatus::StaleError(message) =
                panel.host.status()
            else {
                panic!("failed refresh should keep stale rows with an error");
            };
            assert!(message.contains("View language server logs"));
            assert!(!message.contains("/private/host/path"));
        });
        trusted_worktrees.update(cx, |trusted, cx| {
            trusted.restrict(
                worktree_store.downgrade(),
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            );
        });
        cx.run_until_parked();
        let configuring_before_restricted = configuring_queries.load(Ordering::SeqCst);
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        let denied = lsp_store
            .update(cx, |store, cx| {
                store.synchronize_local_gradle_project(Some(worktree_id), cx)
            })
            .expect_err("untrusted synchronization must deny");
        assert!(denied.to_string().contains("requires project trust"));
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            configuring_before_restricted
        );

        panel.read_with(cx, |panel, _| {
            assert!(panel.host.visible_rows().is_empty());
            assert!(panel.navigation.is_empty());
            assert!(matches!(
                panel.host.status(),
                language_tools::language_tool_tree::LanguageToolTreeStatus::Restricted(_)
            ));
        });
        let queries_before_restricted_refresh = project_queries.load(Ordering::SeqCst);
        panel.update(cx, |panel, cx| panel.refresh(cx));
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(300));
        cx.run_until_parked();
        assert_eq!(
            project_queries.load(Ordering::SeqCst),
            queries_before_restricted_refresh,
            "refresh must not query a restricted worktree's JVM provider"
        );
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }
}
