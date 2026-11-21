//! Staged context targets and the menu model for the dependency diagram
//! canvas, following `sim_ui`'s staged-target pattern: secondary-click staging
//! decides the target, the menu model groups commands with separators and
//! availability-filters them through the same enablement inputs the panel
//! controls use, and dispatch handlers re-check availability so a command
//! invoked on an unsupported target is rejected instead of dispatched.

use std::sync::Mutex;

use crate::dependencies_panel::{
    AddGraphSelectionToSet, ClearGraphInspection, ClearGraphSelectionSet,
    CopyDependencyDiagramSource, InspectGraphAllUsages, InspectGraphNeighbours,
    InspectGraphPathsBetween, InspectGraphRootPaths, InspectGraphSelectionSet, PinGraphSelection,
};
use crate::dependency_diagram_item::{
    ExportDependencyDiagramImage, FitDependencyDiagram, ShowDependencyDiagramPopup,
    ToggleDependencyDiagramStructure, ZoomDependencyDiagramIn, ZoomDependencyDiagramOut,
};
use sim_runtime::graph::GraphPoint;

/// The object a context menu acts on, staged from the secondary-click target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DependencyDiagramContextTarget {
    /// Blank canvas: the menu operates on the view.
    Canvas,
    /// A secondary-click inside the multi-selection preserves the selection.
    Selection,
    /// A secondary-click on an unselected node targets that node.
    Node(String),
}

/// The staged context: the target plus the window position the menu opens at.
#[derive(Clone, Debug, PartialEq)]
pub struct DependencyDiagramContextInvocation {
    pub target: DependencyDiagramContextTarget,
    pub screen_position: GraphPoint,
}

/// The availability inputs shared with the panel's enablement logic, captured
/// when the menu opens.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DependencyDiagramMenuSnapshot {
    pub selection_len: usize,
    pub selection_set_len: usize,
    pub has_inspection: bool,
    pub has_source: bool,
}

/// The commands the diagram menus can dispatch. Panel-level commands route
/// through the panel's own action handlers; canvas-level commands are handled
/// by the diagram item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyDiagramMenuCommand {
    InspectNeighbours,
    InspectRootPaths,
    PathsBetween,
    AllUsages,
    PinSelection,
    AddToSelectionSet,
    InspectSelectionSet,
    ClearSelectionSet,
    ClearInspection,
    ShowPopup,
    CopyDiagramSource,
    ExportImage,
    Fit,
    ZoomIn,
    ZoomOut,
    ToggleStructure,
}

/// Stages the context target from a secondary click: an unselected node
/// targets that node, a click inside the multi-selection preserves the
/// selection, and a blank-canvas click acts on the view.
pub fn stage_dependency_diagram_target(
    hit_node: Option<&str>,
    selection: &std::collections::BTreeSet<String>,
) -> DependencyDiagramContextTarget {
    match hit_node {
        Some(node) if selection.contains(node) => DependencyDiagramContextTarget::Selection,
        Some(node) => DependencyDiagramContextTarget::Node(node.to_owned()),
        None => DependencyDiagramContextTarget::Canvas,
    }
}

/// Whether the command is available for the staged target and state. The menu
/// model filters with this and dispatch handlers re-check it, so a
/// wrong-target invocation is rejected.
pub fn dependency_diagram_menu_available(
    command: DependencyDiagramMenuCommand,
    target: &DependencyDiagramContextTarget,
    snapshot: &DependencyDiagramMenuSnapshot,
) -> bool {
    let node_or_selection = matches!(target, DependencyDiagramContextTarget::Node(_))
        || matches!(target, DependencyDiagramContextTarget::Selection);
    match command {
        DependencyDiagramMenuCommand::InspectNeighbours => node_or_selection,
        DependencyDiagramMenuCommand::InspectRootPaths => {
            matches!(target, DependencyDiagramContextTarget::Node(_))
        }
        DependencyDiagramMenuCommand::AllUsages => node_or_selection,
        DependencyDiagramMenuCommand::PathsBetween => {
            node_or_selection && snapshot.selection_set_len >= 2
        }
        DependencyDiagramMenuCommand::PinSelection => {
            node_or_selection && snapshot.selection_len >= 1
        }
        DependencyDiagramMenuCommand::AddToSelectionSet => node_or_selection,
        DependencyDiagramMenuCommand::InspectSelectionSet => {
            snapshot.selection_set_len >= 1 && node_or_selection
        }
        DependencyDiagramMenuCommand::ClearSelectionSet => snapshot.selection_set_len >= 1,
        DependencyDiagramMenuCommand::ClearInspection => snapshot.has_inspection,
        DependencyDiagramMenuCommand::ShowPopup => true,
        DependencyDiagramMenuCommand::CopyDiagramSource => snapshot.has_source,
        DependencyDiagramMenuCommand::ExportImage => true,
        DependencyDiagramMenuCommand::Fit
        | DependencyDiagramMenuCommand::ZoomIn
        | DependencyDiagramMenuCommand::ZoomOut
        | DependencyDiagramMenuCommand::ToggleStructure => true,
    }
}

/// The accessible label for a command, shared by menus, buttons and tests.
pub fn dependency_diagram_menu_label(command: DependencyDiagramMenuCommand) -> &'static str {
    match command {
        DependencyDiagramMenuCommand::InspectNeighbours => "Inspect neighbours",
        DependencyDiagramMenuCommand::InspectRootPaths => "Inspect root paths",
        DependencyDiagramMenuCommand::AllUsages => "All usages",
        DependencyDiagramMenuCommand::PathsBetween => "Inspect paths between",
        DependencyDiagramMenuCommand::PinSelection => "Pin dependency",
        DependencyDiagramMenuCommand::AddToSelectionSet => "Add to selection set",
        DependencyDiagramMenuCommand::InspectSelectionSet => "Inspect selection set",
        DependencyDiagramMenuCommand::ClearSelectionSet => "Clear selection set",
        DependencyDiagramMenuCommand::ClearInspection => "Clear inspection",
        DependencyDiagramMenuCommand::ShowPopup => "Show Dependency Diagram Popup",
        DependencyDiagramMenuCommand::CopyDiagramSource => "Copy Diagram Source",
        DependencyDiagramMenuCommand::ExportImage => "Export Image",
        DependencyDiagramMenuCommand::Fit => "Fit Diagram",
        DependencyDiagramMenuCommand::ZoomIn => "Zoom In",
        DependencyDiagramMenuCommand::ZoomOut => "Zoom Out",
        DependencyDiagramMenuCommand::ToggleStructure => "Toggle Structure Overview",
    }
}

/// One grouped menu level: frequent actions stay one level deep and groups
/// render with separators between them.
pub fn dependency_diagram_menu_groups(
    target: &DependencyDiagramContextTarget,
    snapshot: &DependencyDiagramMenuSnapshot,
) -> Vec<Vec<DependencyDiagramMenuCommand>> {
    let inspection = [
        DependencyDiagramMenuCommand::InspectNeighbours,
        DependencyDiagramMenuCommand::InspectRootPaths,
        DependencyDiagramMenuCommand::PathsBetween,
        DependencyDiagramMenuCommand::AllUsages,
        DependencyDiagramMenuCommand::PinSelection,
        DependencyDiagramMenuCommand::AddToSelectionSet,
        DependencyDiagramMenuCommand::InspectSelectionSet,
        DependencyDiagramMenuCommand::ClearSelectionSet,
        DependencyDiagramMenuCommand::ClearInspection,
    ];
    let surfaces = [
        DependencyDiagramMenuCommand::ShowPopup,
        DependencyDiagramMenuCommand::CopyDiagramSource,
        DependencyDiagramMenuCommand::ExportImage,
    ];
    let view = [
        DependencyDiagramMenuCommand::Fit,
        DependencyDiagramMenuCommand::ZoomIn,
        DependencyDiagramMenuCommand::ZoomOut,
        DependencyDiagramMenuCommand::ToggleStructure,
    ];
    let groups = vec![inspection.to_vec(), surfaces.to_vec(), view.to_vec()];
    let mut model = Vec::new();
    for group in groups {
        let available: Vec<_> = group
            .into_iter()
            .filter(|command| dependency_diagram_menu_available(*command, target, snapshot))
            .collect();
        if !available.is_empty() {
            model.push(available);
        }
    }
    model
}

/// The boxed action a menu entry dispatches. Panel-level commands are the
/// panel's own actions so routing and enablement stay shared.
pub fn dependency_diagram_menu_action(
    command: DependencyDiagramMenuCommand,
) -> Box<dyn gpui::Action> {
    match command {
        DependencyDiagramMenuCommand::InspectNeighbours => Box::new(InspectGraphNeighbours),
        DependencyDiagramMenuCommand::InspectRootPaths => Box::new(InspectGraphRootPaths),
        DependencyDiagramMenuCommand::AllUsages => Box::new(InspectGraphAllUsages),
        DependencyDiagramMenuCommand::PathsBetween => Box::new(InspectGraphPathsBetween),
        DependencyDiagramMenuCommand::PinSelection => Box::new(PinGraphSelection),
        DependencyDiagramMenuCommand::AddToSelectionSet => Box::new(AddGraphSelectionToSet),
        DependencyDiagramMenuCommand::InspectSelectionSet => Box::new(InspectGraphSelectionSet),
        DependencyDiagramMenuCommand::ClearSelectionSet => Box::new(ClearGraphSelectionSet),
        DependencyDiagramMenuCommand::ClearInspection => Box::new(ClearGraphInspection),
        DependencyDiagramMenuCommand::ShowPopup => Box::new(ShowDependencyDiagramPopup),
        DependencyDiagramMenuCommand::CopyDiagramSource => Box::new(CopyDependencyDiagramSource),
        DependencyDiagramMenuCommand::ExportImage => Box::new(ExportDependencyDiagramImage),
        DependencyDiagramMenuCommand::Fit => Box::new(FitDependencyDiagram),
        DependencyDiagramMenuCommand::ZoomIn => Box::new(ZoomDependencyDiagramIn),
        DependencyDiagramMenuCommand::ZoomOut => Box::new(ZoomDependencyDiagramOut),
        DependencyDiagramMenuCommand::ToggleStructure => Box::new(ToggleDependencyDiagramStructure),
    }
}

/// The staged invocation and the availability snapshot the canvas writes and
/// the menu builder reads.
#[derive(Default)]
pub struct DependencyDiagramContextState {
    pub invocation: Mutex<Option<DependencyDiagramContextInvocation>>,
    pub snapshot: Mutex<DependencyDiagramMenuSnapshot>,
}

impl DependencyDiagramContextState {
    /// Stages an invocation for the next menu build.
    pub fn stage(&self, invocation: DependencyDiagramContextInvocation) {
        *self.invocation.lock().expect("staged invocation") = Some(invocation);
    }

    /// Takes the staged invocation so a menu is built once per staging.
    pub fn take_invocation(&self) -> Option<DependencyDiagramContextInvocation> {
        self.invocation.lock().expect("staged invocation").take()
    }

    pub fn snapshot(&self) -> DependencyDiagramMenuSnapshot {
        *self.snapshot.lock().expect("menu snapshot")
    }
}

/// Builds the native context menu for the staged target: groups render with
/// separators between them, every entry is availability-filtered, and a menu
/// with no available commands is not built at all (wrong-target rejection).
pub fn build_dependency_diagram_context_menu(
    state: &DependencyDiagramContextState,
    window: &mut gpui::Window,
    cx: &mut gpui::App,
) -> Option<gpui::Entity<ui::ContextMenu>> {
    let invocation = state.take_invocation()?;
    let snapshot = state.snapshot();
    let groups = dependency_diagram_menu_groups(&invocation.target, &snapshot);
    if groups.is_empty() {
        return None;
    }
    Some(ui::ContextMenu::build(window, cx, |menu, _, _| {
        let mut menu = menu;
        for (group_index, group) in groups.iter().enumerate() {
            if group_index > 0 {
                menu = menu.separator();
            }
            for command in group {
                menu = menu.action(
                    dependency_diagram_menu_label(*command),
                    dependency_diagram_menu_action(*command),
                );
            }
        }
        menu
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn selection_with(node: &str) -> BTreeSet<String> {
        let mut selection = BTreeSet::new();
        selection.insert(node.to_owned());
        selection
    }

    #[gpui::test]
    fn dependency_diagram_menu_stages_targets(cx: &mut gpui::TestAppContext) {
        let _ = cx;
        // An unselected node targets that node.
        assert_eq!(
            stage_dependency_diagram_target(Some("n1"), &selection_with("other")),
            DependencyDiagramContextTarget::Node("n1".into())
        );
        // A secondary-click inside the multi-selection preserves it.
        assert_eq!(
            stage_dependency_diagram_target(Some("n1"), &selection_with("n1")),
            DependencyDiagramContextTarget::Selection
        );
        // A blank-canvas click acts on the view.
        assert_eq!(
            stage_dependency_diagram_target(None, &selection_with("n1")),
            DependencyDiagramContextTarget::Canvas
        );
    }

    #[gpui::test]
    fn dependency_diagram_menu_groups_and_filters(cx: &mut gpui::TestAppContext) {
        let _ = cx;
        let node = DependencyDiagramContextTarget::Node("n1".into());
        let canvas = DependencyDiagramContextTarget::Canvas;
        // Node target: inspection commands lead, surfaces and view follow.
        let node_snapshot = DependencyDiagramMenuSnapshot {
            selection_len: 1,
            selection_set_len: 2,
            has_inspection: true,
            has_source: true,
        };
        let groups = dependency_diagram_menu_groups(&node, &node_snapshot);
        assert_eq!(
            groups.len(),
            3,
            "node menus group inspection, surfaces, view"
        );
        assert_eq!(
            groups[0]
                .iter()
                .map(|command| dependency_diagram_menu_label(*command))
                .collect::<Vec<_>>()
                .first()
                .copied(),
            Some("Inspect neighbours"),
            "frequent inspection actions lead the menu"
        );
        assert!(groups[0].contains(&DependencyDiagramMenuCommand::PathsBetween));
        // Availability filtering: Copy Diagram Source requires a source.
        let without_source = DependencyDiagramMenuSnapshot {
            has_source: false,
            ..node_snapshot
        };
        let groups = dependency_diagram_menu_groups(&node, &without_source);
        assert!(
            !groups
                .iter()
                .flatten()
                .any(|command| *command == DependencyDiagramMenuCommand::CopyDiagramSource),
            "Copy Diagram Source is filtered without a provider source"
        );
        // Blank canvas: inspection commands are rejected as wrong-target.
        let groups = dependency_diagram_menu_groups(&canvas, &node_snapshot);
        assert!(
            !groups.iter().flatten().any(|command| {
                matches!(
                    command,
                    DependencyDiagramMenuCommand::InspectNeighbours
                        | DependencyDiagramMenuCommand::PinSelection
                )
            }),
            "blank-canvas menus exclude node commands"
        );
        assert!(
            groups
                .iter()
                .flatten()
                .any(|command| *command == DependencyDiagramMenuCommand::Fit)
        );
    }

    #[gpui::test]
    fn dependency_diagram_menu_rejects_wrong_targets(cx: &mut gpui::TestAppContext) {
        let _ = cx;
        let canvas = DependencyDiagramContextTarget::Canvas;
        let snapshot = DependencyDiagramMenuSnapshot {
            selection_len: 1,
            selection_set_len: 0,
            has_inspection: false,
            has_source: true,
        };
        // Wrong-target dispatch is rejected through the same availability
        // logic the menu model filters with.
        assert!(!dependency_diagram_menu_available(
            DependencyDiagramMenuCommand::InspectRootPaths,
            &canvas,
            &snapshot
        ));
        assert!(!dependency_diagram_menu_available(
            DependencyDiagramMenuCommand::PinSelection,
            &canvas,
            &snapshot
        ));
        assert!(!dependency_diagram_menu_available(
            DependencyDiagramMenuCommand::PathsBetween,
            &DependencyDiagramContextTarget::Node("n1".into()),
            &snapshot
        ));
        assert!(dependency_diagram_menu_available(
            DependencyDiagramMenuCommand::Fit,
            &canvas,
            &snapshot
        ));
    }

    #[gpui::test]
    fn analyzer_diagram_linkage_all_usages_targets_dependencies(cx: &mut gpui::TestAppContext) {
        let _ = cx;
        let node = DependencyDiagramContextTarget::Node("n1".into());
        let selection = DependencyDiagramContextTarget::Selection;
        let canvas = DependencyDiagramContextTarget::Canvas;
        let snapshot = DependencyDiagramMenuSnapshot {
            selection_len: 1,
            selection_set_len: 1,
            has_inspection: false,
            has_source: true,
        };
        // All usages targets a dependency node or an in-selection click, and
        // rides in the inspection group one level deep.
        assert!(dependency_diagram_menu_available(
            DependencyDiagramMenuCommand::AllUsages,
            &node,
            &snapshot
        ));
        assert!(dependency_diagram_menu_available(
            DependencyDiagramMenuCommand::AllUsages,
            &selection,
            &snapshot
        ));
        assert!(
            !dependency_diagram_menu_available(
                DependencyDiagramMenuCommand::AllUsages,
                &canvas,
                &snapshot
            ),
            "a blank-canvas menu does not offer All usages"
        );
        let groups = dependency_diagram_menu_groups(&node, &snapshot);
        assert!(
            groups[0].contains(&DependencyDiagramMenuCommand::AllUsages),
            "All usages stays in the inspection group"
        );
        assert_eq!(
            dependency_diagram_menu_label(DependencyDiagramMenuCommand::AllUsages),
            "All usages"
        );
        // The entry dispatches the shared InspectGraphAllUsages command.
        let mapped = dependency_diagram_menu_action(DependencyDiagramMenuCommand::AllUsages);
        assert!(
            mapped.partial_eq(&InspectGraphAllUsages),
            "All usages dispatches the panel's shared command"
        );
    }
}
