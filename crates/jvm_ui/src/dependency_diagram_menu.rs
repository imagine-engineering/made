//! Staged context targets and the menu model for the dependency diagram
//! canvas, following `sim_ui`'s staged-target pattern: secondary-click staging
//! decides the target, the menu model groups commands with separators and
//! availability-filters them through the same enablement inputs the panel
//! controls use, and dispatch handlers re-check availability so a command
//! invoked on an unsupported target is rejected instead of dispatched.

use std::sync::Mutex;

use crate::dependencies_panel::{
    AddGraphSelectionToSet, ClearGraphInspection, ClearGraphSelectionSet,
    CopyDependencyDiagramSource, GraphInspectionMode, InspectGraphAllRootPaths,
    InspectGraphAllUsages, InspectGraphConflictOrigins, InspectGraphNeighbours,
    InspectGraphPathsBetween, InspectGraphRootPaths, InspectGraphSelectionSet, PinGraphSelection,
};
use crate::dependency_diagram_item::{
    ExportDependencyDiagramImage, FitDependencyDiagram, SaveDependencyDiagramImage,
    ToggleDependencyDiagramStructure, UseHorizontalDependencyLayout, UseVerticalDependencyLayout,
    ZoomDependencyDiagramIn, ZoomDependencyDiagramOut,
};
use sim_runtime::graph::GraphPoint;

/// The object a context menu acts on, staged from the secondary-click target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DependencyDiagramContextTarget {
    /// Blank canvas: the menu operates on the view.
    Canvas,
    /// A secondary-click inside the multi-selection preserves the selection.
    Selection(String),
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
    pub selection_set_len: usize,
    pub has_inspection: bool,
    pub has_source: bool,
    pub has_admitted_node: bool,
    pub has_matching_pin: bool,
    pub can_add_to_set: bool,
    pub has_conflict_origins: bool,
}

/// The commands the diagram menus can dispatch. Panel-level commands route
/// through the panel's own action handlers; canvas-level commands are handled
/// by the diagram item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyDiagramMenuCommand {
    InspectNeighbours,
    InspectRootPaths,
    InspectAllRootPaths,
    InspectConflictOrigins,
    PathsBetween,
    AllUsages,
    PinSelection,
    AddToSelectionSet,
    InspectSelectionSet,
    ClearSelectionSet,
    ClearInspection,
    CopyDiagramSource,
    CopyImage,
    ExportImage,
    Fit,
    ZoomIn,
    ZoomOut,
    ToggleStructure,
    LayoutLeftToRight,
    LayoutTopToBottom,
}

impl DependencyDiagramMenuCommand {
    pub(crate) fn inspection_mode(self) -> Option<GraphInspectionMode> {
        match self {
            Self::InspectNeighbours => Some(GraphInspectionMode::Neighbours),
            Self::InspectRootPaths => Some(GraphInspectionMode::ShortestRootPaths),
            Self::InspectAllRootPaths => Some(GraphInspectionMode::AllRootPaths),
            Self::InspectConflictOrigins => Some(GraphInspectionMode::ConflictOrigins),
            Self::PathsBetween => Some(GraphInspectionMode::PathsBetween),
            Self::InspectSelectionSet => Some(GraphInspectionMode::SelectionSetRelations),
            _ => None,
        }
    }
}

impl DependencyDiagramContextTarget {
    pub(crate) fn node(&self) -> Option<&str> {
        match self {
            Self::Canvas => None,
            Self::Node(node) | Self::Selection(node) => Some(node),
        }
    }
}

/// Stages the context target from a secondary click: an unselected node
/// targets that node, a click inside the multi-selection preserves the
/// selection, and a blank-canvas click acts on the view.
pub fn stage_dependency_diagram_target(
    hit_node: Option<&str>,
    selection: &std::collections::BTreeSet<String>,
) -> DependencyDiagramContextTarget {
    match hit_node {
        Some(node) if selection.len() > 1 && selection.contains(node) => {
            DependencyDiagramContextTarget::Selection(node.to_owned())
        }
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
    let node_or_selection = target.node().is_some() && snapshot.has_admitted_node;
    match command {
        DependencyDiagramMenuCommand::InspectNeighbours
        | DependencyDiagramMenuCommand::InspectRootPaths
        | DependencyDiagramMenuCommand::InspectAllRootPaths => {
            node_or_selection && snapshot.has_source
        }
        DependencyDiagramMenuCommand::InspectConflictOrigins => {
            node_or_selection && snapshot.has_source && snapshot.has_conflict_origins
        }
        DependencyDiagramMenuCommand::AllUsages => node_or_selection && snapshot.has_source,
        DependencyDiagramMenuCommand::PathsBetween => {
            node_or_selection && snapshot.has_matching_pin
        }
        DependencyDiagramMenuCommand::PinSelection => node_or_selection && snapshot.has_source,
        DependencyDiagramMenuCommand::AddToSelectionSet => {
            node_or_selection && snapshot.can_add_to_set
        }
        DependencyDiagramMenuCommand::InspectSelectionSet => {
            snapshot.selection_set_len >= 2 && node_or_selection
        }
        DependencyDiagramMenuCommand::ClearSelectionSet => {
            node_or_selection && snapshot.selection_set_len >= 1
        }
        DependencyDiagramMenuCommand::ClearInspection => snapshot.has_inspection,
        DependencyDiagramMenuCommand::CopyDiagramSource => snapshot.has_source,
        DependencyDiagramMenuCommand::CopyImage | DependencyDiagramMenuCommand::ExportImage => {
            snapshot.has_source
        }
        DependencyDiagramMenuCommand::LayoutLeftToRight
        | DependencyDiagramMenuCommand::LayoutTopToBottom => {
            target == &DependencyDiagramContextTarget::Canvas
        }
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
        DependencyDiagramMenuCommand::InspectRootPaths => "Inspect shortest root paths",
        DependencyDiagramMenuCommand::InspectAllRootPaths => "Inspect all root paths",
        DependencyDiagramMenuCommand::InspectConflictOrigins => "Inspect conflict origins",
        DependencyDiagramMenuCommand::AllUsages => "All usages",
        DependencyDiagramMenuCommand::PathsBetween => "Inspect paths between",
        DependencyDiagramMenuCommand::PinSelection => "Pin dependency",
        DependencyDiagramMenuCommand::AddToSelectionSet => "Add to selection set",
        DependencyDiagramMenuCommand::InspectSelectionSet => "Inspect selection set",
        DependencyDiagramMenuCommand::ClearSelectionSet => "Clear selection set",
        DependencyDiagramMenuCommand::ClearInspection => "Clear inspection",
        DependencyDiagramMenuCommand::CopyDiagramSource => "Copy Diagram Source",
        DependencyDiagramMenuCommand::CopyImage => "Copy Diagram Image",
        DependencyDiagramMenuCommand::ExportImage => "Export Image to File",
        DependencyDiagramMenuCommand::Fit => "Fit Diagram",
        DependencyDiagramMenuCommand::ZoomIn => "Zoom In",
        DependencyDiagramMenuCommand::ZoomOut => "Zoom Out",
        DependencyDiagramMenuCommand::ToggleStructure => "Toggle Structure Overview",
        DependencyDiagramMenuCommand::LayoutLeftToRight => "Left to right",
        DependencyDiagramMenuCommand::LayoutTopToBottom => "Top to bottom",
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
        DependencyDiagramMenuCommand::InspectAllRootPaths,
        DependencyDiagramMenuCommand::InspectConflictOrigins,
        DependencyDiagramMenuCommand::PathsBetween,
        DependencyDiagramMenuCommand::AllUsages,
        DependencyDiagramMenuCommand::PinSelection,
        DependencyDiagramMenuCommand::AddToSelectionSet,
        DependencyDiagramMenuCommand::InspectSelectionSet,
        DependencyDiagramMenuCommand::ClearSelectionSet,
        DependencyDiagramMenuCommand::ClearInspection,
    ];
    let surfaces = [
        DependencyDiagramMenuCommand::CopyDiagramSource,
        DependencyDiagramMenuCommand::CopyImage,
        DependencyDiagramMenuCommand::ExportImage,
    ];
    let view = [
        DependencyDiagramMenuCommand::Fit,
        DependencyDiagramMenuCommand::ZoomIn,
        DependencyDiagramMenuCommand::ZoomOut,
        DependencyDiagramMenuCommand::ToggleStructure,
        DependencyDiagramMenuCommand::LayoutLeftToRight,
        DependencyDiagramMenuCommand::LayoutTopToBottom,
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
        DependencyDiagramMenuCommand::InspectAllRootPaths => Box::new(InspectGraphAllRootPaths),
        DependencyDiagramMenuCommand::InspectConflictOrigins => {
            Box::new(InspectGraphConflictOrigins)
        }
        DependencyDiagramMenuCommand::AllUsages => Box::new(InspectGraphAllUsages),
        DependencyDiagramMenuCommand::PathsBetween => Box::new(InspectGraphPathsBetween),
        DependencyDiagramMenuCommand::PinSelection => Box::new(PinGraphSelection),
        DependencyDiagramMenuCommand::AddToSelectionSet => Box::new(AddGraphSelectionToSet),
        DependencyDiagramMenuCommand::InspectSelectionSet => Box::new(InspectGraphSelectionSet),
        DependencyDiagramMenuCommand::ClearSelectionSet => Box::new(ClearGraphSelectionSet),
        DependencyDiagramMenuCommand::ClearInspection => Box::new(ClearGraphInspection),
        DependencyDiagramMenuCommand::CopyDiagramSource => Box::new(CopyDependencyDiagramSource),
        DependencyDiagramMenuCommand::CopyImage => Box::new(ExportDependencyDiagramImage),
        DependencyDiagramMenuCommand::ExportImage => Box::new(SaveDependencyDiagramImage),
        DependencyDiagramMenuCommand::Fit => Box::new(FitDependencyDiagram),
        DependencyDiagramMenuCommand::ZoomIn => Box::new(ZoomDependencyDiagramIn),
        DependencyDiagramMenuCommand::ZoomOut => Box::new(ZoomDependencyDiagramOut),
        DependencyDiagramMenuCommand::ToggleStructure => Box::new(ToggleDependencyDiagramStructure),
        DependencyDiagramMenuCommand::LayoutLeftToRight => Box::new(UseHorizontalDependencyLayout),
        DependencyDiagramMenuCommand::LayoutTopToBottom => Box::new(UseVerticalDependencyLayout),
    }
}

/// The staged invocation and the availability snapshot the canvas writes and
/// the menu builder reads.
#[derive(Default)]
pub struct DependencyDiagramContextState {
    pub invocation: Mutex<Option<DependencyDiagramContextInvocation>>,
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
}

/// Builds the native context menu for the staged target: groups render with
/// separators between them, every entry is availability-filtered, and a menu
/// with no available commands is not built at all (wrong-target rejection).
pub(crate) fn build_dependency_diagram_context_menu(
    state: &DependencyDiagramContextState,
    item: gpui::WeakEntity<crate::dependency_diagram_item::DependencyDiagramItem>,
    window: &mut gpui::Window,
    cx: &mut gpui::App,
) -> Option<gpui::Entity<ui::ContextMenu>> {
    let invocation = state.take_invocation()?;
    let snapshot = item
        .read_with(cx, |item, cx| {
            item.context_menu_snapshot(&invocation.target, cx)
        })
        .ok()?;
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
            use DependencyDiagramMenuCommand::*;

            let mut more = Vec::new();
            let mut comparison = Vec::new();
            let mut layouts = Vec::new();
            let mut clear_inspection = false;
            for &command in group {
                match command {
                    InspectAllRootPaths | InspectConflictOrigins => more.push(command),
                    PinSelection | PathsBetween | AddToSelectionSet | InspectSelectionSet
                    | ClearSelectionSet => comparison.push(command),
                    LayoutLeftToRight | LayoutTopToBottom => layouts.push(command),
                    ClearInspection => clear_inspection = true,
                    _ => {
                        menu = add_menu_command(menu, command, &invocation.target, &item);
                    }
                }
            }
            for (label, commands) in [
                ("More inspections", more),
                ("Compare", comparison),
                ("Layout", layouts),
            ] {
                if !commands.is_empty() {
                    let target = invocation.target.clone();
                    let item = item.clone();
                    menu = menu.submenu(label, move |menu, _, _| {
                        commands.iter().copied().fold(menu, |menu, command| {
                            add_menu_command(menu, command, &target, &item)
                        })
                    });
                }
            }
            if clear_inspection {
                menu =
                    add_menu_command(menu.separator(), ClearInspection, &invocation.target, &item);
            }
        }
        menu
    }))
}

fn add_menu_command(
    menu: ui::ContextMenu,
    command: DependencyDiagramMenuCommand,
    target: &DependencyDiagramContextTarget,
    item: &gpui::WeakEntity<crate::dependency_diagram_item::DependencyDiagramItem>,
) -> ui::ContextMenu {
    use DependencyDiagramMenuCommand::*;

    let label = dependency_diagram_menu_label(command);
    let action = dependency_diagram_menu_action(command);
    if command.inspection_mode().is_some()
        || matches!(
            command,
            ClearInspection | AllUsages | PinSelection | AddToSelectionSet | ClearSelectionSet
        )
    {
        let target = target.clone();
        let item = item.clone();
        menu.entry(label, Some(action), move |_, cx| {
            if let Some(item) = item.upgrade() {
                item.update(cx, |item, cx| {
                    item.inspect_canvas_target(command, &target, cx)
                });
            }
        })
    } else {
        menu.action(label, action)
    }
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
            stage_dependency_diagram_target(
                Some("n1"),
                &BTreeSet::from(["n1".into(), "n2".into()])
            ),
            DependencyDiagramContextTarget::Selection("n1".into())
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
            selection_set_len: 2,
            has_admitted_node: true,
            has_matching_pin: true,
            can_add_to_set: true,
            has_inspection: true,
            has_source: true,
            has_conflict_origins: true,
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
        assert!(groups[0].contains(&DependencyDiagramMenuCommand::InspectConflictOrigins));
        assert!(groups[1].contains(&DependencyDiagramMenuCommand::CopyImage));
        assert!(groups[1].contains(&DependencyDiagramMenuCommand::ExportImage));
        let without_conflict = DependencyDiagramMenuSnapshot {
            has_conflict_origins: false,
            ..node_snapshot
        };
        assert!(!dependency_diagram_menu_available(
            DependencyDiagramMenuCommand::InspectConflictOrigins,
            &node,
            &without_conflict,
        ));
        // Availability filtering: Copy Diagram Source requires a source.
        let without_source = DependencyDiagramMenuSnapshot {
            has_source: false,
            ..node_snapshot
        };
        let groups = dependency_diagram_menu_groups(&node, &without_source);
        assert!(
            !groups.iter().flatten().any(|command| matches!(
                command,
                DependencyDiagramMenuCommand::CopyDiagramSource
                    | DependencyDiagramMenuCommand::CopyImage
                    | DependencyDiagramMenuCommand::ExportImage
            )),
            "Diagram export commands require current provider facts"
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
            has_admitted_node: true,
            selection_set_len: 0,
            has_inspection: false,
            has_source: true,
            ..Default::default()
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
        assert!(dependency_diagram_menu_available(
            DependencyDiagramMenuCommand::LayoutTopToBottom,
            &canvas,
            &snapshot
        ));
        assert!(!dependency_diagram_menu_available(
            DependencyDiagramMenuCommand::LayoutTopToBottom,
            &DependencyDiagramContextTarget::Node("n1".into()),
            &snapshot
        ));
    }

    #[gpui::test]
    fn analyzer_diagram_linkage_all_usages_targets_dependencies(cx: &mut gpui::TestAppContext) {
        let _ = cx;
        let node = DependencyDiagramContextTarget::Node("n1".into());
        let selection = DependencyDiagramContextTarget::Selection("n1".into());
        let canvas = DependencyDiagramContextTarget::Canvas;
        let snapshot = DependencyDiagramMenuSnapshot {
            has_admitted_node: true,
            selection_set_len: 1,
            has_inspection: false,
            has_source: true,
            ..Default::default()
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
