#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
CATALOGS = ROOT / "catalogs"
SOURCE = CATALOGS / "frontend-menus.csv"
OWNERSHIP_OUTPUT = CATALOGS / "native-menu-dispositions.csv"
RUST_OUTPUT = WORKSPACE / "crates/sim_ui/src/generated_menu_catalog.rs"
COMMAND_SOURCE = CATALOGS / "frontend-commands.csv"
COMMAND_OWNERSHIP_OUTPUT = CATALOGS / "native-command-dispositions.csv"
COMMAND_RUST_OUTPUT = WORKSPACE / "crates/sim_ui/src/generated_command_catalog.rs"
KEYBINDING_SOURCE = CATALOGS / "frontend-keybindings.csv"
KEYBINDING_RUST_OUTPUT = WORKSPACE / "crates/sim_ui/src/generated_keybinding_catalog.rs"
KEYMAP_OUTPUT = WORKSPACE / "assets/keymaps/default-sim.json"
ACTIONS_SOURCE = WORKSPACE / "crates/sim_ui/src/actions.rs"
COMPONENT_SOURCE = CATALOGS / "frontend-component-surfaces.csv"
COMPONENT_OWNERSHIP_OUTPUT = CATALOGS / "native-component-dispositions.csv"

GRAPH_SHELL = "comfy-parity-graph-shell-accessibility"
GRAPH_CONTEXT_MENUS = "comfy-parity-graph-context-menu-surfaces"
EXECUTION_UI = "comfy-parity-execution-ui"
WORKFLOW_EXPERIENCE = "comfy-parity-workflow-experience"
ASSET_VIEWERS = "comfy-parity-assets-editors-viewers"
IMAGE_MASK = "comfy-parity-image-mask-content"
MEMORY_PLANNER = "comfy-parity-native-memory-planner"
SETTINGS_UI = "comfy-parity-settings-localization-ui"
DIAGNOSTICS = "comfy-parity-process-diagnostics"
AUTH_CLOUD = "comfy-parity-auth-cloud-telemetry"
UPDATES = "comfy-parity-updates-snapshots"
DESKTOP_UI = "comfy-parity-desktop-native-ui"
COMPATIBILITY = "comfy-parity-backward-compatibility"
THREE_D_CONTENT = "comfy-parity-three-d-latent-content"
EXTENSION_UI = "comfy-parity-frontend-extension-compatibility"
NATIVE_REGISTRY = "comfy-parity-native-registry-integration"
KEY_CONTEXT = "SimGraph"
TEXT_INPUT_KEY_CONTEXT = "SimTextInput"
KEYMAP_CONTEXT = f"{KEY_CONTEXT} && !{TEXT_INPUT_KEY_CONTEXT}"

PLACEMENTS = {
    "Account",
    "AssetBrowser",
    "DesktopIntegration",
    "ExecutionDock",
    "ExtensionHost",
    "GraphWorkspace",
    "HelpCenter",
    "MediaEditor",
    "NodeLibrary",
    "Settings",
    "WorkspaceShell",
}

AVAILABILITY_VARIANTS = {
    "active": "Active",
    "conditional": "Conditional",
    "cloud/paid": "CloudPaid",
    "infrastructure-only": "InfrastructureOnly",
    "experimental": "Experimental",
    "platform-specific": "PlatformSpecific",
    "deprecated/dead": "Deprecated",
    "developer-only": "DeveloperOnly",
}

ITEM_KIND_VARIANTS = {
    "action": "Action",
    "checkbox-action": "CheckboxAction",
    "command-action": "CommandAction",
    "coverage-classification": "CoverageClassification",
    "destructive-action": "DestructiveAction",
    "developer-action": "DeveloperAction",
    "disabled-item": "DisabledItem",
    "dynamic-action": "DynamicAction",
    "dynamic-submenu": "DynamicSubmenu",
    "extension-hook": "ExtensionHook",
    "infrastructure": "Infrastructure",
    "infrastructure-consumer": "InfrastructureConsumer",
    "infrastructure-label": "InfrastructureLabel",
    "infrastructure-submenu": "InfrastructureSubmenu",
    "navigation-action": "NavigationAction",
    "radio-action": "RadioAction",
    "submenu": "Submenu",
    "submenu-action": "SubmenuAction",
    "toggle-action": "ToggleAction",
}

GRAPH_ACTION_COMMANDS = {
    "Comfy.Canvas.CopySelected",
    "Comfy.Canvas.DeleteSelectedItems",
    "Comfy.Canvas.FitView",
    "Comfy.Canvas.Lock",
    "Comfy.Canvas.MoveSelectedNodes.Down",
    "Comfy.Canvas.MoveSelectedNodes.Left",
    "Comfy.Canvas.MoveSelectedNodes.Right",
    "Comfy.Canvas.MoveSelectedNodes.Up",
    "Comfy.Canvas.PasteFromClipboard",
    "Comfy.Canvas.PasteFromClipboardWithConnect",
    "Comfy.Canvas.ResetView",
    "Comfy.Canvas.Resize",
    "Comfy.Canvas.SelectAll",
    "Comfy.Canvas.ToggleLinkVisibility",
    "Comfy.Canvas.ToggleLock",
    "Comfy.Canvas.ToggleMinimap",
    "Comfy.Canvas.ToggleSelectedNodes.Bypass",
    "Comfy.Canvas.ToggleSelectedNodes.Collapse",
    "Comfy.Canvas.ToggleSelectedNodes.Mute",
    "Comfy.Canvas.ToggleSelectedNodes.Pin",
    "Comfy.Canvas.ToggleSelected.Pin",
    "Comfy.Canvas.Unlock",
    "Comfy.Canvas.ZoomIn",
    "Comfy.Canvas.ZoomOut",
    "Comfy.Graph.ConvertToSubgraph",
    "Comfy.Graph.EditSubgraphWidgets",
    "Comfy.Graph.ExitSubgraph",
    "Comfy.Graph.FitGroupToContents",
    "Comfy.Graph.GroupSelectedNodes",
    "Comfy.Graph.ToggleWidgetPromotion",
    "Comfy.Graph.UnpackSubgraph",
    "Comfy.PublishSubgraph",
    "Comfy.RefreshNodeDefinitions",
    "Comfy.Subgraph.SetDescription",
    "Comfy.Subgraph.SetSearchAliases",
    "Comfy.ToggleCanvasInfo",
    "Experimental.ToggleVueNodes",
}

COMMAND_INPUT_REQUIREMENTS = {
    "Comfy.Graph.ConvertToSubgraph": "subgraph name",
    "Comfy.Graph.EditSubgraphWidgets": "hovered widget and exposure state",
    "Comfy.Graph.ToggleWidgetPromotion": "hovered widget and exposure state",
    "Comfy.RefreshNodeDefinitions": "resolved node definition",
    "Comfy.Subgraph.SetDescription": "subgraph definition and description",
    "Comfy.Subgraph.SetSearchAliases": "subgraph definition and search aliases",
}

NATIVE_COMMAND_ACTIONS = {
    "Comfy.Undo": "GraphUndo",
    "Comfy.Redo": "GraphRedo",
    "Comfy.Canvas.CopySelected": "GraphCopy",
    "Comfy.Canvas.DeleteSelectedItems": "GraphDelete",
    "Comfy.Canvas.FitView": "GraphFitView",
    "Comfy.Canvas.Lock": "LockCanvas",
    "Comfy.Canvas.MoveSelectedNodes.Down": "MoveSelectedDown",
    "Comfy.Canvas.MoveSelectedNodes.Left": "MoveSelectedLeft",
    "Comfy.Canvas.MoveSelectedNodes.Right": "MoveSelectedRight",
    "Comfy.Canvas.MoveSelectedNodes.Up": "MoveSelectedUp",
    "Comfy.Canvas.PasteFromClipboard": "GraphPaste",
    "Comfy.Canvas.PasteFromClipboardWithConnect": "PasteWithConnect",
    "Comfy.Canvas.ResetView": "ResetView",
    "Comfy.Canvas.Resize": "ResizeSelectedNodes",
    "Comfy.Canvas.SelectAll": "GraphSelectAll",
    "Comfy.Canvas.ToggleLinkVisibility": "ToggleLinkVisibility",
    "Comfy.Canvas.ToggleLock": "ToggleCanvasLock",
    "Comfy.Canvas.ToggleMinimap": "ToggleMinimap",
    "Comfy.Canvas.ToggleSelectedNodes.Bypass": "ToggleSelectedBypass",
    "Comfy.Canvas.ToggleSelectedNodes.Collapse": "ToggleSelectedCollapse",
    "Comfy.Canvas.ToggleSelectedNodes.Mute": "ToggleSelectedMute",
    "Comfy.Canvas.ToggleSelectedNodes.Pin": "ToggleSelectedNodesPin",
    "Comfy.Canvas.ToggleSelected.Pin": "ToggleSelectedItemsPin",
    "Comfy.Canvas.Unlock": "UnlockCanvas",
    "Comfy.Canvas.ZoomIn": "GraphZoomIn",
    "Comfy.Canvas.ZoomOut": "GraphZoomOut",
    "Comfy.ClearPendingTasks": "ClearPendingTasks",
    "Comfy.Interrupt": "Interrupt",
    "Comfy.QueuePrompt": "QueuePrompt",
    "Comfy.QueuePromptFront": "QueuePromptFront",
    "Comfy.QueueSelectedOutputNodes": "QueueSelectedOutputNodes",
    "Comfy.Queue.ToggleOverlay": "ToggleQueueOverlay",
    "Comfy.ToggleQPOV2": "ToggleQpov2",
    "Comfy.Graph.ExitSubgraph": "ExitSubgraph",
    "Comfy.Graph.FitGroupToContents": "FitGroupToContents",
    "Comfy.Graph.GroupSelectedNodes": "GroupSelectedNodes",
    "Comfy.Graph.UnpackSubgraph": "UnpackSubgraph",
    "Comfy.PublishSubgraph": "PublishSubgraph",
    "Comfy.RefreshNodeDefinitions": "RefreshNodeDefinitions",
    "Comfy.Graph.ConvertToSubgraph": "ConvertToSubgraph",
}

NATIVE_EXECUTABLE_COMMANDS = set(NATIVE_COMMAND_ACTIONS) - set(COMMAND_INPUT_REQUIREMENTS)
NATIVE_EXECUTABLE_COMMANDS.add("Comfy.SaveWorkflow")

NATIVE_COMMAND_ACTIONS.update(
    {
        "Workspace.ToggleSidebarTab.workflows": "ToggleWorkflowsSidebar",
        "Workspace.ToggleSidebarTab.node-library": "ToggleNodeLibrarySidebar",
        "Workspace.ToggleSidebarTab.model-library": "ToggleModelLibrarySidebar",
        "Workspace.ToggleSidebarTab.assets": "ToggleAssetsSidebar",
        "Comfy.ToggleLinear": "ToggleLinear",
        "Comfy.SaveWorkflow": "SaveWorkflow",
        "Comfy.OpenWorkflow": "OpenWorkflow",
        "Comfy.ShowSettingsDialog": "ShowSettings",
        "Workspace.ToggleBottomPanel.Shortcuts": "ShowKeybindings",
        "Workspace.ToggleBottomPanelTab.logs-terminal": "ToggleLogsPanel",
    }
)

NATIVE_MENU_ACTIONS = {
    "COMFY-MENU-006": "CopySelectedExecutionId",
    "COMFY-MENU-008": "CopySelectedExecutionError",
    "COMFY-MENU-010": "RemoveSelectedExecution",
    "COMFY-MENU-011": "CancelSelectedExecution",
    "COMFY-MENU-057": "ToggleDockedExecutionHistory",
    "COMFY-MENU-058": "ToggleExecutionProgress",
    "COMFY-MENU-059": "ClearExecutionHistory",
    "COMFY-MENU-060": "ExecutionRunManual",
    "COMFY-MENU-061": "ExecutionRunOnChange",
    "COMFY-MENU-062": "ExecutionRunInstantIdle",
}

# This is the generation-time semantic adapter for Task 24's exact source rows. The generated
# Rust catalog is the only production registry; context_menu.rs consumes these typed fields and
# never repeats feature identifiers or source-menu ownership decisions.
GRAPH_CONTEXT_MENU_ACTIONS = {
    "COMFY-GRAPH-124": ("AdjustNodeSize", "Node"),
    "COMFY-GRAPH-125": ("ToggleNodeCollapse", "Node"),
    "COMFY-GRAPH-126": ("ChooseNodeShape", "Node"),
    "COMFY-GRAPH-127": ("ChooseNodeColor", "Node"),
    "COMFY-GRAPH-128": ("ToggleNodePin", "Node"),
    "COMFY-GRAPH-129": ("ToggleNodeBypass", "Node"),
    "COMFY-GRAPH-132": ("RenameSelection", "Selection"),
    "COMFY-GRAPH-133": ("CopySelection", "Selection"),
    "COMFY-GRAPH-134": ("DuplicateSelection", "Selection"),
    "COMFY-GRAPH-135": ("ConvertToSubgraph", "Selection"),
    "COMFY-GRAPH-136": ("PublishSubgraph", "Selection"),
    "COMFY-GRAPH-137": ("UnpackSubgraph", "Selection"),
    "COMFY-GRAPH-138": ("FrameSelection", "Selection"),
    "COMFY-GRAPH-139": ("AlignSelection", "Selection"),
    "COMFY-GRAPH-140": ("DistributeSelection", "Selection"),
    "COMFY-GRAPH-141": ("DeleteSelection", "Selection"),
    "COMFY-GRAPH-142": ("FitGroup", "Group"),
    "COMFY-GRAPH-143": ("ChooseGroupNodeShape", "Group"),
    "COMFY-GRAPH-144": ("ChooseGroupColor", "Group"),
    "COMFY-GRAPH-145": ("ChooseGroupMode", "Group"),
    "COMFY-MENU-063": ("CanvasSelectMode", "CanvasMode"),
    "COMFY-MENU-064": ("CanvasHandMode", "CanvasMode"),
    "COMFY-MENU-080": ("ToggleNodesTwo", "Canvas"),
    "COMFY-MENU-115": ("AddGroup", "Canvas"),
    "COMFY-MENU-116": ("Paste", "Canvas"),
    "COMFY-MENU-117": ("ConvertToSubgraph", "Selection"),
    "COMFY-MENU-118": ("AlignSelection", "Selection"),
    "COMFY-MENU-119": ("ConvertToSubgraph", "Node"),
    "COMFY-MENU-120": ("OpenNodeProperties", "Node"),
    "COMFY-MENU-121": ("OpenNodePropertiesPanel", "Node"),
    "COMFY-MENU-122": ("RenameNode", "Node"),
    "COMFY-MENU-123": ("ChooseNodeMode", "Node"),
    "COMFY-MENU-124": ("AdjustNodeSize", "Node"),
    "COMFY-MENU-125": ("ToggleNodeCollapse", "Node"),
    "COMFY-MENU-126": ("ToggleAdvancedWidgets", "Node"),
    "COMFY-MENU-127": ("ToggleNodePin", "Node"),
    "COMFY-MENU-128": ("ChooseNodeColor", "Node"),
    "COMFY-MENU-129": ("ChooseNodeShape", "Node"),
    "COMFY-MENU-130": ("DuplicateSelection", "Node"),
    "COMFY-MENU-131": ("AlignSelection", "Node"),
    "COMFY-MENU-132": ("DistributeSelection", "Node"),
    "COMFY-MENU-133": ("DeleteSelection", "Node"),
    "COMFY-MENU-134": ("ToggleGroupPin", "Group"),
    "COMFY-MENU-135": ("RenameGroup", "Group"),
    "COMFY-MENU-136": ("ChooseGroupColor", "Group"),
    "COMFY-MENU-137": ("ChooseGroupFontSize", "Group"),
    "COMFY-MENU-138": ("DeleteGroup", "Group"),
    "COMFY-MENU-139": ("DisconnectSlot", "Slot"),
    "COMFY-MENU-140": ("RenameSlot", "Slot"),
    "COMFY-MENU-141": ("DeleteSlot", "Slot"),
    "COMFY-MENU-142": ("ToggleRerouteType", "Reroute"),
    "COMFY-MENU-143": ("ToggleDefaultRerouteType", "Reroute"),
    "COMFY-MENU-152": ("AddGroupForSelection", "Selection"),
    "COMFY-MENU-153": ("AddSelectionToGroup", "Group"),
    "COMFY-MENU-154": ("SelectGroupNodes", "Group"),
}

GRAPH_CONTEXT_MENU_INFRASTRUCTURE = {
    "COMFY-MENU-155": "RegisterMenuGroup",
    "COMFY-MENU-156": "CommandAdapter",
    "COMFY-MENU-157": "CoreMenuLoader",
    "COMFY-MENU-159": "TranslatedRegistryItems",
    "COMFY-MENU-160": "DropdownRenderer",
    "COMFY-MENU-164": "ContextMenuConverter",
    "COMFY-MENU-165": "MergedMoreOptions",
    "COMFY-MENU-166": "NativeContextRenderer",
}

COMPONENT_SURFACE_DISPOSITIONS = {
    "application-ui": ("place", "WorkspaceShellItem", WORKFLOW_EXPERIENCE),
    "graph-editor": ("place", "GraphWorkspaceItem", ASSET_VIEWERS),
    "asset-viewer-editor": ("place", "AssetWorkspaceItem", ASSET_VIEWERS),
    "workflow-experience": ("place", "WorkflowWorkspaceItem", WORKFLOW_EXPERIENCE),
    "cloud-account-workspace": ("place", "AccountPopover", AUTH_CLOUD),
    "settings": ("place", "SettingsWorkspaceItem", SETTINGS_UI),
    "frontend-extension-manager": ("place", "ExtensionCompatibilityHost", EXTENSION_UI),
    "queue-execution-ui": ("place", "ExecutionDockPanel", EXECUTION_UI),
    "desktop-native-ui": ("place", "DesktopCompatibilityHost", DESKTOP_UI),
    "desktop-diagnostics": ("place", "DiagnosticsDockPanel", DIAGNOSTICS),
    "desktop-installation": ("place", "DesktopMigrationModal", "comfy-parity-desktop-installations"),
    "desktop-update": ("place", "UpdateStatusSurface", UPDATES),
    "website-cloud": ("defer", "ServiceContractAndAuthorization", AUTH_CLOUD),
}

COMPONENT_SURFACE_OVERRIDES = {
    "COMFY-FRONTEND-SURFACE-922B12C3CA3D": ("place", "ExecutionDockPanel", EXECUTION_UI),
    "COMFY-FRONTEND-SURFACE-F7223A6667BB": ("place", "GraphWorkspaceItem", EXECUTION_UI),
    "COMFY-FRONTEND-SURFACE-A14F4CA91E43": ("place", "ExecutionDockPanel", EXECUTION_UI),
    "COMFY-FRONTEND-SURFACE-E721F4A4F9B9": ("place", "ExecutionDockPanel", EXECUTION_UI),
    "COMFY-FRONTEND-SURFACE-97D04E89D68E": ("place", "ExecutionDockPanel", EXECUTION_UI),
    "COMFY-FRONTEND-SURFACE-F69CDE266EDA": ("place", "ExecutionDockPanel", EXECUTION_UI),
    "COMFY-FRONTEND-SURFACE-6F5EE356A779": ("place", "ExtensionCompatibilityHost", EXTENSION_UI),
}


def feature_ids(prefix: str, *numbers: int) -> set[str]:
    return {f"COMFY-{prefix}-{number:03d}" for number in numbers}


def inclusive(start: int, end: int) -> tuple[int, ...]:
    return tuple(range(start, end + 1))


def build_ledger() -> dict[str, tuple[str, str, str]]:
    ledger: dict[str, tuple[str, str, str]] = {}

    def assign(
        placement: str,
        owner: str,
        rationale: str,
        identifiers: set[str],
    ) -> None:
        if placement not in PLACEMENTS:
            raise RuntimeError(f"unknown native placement {placement}")
        duplicates = sorted(identifiers.intersection(ledger))
        if duplicates:
            raise RuntimeError(f"duplicate menu dispositions: {duplicates}")
        for identifier in identifiers:
            ledger[identifier] = (placement, owner, rationale)

    assign(
        "WorkspaceShell",
        WORKFLOW_EXPERIENCE,
        "workflow lifecycle, template, tab, and App Builder surface",
        feature_ids("UI", *inclusive(81, 86), 92)
        | feature_ids("WORKFLOW", *inclusive(86, 98))
        | feature_ids("QUEUE", 40)
        | feature_ids("MENU", 4, 5, *inclusive(42, 44), 81, *inclusive(161, 163)),
    )
    assign(
        "GraphWorkspace",
        GRAPH_SHELL,
        "registered command-backed graph shell action",
        feature_ids("UI", *inclusive(87, 91)),
    )
    assign(
        "GraphWorkspace",
        NATIVE_REGISTRY,
        "native node-definition registry refresh surface",
        feature_ids("UI", 94),
    )
    assign(
        "GraphWorkspace",
        GRAPH_CONTEXT_MENUS,
        "native graph, selection, node, group, reroute, and subgraph context menu",
        feature_ids("GRAPH", *inclusive(124, 129), *inclusive(132, 145))
        | feature_ids(
            "MENU",
            63,
            64,
            80,
            *inclusive(115, 143),
            *inclusive(152, 154),
            *inclusive(164, 166),
        ),
    )
    assign(
        "WorkspaceShell",
        GRAPH_CONTEXT_MENUS,
        "graph context-menu registry and canvas extension infrastructure",
        feature_ids("MENU", *inclusive(155, 157), 159, 160),
    )
    assign(
        "GraphWorkspace",
        EXECUTION_UI,
        "selected graph branch execution action",
        feature_ids("GRAPH", 130),
    )
    assign(
        "ExecutionDock",
        EXECUTION_UI,
        "native job, run-mode, and history action",
        feature_ids("MENU", 6, 8, 10, 11, *inclusive(57, 62)),
    )
    assign(
        "GraphWorkspace",
        ASSET_VIEWERS,
        "node creation, node information, and node-template surface",
        feature_ids("GRAPH", 131) | feature_ids("MENU", 114, *inclusive(144, 146)),
    )
    assign(
        "AssetBrowser",
        ASSET_VIEWERS,
        "asset browsing, filtering, settings, and media context surface",
        feature_ids("MENU", *inclusive(1, 3), 7, *inclusive(12, 24), *inclusive(65, 75)),
    )
    assign(
        "NodeLibrary",
        ASSET_VIEWERS,
        "node library and tree-folder context surface",
        feature_ids("MENU", *inclusive(76, 78), 170),
    )
    assign(
        "ExtensionHost",
        ASSET_VIEWERS,
        "extension manager and custom-node discovery surface",
        feature_ids("MENU", 50, 83),
    )
    assign(
        "MediaEditor",
        IMAGE_MASK,
        "image, mask, crop, and clipspace surface",
        feature_ids("UI", 93) | feature_ids("ASSET", *inclusive(64, 68)),
    )
    assign(
        "ExecutionDock",
        MEMORY_PLANNER,
        "native model and execution-cache memory action",
        feature_ids("UI", 95, 96),
    )
    assign(
        "HelpCenter",
        SETTINGS_UI,
        "help, documentation, public navigation, and discoverability surface",
        feature_ids("UI", *inclusive(97, 101))
        | feature_ids(
            "MENU",
            *inclusive(47, 49),
            53,
            84,
            85,
            89,
            *inclusive(92, 94),
            *inclusive(96, 111),
            113,
            171,
            172,
        ),
    )
    assign(
        "Settings",
        SETTINGS_UI,
        "keybinding, theme, and settings surface",
        feature_ids("MENU", *inclusive(33, 41), 79, 82),
    )
    assign(
        "HelpCenter",
        DIAGNOSTICS,
        "support, feedback, and diagnostic surface",
        feature_ids("UI", 102) | feature_ids("MENU", 45, 46),
    )
    assign(
        "WorkspaceShell",
        DIAGNOSTICS,
        "logs and terminal surface",
        feature_ids("MENU", 9),
    )
    assign(
        "Account",
        AUTH_CLOUD,
        "profile, workspace membership, and authenticated cloud surface",
        feature_ids("MENU", *inclusive(25, 32), *inclusive(86, 88), 95),
    )
    assign(
        "ExtensionHost",
        UPDATES,
        "extension update surface",
        feature_ids("MENU", 51),
    )
    assign(
        "DesktopIntegration",
        UPDATES,
        "desktop application update surface",
        feature_ids("MENU", 55),
    )
    assign(
        "HelpCenter",
        UPDATES,
        "release and update information surface",
        feature_ids("MENU", 56),
    )
    assign(
        "HelpCenter",
        DESKTOP_UI,
        "desktop troubleshooting navigation surface",
        feature_ids("MENU", 52),
    )
    assign(
        "DesktopIntegration",
        DESKTOP_UI,
        "desktop menu and platform coverage surface",
        feature_ids("MENU", 54, 173),
    )
    assign(
        "HelpCenter",
        COMPATIBILITY,
        "inactive legacy website navigation surface",
        feature_ids("MENU", 90, 91, 112),
    )
    assign(
        "MediaEditor",
        THREE_D_CONTENT,
        "native 3D node and viewer surface",
        feature_ids("MENU", *inclusive(147, 151)),
    )
    assign(
        "ExtensionHost",
        EXTENSION_UI,
        "versioned extension menu hook and registry surface",
        feature_ids("MENU", 158, *inclusive(167, 169)),
    )
    return ledger


def rust_string(value: str) -> str:
    for legacy, canonical in (("Comfy.", "Sim."), ("Comfy-Desktop.", "Sim-Desktop."),
                              ("comfy_shell::", "sim_shell::"), ("comfy_graph::", "sim_graph::")):
        if value.startswith(legacy):
            value = canonical + value[len(legacy):]
            break
    if value == "Comfy":
        value = "Sim"
    return json.dumps(value, ensure_ascii=False)


def native_action_names() -> dict[str, str]:
    source = ACTIONS_SOURCE.read_text(encoding="utf-8")
    names = dict(
        re.findall(
            r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*=>\s*\("([^"]+)",',
            source,
            flags=re.MULTILINE,
        )
    )
    if not names:
        raise RuntimeError("actions.rs contains no NativeAction name mappings")
    return names


def keymap_keystroke(combo: str) -> str:
    return combo.lower().replace(" + ", "-")


def command_status_literal(
    availability: str,
    owner: str,
    input_requirement: str,
    has_native_action: bool,
) -> str:
    owner_literal = rust_string(owner)
    if availability == "deprecated/dead":
        return f"CommandNativeStatus::Legacy {{ owner: {owner_literal} }}"
    if availability != "active":
        gates = {
            "platform-specific": "platform-specific",
            "experimental": "experimental",
            "developer-only": "developer-only",
            "cloud/paid": "cloud-or-paid-service",
        }
        try:
            gate = gates[availability]
        except KeyError as error:
            raise RuntimeError(f"unknown gated command availability {availability!r}") from error
        return (
            "CommandNativeStatus::Gated { "
            f"owner: {owner_literal}, gate: {rust_string(gate)}"
            " }"
        )
    if input_requirement:
        return (
            "CommandNativeStatus::RequiresInput { "
            f"owner: {owner_literal}, input: {rust_string(input_requirement)}"
            " }"
        )
    if has_native_action:
        return "CommandNativeStatus::Executable"
    return f"CommandNativeStatus::LaterOwned {{ owner: {owner_literal} }}"


def command_placement_and_owner(command_id: str) -> tuple[str, str]:
    if command_id in GRAPH_ACTION_COMMANDS:
        if command_id == "Comfy.RefreshNodeDefinitions":
            return "GraphWorkspace", NATIVE_REGISTRY
        if command_id in COMMAND_INPUT_REQUIREMENTS:
            return "GraphWorkspace", GRAPH_CONTEXT_MENUS
        return "GraphWorkspace", GRAPH_SHELL
    if command_id in {"Comfy.Undo", "Comfy.Redo"}:
        return "GraphWorkspace", GRAPH_SHELL
    if command_id.startswith("Comfy-Desktop."):
        owner = (
            UPDATES
            if command_id in {"Comfy-Desktop.CheckForUpdates", "Comfy-Desktop.Reinstall"}
            else DESKTOP_UI
        )
        return "DesktopIntegration", owner
    if command_id.startswith("Comfy.Queue") or command_id in {
        "Comfy.Interrupt",
        "Comfy.ClearPendingTasks",
        "Comfy.ToggleQPOV2",
    }:
        return "ExecutionDock", EXECUTION_UI
    if command_id.startswith("Comfy.Memory."):
        return "ExecutionDock", MEMORY_PLANNER
    if command_id.startswith("Comfy.MaskEditor.") or command_id == "Comfy.OpenClipspace":
        return "MediaEditor", IMAGE_MASK
    if command_id.startswith("Comfy.3DViewer."):
        return "MediaEditor", THREE_D_CONTENT
    if "Manager" in command_id:
        if command_id == "Comfy.Manager.ShowUpdateAvailablePacks":
            return "ExtensionHost", UPDATES
        owner = (
            COMPATIBILITY
            if command_id
            in {
                "Comfy.Manager.CustomNodesManager.ShowLegacyCustomNodesMenu",
                "Comfy.Manager.ShowLegacyManagerMenu",
            }
            else ASSET_VIEWERS
        )
        return "ExtensionHost", owner
    if "Settings" in command_id or command_id == "Comfy.ToggleTheme":
        return "Settings", SETTINGS_UI
    if command_id.startswith("Comfy.Help.") or command_id == "Comfy.ToggleHelpCenter":
        return "HelpCenter", SETTINGS_UI
    if command_id.startswith("Comfy.User."):
        return "Account", AUTH_CLOUD
    if command_id in {"Comfy.BrowseModelAssets", "Comfy.ToggleAssetAPI"}:
        return "AssetBrowser", ASSET_VIEWERS
    if command_id == "Comfy.Dev.ShowModelSelector":
        return "NodeLibrary", ASSET_VIEWERS
    if command_id in {
        "Comfy.BrowseTemplates",
        "Comfy.ToggleLinear",
        "Workspace.CloseWorkflow",
        "Workspace.NextOpenedWorkflow",
        "Workspace.PreviousOpenedWorkflow",
        "Workspace.ToggleFocusMode",
        "Workspace.ToggleSidebarTab.workflows",
        "Comfy.NewBlankWorkflow",
        "Comfy.ClearWorkflow",
        "Comfy.DuplicateWorkflow",
        "Comfy.ExportWorkflow",
        "Comfy.ExportWorkflowAPI",
        "Comfy.LoadDefaultWorkflow",
        "Comfy.OpenWorkflow",
        "Comfy.RenameWorkflow",
        "Comfy.SaveWorkflow",
        "Comfy.SaveWorkflowAs",
    }:
        return "WorkspaceShell", WORKFLOW_EXPERIENCE
    if command_id in {
        "Workspace.SearchBox.Toggle",
        "Workspace.ToggleSidebarTab.apps",
        "Workspace.ToggleSidebarTab.assets",
        "Workspace.ToggleSidebarTab.model-library",
        "Workspace.ToggleSidebarTab.node-library",
    }:
        return "AssetBrowser", ASSET_VIEWERS
    if command_id in {
        "Workspace.ToggleBottomPanelTab.shortcuts-essentials",
        "Workspace.ToggleBottomPanelTab.shortcuts-view-controls",
        "Workspace.ToggleBottomPanel.Shortcuts",
    }:
        return "Settings", SETTINGS_UI
    if command_id.startswith("Workspace.ToggleBottomPanel"):
        return "WorkspaceShell", DIAGNOSTICS
    if command_id == "Comfy.ContactSupport":
        return "HelpCenter", DIAGNOSTICS
    if command_id.startswith("Workspace."):
        return "WorkspaceShell", WORKFLOW_EXPERIENCE
    return "WorkspaceShell", DESKTOP_UI


def generate_command_catalog() -> dict[str, dict[str, str]]:
    with COMMAND_SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 118 or len({row["command_id"] for row in source_rows}) != 118:
        raise RuntimeError("frontend command catalog must contain 118 unique commands")
    output_rows: list[dict[str, str]] = []
    rust_rows: list[str] = []
    for row in source_rows:
        command_id = row["command_id"]
        placement, owner = command_placement_and_owner(command_id)
        native_action = NATIVE_COMMAND_ACTIONS.get(command_id, "")
        input_requirement = COMMAND_INPUT_REQUIREMENTS.get(command_id, "")
        if row["availability"] == "deprecated/dead":
            disposition = "legacy"
        elif row["availability"] != "active":
            disposition = "gated"
        elif input_requirement:
            disposition = "requires-input"
        elif command_id in NATIVE_EXECUTABLE_COMMANDS:
            disposition = "executable"
        else:
            disposition = "later-owned"
        output = {
            "feature_id": row["feature_id"],
            "command_id": command_id,
            "placement": placement,
            "owner_task_id": owner,
            "disposition": disposition,
            "native_action": native_action,
            "input_requirement": input_requirement,
        }
        output_rows.append(output)
        availability_variant = AVAILABILITY_VARIANTS.get(row["availability"])
        if availability_variant is None:
            raise RuntimeError(
                f"unknown command availability {row['availability']!r} for {command_id}"
            )
        rust_rows.extend(
            [
                "    GeneratedCommandCatalogRow {",
                f"        feature_id: {rust_string(row['feature_id'])},",
                f"        command_id: {rust_string(command_id)},",
                f"        label: {rust_string(row['label'])},",
                f"        availability: CommandAvailability::{availability_variant},",
                f"        placement: NativePlacement::{placement},",
                f"        owner: {rust_string(owner)},",
                f"        status: {command_status_literal(row['availability'], owner, input_requirement, command_id in NATIVE_EXECUTABLE_COMMANDS)},",
                (
                    f"        native_action: Some(NativeAction::{native_action}),"
                    if native_action
                    else "        native_action: None,"
                ),
                "    },",
            ]
        )
    with COMMAND_OWNERSHIP_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    rust_source = [
        "use crate::actions::{CommandAvailability, CommandNativeStatus, NativeAction, NativePlacement};",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub struct GeneratedCommandCatalogRow {",
        "    pub feature_id: &'static str,",
        "    pub command_id: &'static str,",
        "    pub label: &'static str,",
        "    pub availability: CommandAvailability,",
        "    pub placement: NativePlacement,",
        "    pub owner: &'static str,",
        "    pub status: CommandNativeStatus,",
        "    pub native_action: Option<NativeAction>,",
        "}",
        "",
        f"pub static GENERATED_COMMAND_CATALOG: [GeneratedCommandCatalogRow; {len(output_rows)}] = [",
        *rust_rows,
        "];",
        "",
    ]
    COMMAND_RUST_OUTPUT.write_text("\n".join(rust_source), encoding="utf-8")
    return {row["command_id"]: row for row in output_rows}


def generate_keybinding_catalog(command_rows: dict[str, dict[str, str]]) -> None:
    with KEYBINDING_SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 34 or len({row["feature_id"] for row in source_rows}) != 34:
        raise RuntimeError("frontend keybinding catalog must contain 34 unique rows")
    action_names = native_action_names()
    bindings: dict[str, str] = {}
    rust_rows: list[str] = []
    for row in source_rows:
        if row["availability"] != "active":
            raise RuntimeError(
                f"keybinding {row['feature_id']} is not active: {row['availability']}"
            )
        command = command_rows.get(row["command_id"])
        if command is None:
            raise RuntimeError(
                f"keybinding {row['feature_id']} references unknown command {row['command_id']}"
            )
        native_action = command["native_action"]
        if not native_action:
            raise RuntimeError(
                f"keybinding {row['feature_id']} has no typed NativeAction adapter"
            )
        try:
            action_name = action_names[native_action]
        except KeyError as error:
            raise RuntimeError(
                f"NativeAction::{native_action} has no canonical GPUI action name"
            ) from error
        keystroke = keymap_keystroke(row["combo"])
        if keystroke in bindings:
            raise RuntimeError(f"duplicate generated Comfy keybinding {keystroke}")
        bindings[keystroke] = action_name
        rust_rows.extend(
            [
                "    GeneratedKeybindingCatalogRow {",
                f"        feature_id: {rust_string(row['feature_id'])},",
                f"        source_combo: {rust_string(row['combo'])},",
                f"        keystroke: {rust_string(keystroke)},",
                f"        command_id: {rust_string(row['command_id'])},",
                f"        native_action: NativeAction::{native_action},",
                "        availability: CommandAvailability::Active,",
                "    },",
            ]
        )
    keymap = [{"context": KEYMAP_CONTEXT, "bindings": bindings}]
    KEYMAP_OUTPUT.write_text(
        json.dumps(keymap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rust_source = [
        "use crate::actions::{CommandAvailability, NativeAction};",
        "",
        f"pub const GENERATED_KEY_CONTEXT: &str = {rust_string(KEY_CONTEXT)};",
        f"pub const GENERATED_TEXT_INPUT_KEY_CONTEXT: &str = {rust_string(TEXT_INPUT_KEY_CONTEXT)};",
        f"pub const GENERATED_KEYMAP_CONTEXT: &str = {rust_string(KEYMAP_CONTEXT)};",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]",
        "pub struct GeneratedKeybindingCatalogRow {",
        "    pub feature_id: &'static str,",
        "    pub source_combo: &'static str,",
        "    pub keystroke: &'static str,",
        "    pub command_id: &'static str,",
        "    pub native_action: NativeAction,",
        "    pub availability: CommandAvailability,",
        "}",
        "",
        f"pub static GENERATED_KEYBINDING_CATALOG: [GeneratedKeybindingCatalogRow; {len(source_rows)}] = [",
        *rust_rows,
        "];",
        "",
    ]
    KEYBINDING_RUST_OUTPUT.write_text("\n".join(rust_source), encoding="utf-8")


def generate_component_catalog() -> list[str]:
    with COMPONENT_SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 805 or len({row["feature_id"] for row in source_rows}) != 805:
        raise RuntimeError("frontend component catalog must contain 805 unique rows")
    output_rows: list[dict[str, str]] = []
    rust_rows: list[str] = []
    for row in source_rows:
        try:
            kind, placement, owner = COMPONENT_SURFACE_OVERRIDES.get(
                row["feature_id"], COMPONENT_SURFACE_DISPOSITIONS[row["domain"]]
            )
        except KeyError as error:
            raise RuntimeError(
                f"component {row['feature_id']} has unknown domain {row['domain']!r}"
            ) from error
        output_rows.append(
            {
                "feature_id": row["feature_id"],
                "disposition": kind,
                "placement": placement,
                "owner_task_id": owner,
            }
        )
        kind_variant = "Place" if kind == "place" else "Defer"
        rust_rows.extend(
            [
                "    GeneratedComponentCatalogRow {",
                f"        feature_id: {rust_string(row['feature_id'])},",
                f"        domain: {rust_string(row['domain'])},",
                f"        disposition: GeneratedComponentDisposition::{kind_variant},",
                f"        placement: GeneratedComponentPlacement::{placement},",
                f"        owner: {rust_string(owner)},",
                "    },",
            ]
        )
    with COMPONENT_OWNERSHIP_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    return rust_rows


def main() -> None:
    command_rows = generate_command_catalog()
    generate_keybinding_catalog(command_rows)
    component_rust_rows = generate_component_catalog()
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    ledger = build_ledger()
    source_ids = {row["feature_id"] for row in source_rows}
    if len(source_ids) != len(source_rows):
        raise RuntimeError("frontend menu source contains duplicate feature IDs")
    if source_ids != set(ledger):
        missing = sorted(source_ids - set(ledger))
        extra = sorted(set(ledger) - source_ids)
        raise RuntimeError(f"menu disposition closure failed: missing={missing}, extra={extra}")
    graph_context_owner_ids = {
        feature_id
        for feature_id, (_, owner, _) in ledger.items()
        if owner == GRAPH_CONTEXT_MENUS
    }
    graph_context_binding_ids = set(GRAPH_CONTEXT_MENU_ACTIONS) | set(
        GRAPH_CONTEXT_MENU_INFRASTRUCTURE
    )
    if set(GRAPH_CONTEXT_MENU_ACTIONS) & set(GRAPH_CONTEXT_MENU_INFRASTRUCTURE):
        raise RuntimeError("graph context action and infrastructure bindings overlap")
    if graph_context_owner_ids != graph_context_binding_ids:
        missing = sorted(graph_context_owner_ids - graph_context_binding_ids)
        extra = sorted(graph_context_binding_ids - graph_context_owner_ids)
        raise RuntimeError(
            f"graph context binding closure failed: missing={missing}, extra={extra}"
        )

    output_rows: list[dict[str, str]] = []
    rust_rows: list[str] = []
    for row in source_rows:
        feature_id = row["feature_id"]
        placement, owner, rationale = ledger[feature_id]
        availability = row["availability"]
        availability_variant = AVAILABILITY_VARIANTS.get(availability)
        if availability_variant is None:
            raise RuntimeError(f"unknown menu availability {availability!r} for {feature_id}")
        item_kind_variant = ITEM_KIND_VARIANTS.get(row["item_kind"])
        if item_kind_variant is None:
            raise RuntimeError(
                f"unknown menu item kind {row['item_kind']!r} for {feature_id}"
            )
        command_id = ""
        if row["item_kind"] == "command-action":
            prefix = "execute command "
            target = row["action_or_target"]
            if not target.startswith(prefix):
                raise RuntimeError(f"malformed command target for {feature_id}: {target!r}")
            command_id = target[len(prefix):]
            if command_id != row["item_id"]:
                raise RuntimeError(f"command identity mismatch for {feature_id}")
            command_row = command_rows.get(command_id)
            if command_row is None:
                raise RuntimeError(f"menu {feature_id} references unknown command {command_id}")
            if command_row["placement"] != placement or command_row["owner_task_id"] != owner:
                raise RuntimeError(
                    f"menu {feature_id} command disposition disagrees with {command_id}"
                )
        native_action = (
            NATIVE_MENU_ACTIONS.get(feature_id, "") if availability == "active" else ""
        )
        context_action, context_surface = GRAPH_CONTEXT_MENU_ACTIONS.get(
            feature_id, ("", "")
        )
        context_infrastructure = GRAPH_CONTEXT_MENU_INFRASTRUCTURE.get(feature_id, "")
        if context_infrastructure:
            if availability != "infrastructure-only":
                raise RuntimeError(
                    f"infrastructure binding {feature_id} is not infrastructure-only"
                )
            context_surface = "Infrastructure"
        if availability == "deprecated/dead":
            disposition = "legacy"
        elif context_infrastructure:
            disposition = "infrastructure"
        elif context_action:
            disposition = "native"
        elif availability != "active":
            disposition = "gated-or-deferred"
        elif command_id and command_rows[command_id]["disposition"] == "executable":
            disposition = "canonical-command"
        elif native_action:
            disposition = "native"
        else:
            disposition = "deferred"
        output_rows.append(
            {
                "feature_id": feature_id,
                "placement": placement,
                "owner_task_id": owner,
                "command_id": command_id,
                "disposition": disposition,
                "native_action": native_action,
                "context_action": context_action,
                "context_infrastructure": context_infrastructure,
                "context_surface": context_surface,
                "rationale": rationale,
            }
        )
        command_literal = f"Some({rust_string(command_id)})" if command_id else "None"
        action_literal = (
            f"Some(NativeAction::{native_action})" if native_action else "None"
        )
        context_action_literal = (
            f"Some(GeneratedGraphContextAction::{context_action})"
            if context_action
            else "None"
        )
        context_infrastructure_literal = (
            f"Some(GeneratedGraphContextInfrastructure::{context_infrastructure})"
            if context_infrastructure
            else "None"
        )
        context_surface_literal = (
            f"Some(GeneratedGraphContextSurface::{context_surface})"
            if context_surface
            else "None"
        )
        rust_rows.extend(
            [
                "    GeneratedMenuCatalogRow {",
                f"        feature_id: {rust_string(feature_id)},",
                f"        menu_surface: {rust_string(row['menu_surface'])},",
                f"        label: {rust_string(row['label_or_action'])},",
                f"        source_condition: {rust_string(row['condition'])},",
                f"        item_kind: GeneratedMenuItemKind::{item_kind_variant},",
                f"        availability: GeneratedMenuAvailability::{availability_variant},",
                f"        placement: NativePlacement::{placement},",
                f"        owner: {rust_string(owner)},",
                f"        command_id: {command_literal},",
                f"        native_action: {action_literal},",
                f"        context_action: {context_action_literal},",
                f"        context_infrastructure: {context_infrastructure_literal},",
                f"        context_surface: {context_surface_literal},",
                "    },",
            ]
        )

    graph_context_rows = [
        row for row in output_rows if row["owner_task_id"] == GRAPH_CONTEXT_MENUS
    ]
    graph_context_actions = [row for row in graph_context_rows if row["context_action"]]
    graph_context_infrastructure = [
        row for row in graph_context_rows if row["context_infrastructure"]
    ]
    if (
        len(graph_context_rows) != 63
        or len(graph_context_actions) != 55
        or len(graph_context_infrastructure) != 8
    ):
        raise RuntimeError(
            "graph context registry must contain exactly 63 rows: "
            "55 actions and 8 infrastructure bindings"
        )
    if any(
        bool(row["context_action"]) == bool(row["context_infrastructure"])
        or not row["context_surface"]
        for row in graph_context_rows
    ):
        raise RuntimeError(
            "each graph context row must have one action or infrastructure binding and a surface"
        )

    with OWNERSHIP_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    rust = [
        "use crate::actions::{NativeAction, NativePlacement};",
        "use serde::Serialize;",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]",
        "pub enum GeneratedMenuAvailability {",
        "    Active,",
        "    Conditional,",
        "    CloudPaid,",
        "    InfrastructureOnly,",
        "    Experimental,",
        "    PlatformSpecific,",
        "    Deprecated,",
        "    DeveloperOnly,",
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]",
        "pub enum GeneratedGraphContextAction {",
        *[
            f"    {action},"
            for action in sorted(
                {action for action, _ in GRAPH_CONTEXT_MENU_ACTIONS.values()}
            )
        ],
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]",
        "pub enum GeneratedGraphContextInfrastructure {",
        *[
            f"    {infrastructure},"
            for infrastructure in sorted(set(GRAPH_CONTEXT_MENU_INFRASTRUCTURE.values()))
        ],
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]",
        "pub enum GeneratedGraphContextSurface {",
        *[
            f"    {surface},"
            for surface in sorted(
                {surface for _, surface in GRAPH_CONTEXT_MENU_ACTIONS.values()}
                | {"Infrastructure"}
            )
        ],
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]",
        "pub enum GeneratedMenuItemKind {",
        *[f"    {variant}," for variant in ITEM_KIND_VARIANTS.values()],
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub struct GeneratedMenuCatalogRow {",
        "    pub feature_id: &'static str,",
        "    pub menu_surface: &'static str,",
        "    pub label: &'static str,",
        "    pub source_condition: &'static str,",
        "    pub item_kind: GeneratedMenuItemKind,",
        "    pub availability: GeneratedMenuAvailability,",
        "    pub placement: NativePlacement,",
        "    pub owner: &'static str,",
        "    pub command_id: Option<&'static str>,",
        "    pub native_action: Option<NativeAction>,",
        "    pub context_action: Option<GeneratedGraphContextAction>,",
        "    pub context_infrastructure: Option<GeneratedGraphContextInfrastructure>,",
        "    pub context_surface: Option<GeneratedGraphContextSurface>,",
        "}",
        "",
        f"pub static GENERATED_MENU_CATALOG: [GeneratedMenuCatalogRow; {len(source_rows)}] = [",
        *rust_rows,
        "];",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub enum GeneratedComponentDisposition {",
        "    Place,",
        "    Defer,",
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub enum GeneratedComponentPlacement {",
        *[
            f"    {placement},"
            for placement in sorted(
                {value[1] for value in COMPONENT_SURFACE_DISPOSITIONS.values()}
                | {value[1] for value in COMPONENT_SURFACE_OVERRIDES.values()}
            )
        ],
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub struct GeneratedComponentCatalogRow {",
        "    pub feature_id: &'static str,",
        "    pub domain: &'static str,",
        "    pub disposition: GeneratedComponentDisposition,",
        "    pub placement: GeneratedComponentPlacement,",
        "    pub owner: &'static str,",
        "}",
        "",
        "pub static GENERATED_COMPONENT_CATALOG: [GeneratedComponentCatalogRow; 805] = [",
        *component_rust_rows,
        "];",
        "",
    ]
    RUST_OUTPUT.write_text("\n".join(rust), encoding="utf-8")
    subprocess.run(
        [
            "rustfmt",
            "--edition",
            "2024",
            str(COMMAND_RUST_OUTPUT),
            str(KEYBINDING_RUST_OUTPUT),
            str(RUST_OUTPUT),
        ],
        check=True,
    )
    print(
        f"Generated {len(command_rows)} command, 34 keybinding, {len(output_rows)} menu, "
        "and 805 component dispositions."
    )


if __name__ == "__main__":
    main()
