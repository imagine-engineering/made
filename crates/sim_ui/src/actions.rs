use crate::generated_command_catalog::GENERATED_COMMAND_CATALOG;
use gpui::{MenuItem, actions};
use serde::Serialize;
use sim_runtime::CatalogGraphAction;

actions!(
    sim_shell,
    [
        #[action(deprecated_aliases = ["comfy_shell::QueuePrompt"])]
        QueuePrompt,
        #[action(deprecated_aliases = ["comfy_shell::QueuePromptFront"])]
        QueuePromptFront,
        #[action(deprecated_aliases = ["comfy_shell::QueueSelectedOutputNodes"])]
        QueueSelectedOutputNodes,
        #[action(deprecated_aliases = ["comfy_shell::Interrupt"])]
        Interrupt,
        #[action(deprecated_aliases = ["comfy_shell::ClearPendingTasks"])]
        ClearPendingTasks,
        #[action(deprecated_aliases = ["comfy_shell::ToggleQueueOverlay"])]
        ToggleQueueOverlay,
        #[action(deprecated_aliases = ["comfy_shell::ToggleQpov2"])]
        ToggleQpov2,
        #[action(deprecated_aliases = ["comfy_shell::ToggleExecutionPanel"])]
        ToggleExecutionPanel,
        #[action(deprecated_aliases = ["comfy_shell::ToggleGraphPropertiesPanel"])]
        ToggleGraphPropertiesPanel,
        #[action(deprecated_aliases = ["comfy_shell::ClearExecutionHistory"])]
        ClearExecutionHistory,
        #[action(deprecated_aliases = ["comfy_shell::CancelSelectedExecution"])]
        CancelSelectedExecution,
        #[action(deprecated_aliases = ["comfy_shell::RetrySelectedExecution"])]
        RetrySelectedExecution,
        #[action(deprecated_aliases = ["comfy_shell::RemoveSelectedExecution"])]
        RemoveSelectedExecution,
        #[action(deprecated_aliases = ["comfy_shell::CopySelectedExecutionId"])]
        CopySelectedExecutionId,
        #[action(deprecated_aliases = ["comfy_shell::CopySelectedExecutionError"])]
        CopySelectedExecutionError,
        #[action(deprecated_aliases = ["comfy_shell::ToggleDockedExecutionHistory"])]
        ToggleDockedExecutionHistory,
        #[action(deprecated_aliases = ["comfy_shell::ToggleExecutionProgress"])]
        ToggleExecutionProgress,
        #[action(deprecated_aliases = ["comfy_shell::ExecutionRunManual"])]
        ExecutionRunManual,
        #[action(deprecated_aliases = ["comfy_shell::ExecutionRunOnChange"])]
        ExecutionRunOnChange,
        #[action(deprecated_aliases = ["comfy_shell::ExecutionRunInstantIdle"])]
        ExecutionRunInstantIdle,
        #[action(deprecated_aliases = ["comfy_shell::RestoreExecutionNavigation"])]
        RestoreExecutionNavigation,
        #[action(deprecated_aliases = ["comfy_shell::RefreshNodeDefinitions"])]
        RefreshNodeDefinitions,
        #[action(deprecated_aliases = ["comfy_shell::ToggleWorkflowsSidebar"])]
        ToggleWorkflowsSidebar,
        #[action(deprecated_aliases = ["comfy_shell::ToggleNodeLibrarySidebar"])]
        ToggleNodeLibrarySidebar,
        #[action(deprecated_aliases = ["comfy_shell::ToggleModelLibrarySidebar"])]
        ToggleModelLibrarySidebar,
        #[action(deprecated_aliases = ["comfy_shell::ToggleAssetsSidebar"])]
        ToggleAssetsSidebar,
        #[action(deprecated_aliases = ["comfy_shell::ToggleLinear"])]
        ToggleLinear,
        #[action(deprecated_aliases = ["comfy_shell::SaveWorkflow"])]
        SaveWorkflow,
        #[action(deprecated_aliases = ["comfy_shell::OpenWorkflow"])]
        OpenWorkflow,
        #[action(deprecated_aliases = ["comfy_shell::GroupSelectedNodes"])]
        GroupSelectedNodes,
        #[action(deprecated_aliases = ["comfy_shell::ShowSettings"])]
        ShowSettings,
        #[action(deprecated_aliases = ["comfy_shell::ShowKeybindings"])]
        ShowKeybindings,
        #[action(deprecated_aliases = ["comfy_shell::ToggleSelectedItemsPin"])]
        ToggleSelectedItemsPin,
        #[action(deprecated_aliases = ["comfy_shell::ToggleSelectedCollapse"])]
        ToggleSelectedCollapse,
        #[action(deprecated_aliases = ["comfy_shell::ToggleSelectedBypass"])]
        ToggleSelectedBypass,
        #[action(deprecated_aliases = ["comfy_shell::ToggleSelectedMute"])]
        ToggleSelectedMute,
        #[action(deprecated_aliases = ["comfy_shell::ToggleLogsPanel"])]
        ToggleLogsPanel,
        #[action(deprecated_aliases = ["comfy_shell::ConvertToSubgraph"])]
        ConvertToSubgraph,
        #[action(deprecated_aliases = ["comfy_shell::ToggleMinimap"])]
        ToggleMinimap,
        #[action(deprecated_aliases = ["comfy_shell::UnlockCanvas"])]
        UnlockCanvas,
        #[action(deprecated_aliases = ["comfy_shell::LockCanvas"])]
        LockCanvas,
        #[action(deprecated_aliases = ["comfy_shell::ExitSubgraph"])]
        ExitSubgraph,
        #[action(deprecated_aliases = ["comfy_shell::PasteWithConnect"])]
        PasteWithConnect,
        #[action(deprecated_aliases = ["comfy_shell::MoveSelectedDown"])]
        MoveSelectedDown,
        #[action(deprecated_aliases = ["comfy_shell::MoveSelectedLeft"])]
        MoveSelectedLeft,
        #[action(deprecated_aliases = ["comfy_shell::MoveSelectedRight"])]
        MoveSelectedRight,
        #[action(deprecated_aliases = ["comfy_shell::MoveSelectedUp"])]
        MoveSelectedUp,
        #[action(deprecated_aliases = ["comfy_shell::ResetView"])]
        ResetView,
        #[action(deprecated_aliases = ["comfy_shell::ResizeSelectedNodes"])]
        ResizeSelectedNodes,
        #[action(deprecated_aliases = ["comfy_shell::ToggleLinkVisibility"])]
        ToggleLinkVisibility,
        #[action(deprecated_aliases = ["comfy_shell::ToggleCanvasLock"])]
        ToggleCanvasLock,
        #[action(deprecated_aliases = ["comfy_shell::ToggleSelectedNodesPin"])]
        ToggleSelectedNodesPin,
        #[action(deprecated_aliases = ["comfy_shell::FitGroupToContents"])]
        FitGroupToContents,
        #[action(deprecated_aliases = ["comfy_shell::UnpackSubgraph"])]
        UnpackSubgraph,
        #[action(deprecated_aliases = ["comfy_shell::PublishSubgraph"])]
        PublishSubgraph,
    ]
);

macro_rules! native_actions {
    ($($variant:ident => ($name:literal, $action:path)),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
        #[serde(rename_all = "kebab-case")]
        pub enum NativeAction {
            $($variant),+
        }

        impl NativeAction {
            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name),+
                }
            }

            pub fn menu_item(self, label: impl Into<gpui::SharedString>) -> MenuItem {
                match self {
                    $(Self::$variant => MenuItem::action(label, $action)),+
                }
            }
        }
    };
}

native_actions!(
    GraphUndo => ("sim_graph::GraphUndo", crate::GraphUndo),
    GraphRedo => ("sim_graph::GraphRedo", crate::GraphRedo),
    GraphCopy => ("sim_graph::GraphCopy", crate::GraphCopy),
    GraphDelete => ("sim_graph::GraphDelete", crate::GraphDelete),
    GraphFitView => ("sim_graph::GraphFitView", crate::GraphFitView),
    LockCanvas => ("sim_shell::LockCanvas", crate::LockCanvas),
    MoveSelectedDown => ("sim_shell::MoveSelectedDown", crate::MoveSelectedDown),
    MoveSelectedLeft => ("sim_shell::MoveSelectedLeft", crate::MoveSelectedLeft),
    MoveSelectedRight => ("sim_shell::MoveSelectedRight", crate::MoveSelectedRight),
    MoveSelectedUp => ("sim_shell::MoveSelectedUp", crate::MoveSelectedUp),
    GraphPaste => ("sim_graph::GraphPaste", crate::GraphPaste),
    PasteWithConnect => ("sim_shell::PasteWithConnect", crate::PasteWithConnect),
    ResetView => ("sim_shell::ResetView", crate::ResetView),
    ResizeSelectedNodes => ("sim_shell::ResizeSelectedNodes", crate::ResizeSelectedNodes),
    GraphSelectAll => ("sim_graph::GraphSelectAll", crate::GraphSelectAll),
    ToggleLinkVisibility => ("sim_shell::ToggleLinkVisibility", crate::ToggleLinkVisibility),
    ToggleCanvasLock => ("sim_shell::ToggleCanvasLock", crate::ToggleCanvasLock),
    ToggleMinimap => ("sim_shell::ToggleMinimap", crate::ToggleMinimap),
    ToggleSelectedBypass => ("sim_shell::ToggleSelectedBypass", crate::ToggleSelectedBypass),
    ToggleSelectedCollapse => ("sim_shell::ToggleSelectedCollapse", crate::ToggleSelectedCollapse),
    ToggleSelectedMute => ("sim_shell::ToggleSelectedMute", crate::ToggleSelectedMute),
    ToggleSelectedNodesPin => ("sim_shell::ToggleSelectedNodesPin", crate::ToggleSelectedNodesPin),
    ToggleSelectedItemsPin => ("sim_shell::ToggleSelectedItemsPin", crate::ToggleSelectedItemsPin),
    UnlockCanvas => ("sim_shell::UnlockCanvas", crate::UnlockCanvas),
    GraphZoomIn => ("sim_graph::GraphZoomIn", crate::GraphZoomIn),
    GraphZoomOut => ("sim_graph::GraphZoomOut", crate::GraphZoomOut),
    ClearPendingTasks => ("sim_shell::ClearPendingTasks", crate::ClearPendingTasks),
    Interrupt => ("sim_shell::Interrupt", crate::Interrupt),
    QueuePrompt => ("sim_shell::QueuePrompt", crate::QueuePrompt),
    QueuePromptFront => ("sim_shell::QueuePromptFront", crate::QueuePromptFront),
    QueueSelectedOutputNodes => ("sim_shell::QueueSelectedOutputNodes", crate::QueueSelectedOutputNodes),
    ToggleQueueOverlay => ("sim_shell::ToggleQueueOverlay", crate::ToggleQueueOverlay),
    ToggleQpov2 => ("sim_shell::ToggleQpov2", crate::ToggleQpov2),
    ExitSubgraph => ("sim_shell::ExitSubgraph", crate::ExitSubgraph),
    FitGroupToContents => ("sim_shell::FitGroupToContents", crate::FitGroupToContents),
    GroupSelectedNodes => ("sim_shell::GroupSelectedNodes", crate::GroupSelectedNodes),
    UnpackSubgraph => ("sim_shell::UnpackSubgraph", crate::UnpackSubgraph),
    PublishSubgraph => ("sim_shell::PublishSubgraph", crate::PublishSubgraph),
    RefreshNodeDefinitions => ("sim_shell::RefreshNodeDefinitions", crate::RefreshNodeDefinitions),
    ConvertToSubgraph => ("sim_shell::ConvertToSubgraph", crate::ConvertToSubgraph),
    ToggleWorkflowsSidebar => ("sim_shell::ToggleWorkflowsSidebar", crate::ToggleWorkflowsSidebar),
    ToggleNodeLibrarySidebar => ("sim_shell::ToggleNodeLibrarySidebar", crate::ToggleNodeLibrarySidebar),
    ToggleModelLibrarySidebar => ("sim_shell::ToggleModelLibrarySidebar", crate::ToggleModelLibrarySidebar),
    ToggleAssetsSidebar => ("sim_shell::ToggleAssetsSidebar", crate::ToggleAssetsSidebar),
    ToggleLinear => ("sim_shell::ToggleLinear", crate::ToggleLinear),
    SaveWorkflow => ("sim_shell::SaveWorkflow", crate::SaveWorkflow),
    OpenWorkflow => ("sim_shell::OpenWorkflow", crate::OpenWorkflow),
    ShowSettings => ("sim_shell::ShowSettings", crate::ShowSettings),
    ShowKeybindings => ("sim_shell::ShowKeybindings", crate::ShowKeybindings),
    ToggleLogsPanel => ("sim_shell::ToggleLogsPanel", crate::ToggleLogsPanel),
    CopySelectedExecutionId => ("sim_shell::CopySelectedExecutionId", crate::CopySelectedExecutionId),
    CopySelectedExecutionError => ("sim_shell::CopySelectedExecutionError", crate::CopySelectedExecutionError),
    RemoveSelectedExecution => ("sim_shell::RemoveSelectedExecution", crate::RemoveSelectedExecution),
    CancelSelectedExecution => ("sim_shell::CancelSelectedExecution", crate::CancelSelectedExecution),
    ToggleDockedExecutionHistory => ("sim_shell::ToggleDockedExecutionHistory", crate::ToggleDockedExecutionHistory),
    ToggleExecutionProgress => ("sim_shell::ToggleExecutionProgress", crate::ToggleExecutionProgress),
    ClearExecutionHistory => ("sim_shell::ClearExecutionHistory", crate::ClearExecutionHistory),
    ExecutionRunManual => ("sim_shell::ExecutionRunManual", crate::ExecutionRunManual),
    ExecutionRunOnChange => ("sim_shell::ExecutionRunOnChange", crate::ExecutionRunOnChange),
    ExecutionRunInstantIdle => ("sim_shell::ExecutionRunInstantIdle", crate::ExecutionRunInstantIdle),
);

pub const DEFAULT_SIM_KEYMAP_PATH: &str = "keymaps/default-sim.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CommandAvailability {
    Active,
    PlatformSpecific,
    Experimental,
    DeveloperOnly,
    CloudPaid,
    Deprecated,
}

impl CommandAvailability {
    pub fn catalog_name(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::PlatformSpecific => "platform-specific",
            Self::Experimental => "experimental",
            Self::DeveloperOnly => "developer-only",
            Self::CloudPaid => "cloud/paid",
            Self::Deprecated => "deprecated/dead",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativePlacement {
    GraphWorkspace,
    ExecutionDock,
    WorkspaceShell,
    Settings,
    DesktopIntegration,
    NodeLibrary,
    AssetBrowser,
    ExtensionHost,
    MediaEditor,
    HelpCenter,
    Account,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum CommandNativeStatus {
    Executable,
    Infrastructure {
        owner: &'static str,
    },
    RequiresInput {
        owner: &'static str,
        input: &'static str,
    },
    LaterOwned {
        owner: &'static str,
    },
    Gated {
        owner: &'static str,
        gate: &'static str,
    },
    Legacy {
        owner: &'static str,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct CommandRegistration {
    pub feature_id: &'static str,
    pub command_id: &'static str,
    pub label: &'static str,
    pub availability: CommandAvailability,
    pub placement: NativePlacement,
    pub owner: &'static str,
    pub status: CommandNativeStatus,
    pub graph_action: Option<CatalogGraphAction>,
    pub gpui_action: Option<NativeAction>,
}

pub fn command_registry() -> impl ExactSizeIterator<Item = CommandRegistration> {
    GENERATED_COMMAND_CATALOG.iter().copied().map(|row| {
        let graph_action = CatalogGraphAction::ALL
            .iter()
            .copied()
            .find(|action| action.command_id() == row.command_id);
        CommandRegistration {
            feature_id: row.feature_id,
            command_id: row.command_id,
            label: row.label,
            availability: row.availability,
            placement: row.placement,
            owner: row.owner,
            status: row.status,
            graph_action,
            gpui_action: row.native_action,
        }
    })
}

pub fn command_registration(command_id: &str) -> Option<CommandRegistration> {
    command_registry().find(|registration| registration.command_id == command_id)
}

#[cfg(test)]
pub(crate) fn native_command_id_from_catalog(command_id: &str) -> std::borrow::Cow<'_, str> {
    if let Some(suffix) = command_id.strip_prefix("Comfy.") {
        format!("Sim.{suffix}").into()
    } else if let Some(suffix) = command_id.strip_prefix("Comfy-Desktop.") {
        format!("Sim-Desktop.{suffix}").into()
    } else {
        command_id.into()
    }
}

#[cfg(all(test, feature = "test-support"))]
mod sim_rename_tests {
    use super::*;
    use gpui::{KeyBindingContextPredicate, KeyContext, TestAppContext};

    #[gpui::test]
    fn sim_actions_restore_legacy_bindings_through_the_registered_actions(cx: &mut TestAppContext) {
        cx.update(|cx| {
            for (legacy, canonical) in [
                ("comfy_shell::SaveWorkflow", "sim_shell::SaveWorkflow"),
                ("comfy_shell::QueuePrompt", "sim_shell::QueuePrompt"),
                ("comfy_shell::ShowSettings", "sim_shell::ShowSettings"),
                ("comfy_graph::GraphUndo", "sim_graph::GraphUndo"),
                ("comfy_graph::GraphPaste", "sim_graph::GraphPaste"),
            ] {
                let old = cx
                    .build_action(legacy, None)
                    .expect("legacy action resolves");
                let new = cx
                    .build_action(canonical, None)
                    .expect("Sim action resolves");
                assert_eq!(old.name(), new.name());
                assert_eq!(old.as_any().type_id(), new.as_any().type_id());
            }
        });
    }

    #[test]
    fn sim_commands_use_the_source_catalog_projection_without_runtime_aliases() {
        for (legacy, canonical) in [
            ("Comfy.QueuePrompt", "Sim.QueuePrompt"),
            ("Comfy.OpenWorkflow", "Sim.OpenWorkflow"),
            (
                "Comfy.Graph.ConvertToSubgraph",
                "Sim.Graph.ConvertToSubgraph",
            ),
        ] {
            assert!(command_registration(legacy).is_none());
            let old = command_registration(&native_command_id_from_catalog(legacy))
                .expect("catalog projects to Sim");
            let new = command_registration(canonical).expect("Sim command resolves");
            assert_eq!(old, new);
        }
        assert!(command_registration("Comfy.DoesNotExist").is_none());
        assert!(command_registration("Sim.DoesNotExist").is_none());
    }

    #[test]
    fn sim_context_keeps_legacy_custom_bindings_and_text_input_exclusions() {
        let context =
            KeyContext::parse(crate::SIM_GRAPH_KEY_CONTEXT).expect("graph contexts parse");
        let old = KeyBindingContextPredicate::parse("ComfyGraph && !ComfyTextInput")
            .expect("old predicate parses");
        let new = KeyBindingContextPredicate::parse(crate::SIM_KEYMAP_CONTEXT)
            .expect("new predicate parses");
        assert!(old.eval(std::slice::from_ref(&context)));
        assert!(new.eval(&[context]));
    }
}
