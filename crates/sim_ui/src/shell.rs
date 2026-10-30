use crate::{
    CommandNativeStatus, NativeAction, NativePlacement, command_registration,
    generated_keybinding_catalog::GENERATED_KEYBINDING_CATALOG,
    generated_menu_catalog::{GENERATED_COMPONENT_CATALOG, GENERATED_MENU_CATALOG},
};
use gpui::{Menu, MenuItem};
use serde::Serialize;

pub use crate::generated_keybinding_catalog::GENERATED_KEY_CONTEXT as SIM_CANONICAL_GRAPH_KEY_CONTEXT;
pub use crate::generated_keybinding_catalog::GENERATED_KEYMAP_CONTEXT as SIM_KEYMAP_CONTEXT;
pub use crate::generated_keybinding_catalog::GENERATED_TEXT_INPUT_KEY_CONTEXT as SIM_CANONICAL_TEXT_INPUT_KEY_CONTEXT;

pub const SIM_GRAPH_KEY_CONTEXT: &str = "SimGraph ComfyGraph";
pub const SIM_TEXT_INPUT_KEY_CONTEXT: &str = "SimTextInput ComfyTextInput";
pub use crate::generated_keybinding_catalog::GeneratedKeybindingCatalogRow as KeybindingRegistration;
pub use crate::generated_menu_catalog::{
    GeneratedComponentCatalogRow, GeneratedComponentDisposition, GeneratedComponentPlacement,
    GeneratedGraphContextAction, GeneratedGraphContextInfrastructure, GeneratedGraphContextSurface,
    GeneratedMenuAvailability, GeneratedMenuItemKind,
};

pub const GRAPH_CONTEXT_MENU_OWNER: &str = "comfy-parity-graph-context-menu-surfaces";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum CommandDispatchOutcome {
    Executed {
        command_id: String,
    },
    Infrastructure {
        command_id: String,
        owner: &'static str,
    },
    RequiresInput {
        command_id: String,
        input: &'static str,
    },
    LaterOwned {
        command_id: String,
        owner: &'static str,
    },
    Gated {
        command_id: String,
        owner: &'static str,
        gate: &'static str,
    },
    Legacy {
        command_id: String,
        owner: &'static str,
    },
    Rejected {
        command_id: String,
        error: String,
    },
    Unknown {
        command_id: String,
    },
}

impl CommandDispatchOutcome {
    pub fn is_executed(&self) -> bool {
        matches!(self, Self::Executed { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ResolvedKeybindingRegistration {
    pub binding: KeybindingRegistration,
    pub placement: NativePlacement,
    pub owner: &'static str,
    pub status: CommandNativeStatus,
    pub native_action: NativeAction,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum KeybindingRegistryError {
    #[error("keybinding feature `{feature_id}` references unknown command `{command_id}`")]
    UnknownCommand {
        feature_id: &'static str,
        command_id: &'static str,
    },
    #[error("keybinding feature `{feature_id}` has no native action for command `{command_id}`")]
    MissingNativeAction {
        feature_id: &'static str,
        command_id: &'static str,
    },
    #[error(
        "keybinding feature `{feature_id}` disagrees with command `{command_id}` about its native action"
    )]
    NativeActionMismatch {
        feature_id: &'static str,
        command_id: &'static str,
    },
}

pub fn keybinding_registry()
-> impl ExactSizeIterator<Item = Result<ResolvedKeybindingRegistration, KeybindingRegistryError>> {
    GENERATED_KEYBINDING_CATALOG.iter().copied().map(|binding| {
        let command = command_registration(binding.command_id).ok_or(
            KeybindingRegistryError::UnknownCommand {
                feature_id: binding.feature_id,
                command_id: binding.command_id,
            },
        )?;
        let native_action =
            command
                .gpui_action
                .ok_or(KeybindingRegistryError::MissingNativeAction {
                    feature_id: binding.feature_id,
                    command_id: binding.command_id,
                })?;
        if native_action != binding.native_action {
            return Err(KeybindingRegistryError::NativeActionMismatch {
                feature_id: binding.feature_id,
                command_id: binding.command_id,
            });
        }
        Ok(ResolvedKeybindingRegistration {
            binding,
            placement: command.placement,
            owner: command.owner,
            status: command.status,
            native_action,
        })
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MenuReconciliation {
    pub feature_id: String,
    pub label: String,
    pub source_condition: String,
    pub item_kind: GeneratedMenuItemKind,
    pub availability: MenuAvailability,
    pub placement: NativePlacement,
    pub owner: &'static str,
    pub command_id: Option<String>,
    pub status: CommandNativeStatus,
    pub native_action: Option<NativeAction>,
    pub context_action: Option<GeneratedGraphContextAction>,
    pub context_infrastructure: Option<GeneratedGraphContextInfrastructure>,
    pub context_surface: Option<GeneratedGraphContextSurface>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MenuAvailability {
    Active,
    Conditional,
    CloudPaid,
    InfrastructureOnly,
    Experimental,
    PlatformSpecific,
    Deprecated,
    DeveloperOnly,
}

impl MenuAvailability {
    fn from_generated(value: GeneratedMenuAvailability) -> Self {
        match value {
            GeneratedMenuAvailability::Active => Self::Active,
            GeneratedMenuAvailability::Conditional => Self::Conditional,
            GeneratedMenuAvailability::CloudPaid => Self::CloudPaid,
            GeneratedMenuAvailability::InfrastructureOnly => Self::InfrastructureOnly,
            GeneratedMenuAvailability::Experimental => Self::Experimental,
            GeneratedMenuAvailability::PlatformSpecific => Self::PlatformSpecific,
            GeneratedMenuAvailability::Deprecated => Self::Deprecated,
            GeneratedMenuAvailability::DeveloperOnly => Self::DeveloperOnly,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MenuRegistryError {
    #[error("menu feature `{feature_id}` references unknown command `{command_id}`")]
    UnknownCommand {
        feature_id: &'static str,
        command_id: &'static str,
    },
    #[error(
        "menu feature `{feature_id}` maps command `{command_id}` to {actual_placement:?}/{actual_owner}, expected {expected_placement:?}/{expected_owner}"
    )]
    CommandDispositionMismatch {
        feature_id: &'static str,
        command_id: &'static str,
        actual_placement: NativePlacement,
        actual_owner: &'static str,
        expected_placement: NativePlacement,
        expected_owner: &'static str,
    },
    #[error("menu feature `{feature_id}` has conflicting graph context bindings")]
    ConflictingGraphContextBindings { feature_id: &'static str },
    #[error("menu feature `{feature_id}` has an invalid graph context infrastructure disposition")]
    InvalidGraphContextInfrastructure { feature_id: &'static str },
}

fn resolve_generated_menu(
    generated: &crate::generated_menu_catalog::GeneratedMenuCatalogRow,
) -> Result<MenuReconciliation, MenuRegistryError> {
    if generated.context_action.is_some() && generated.context_infrastructure.is_some() {
        return Err(MenuRegistryError::ConflictingGraphContextBindings {
            feature_id: generated.feature_id,
        });
    }
    if generated.context_infrastructure.is_some()
        && (generated.availability != GeneratedMenuAvailability::InfrastructureOnly
            || generated.context_surface != Some(GeneratedGraphContextSurface::Infrastructure)
            || generated.command_id.is_some()
            || generated.native_action.is_some())
    {
        return Err(MenuRegistryError::InvalidGraphContextInfrastructure {
            feature_id: generated.feature_id,
        });
    }
    let (label, active_status, active_action) = match generated.command_id {
        Some(command_id) => {
            let command =
                command_registration(command_id).ok_or(MenuRegistryError::UnknownCommand {
                    feature_id: generated.feature_id,
                    command_id,
                })?;
            if command.placement != generated.placement || command.owner != generated.owner {
                return Err(MenuRegistryError::CommandDispositionMismatch {
                    feature_id: generated.feature_id,
                    command_id,
                    actual_placement: command.placement,
                    actual_owner: command.owner,
                    expected_placement: generated.placement,
                    expected_owner: generated.owner,
                });
            }
            (command.label, command.status, command.gpui_action)
        }
        None if generated.native_action.is_some() => (
            generated.label,
            CommandNativeStatus::Executable,
            generated.native_action,
        ),
        None if generated.context_action.is_some() => {
            (generated.label, CommandNativeStatus::Executable, None)
        }
        None if generated.context_infrastructure.is_some() => (
            generated.label,
            CommandNativeStatus::Infrastructure {
                owner: generated.owner,
            },
            None,
        ),
        None => (
            generated.label,
            CommandNativeStatus::LaterOwned {
                owner: generated.owner,
            },
            None,
        ),
    };
    let availability = MenuAvailability::from_generated(generated.availability);
    let (owner, status) = if generated.context_action.is_some() {
        (generated.owner, CommandNativeStatus::Executable)
    } else if generated.context_infrastructure.is_some() {
        (
            generated.owner,
            CommandNativeStatus::Infrastructure {
                owner: generated.owner,
            },
        )
    } else {
        menu_status(availability, generated.owner, active_status)
    };
    let native_action = (status == CommandNativeStatus::Executable)
        .then_some(active_action)
        .flatten();
    Ok(MenuReconciliation {
        feature_id: generated.feature_id.to_owned(),
        label: label.to_owned(),
        source_condition: generated.source_condition.to_owned(),
        item_kind: generated.item_kind,
        availability,
        placement: generated.placement,
        owner,
        command_id: generated.command_id.map(str::to_owned),
        status,
        native_action,
        context_action: generated.context_action,
        context_infrastructure: generated.context_infrastructure,
        context_surface: generated.context_surface,
    })
}

pub fn menu_registry()
-> impl ExactSizeIterator<Item = Result<MenuReconciliation, MenuRegistryError>> {
    GENERATED_MENU_CATALOG.iter().map(resolve_generated_menu)
}

pub fn graph_context_registry()
-> impl Iterator<Item = Result<MenuReconciliation, MenuRegistryError>> {
    GENERATED_MENU_CATALOG
        .iter()
        .filter(|row| row.owner == GRAPH_CONTEXT_MENU_OWNER)
        .map(resolve_generated_menu)
}

pub fn menu_registration(
    feature_id: &str,
) -> Result<Option<MenuReconciliation>, MenuRegistryError> {
    GENERATED_MENU_CATALOG
        .iter()
        .find(|row| row.feature_id == feature_id)
        .map(resolve_generated_menu)
        .transpose()
}

pub fn component_surface_registry() -> impl ExactSizeIterator<Item = GeneratedComponentCatalogRow> {
    GENERATED_COMPONENT_CATALOG.iter().copied()
}

fn menu_status(
    availability: MenuAvailability,
    owner: &'static str,
    active_status: CommandNativeStatus,
) -> (&'static str, CommandNativeStatus) {
    match availability {
        MenuAvailability::Active => (owner, active_status),
        MenuAvailability::Conditional => (
            owner,
            CommandNativeStatus::Gated {
                owner,
                gate: "conditional",
            },
        ),
        MenuAvailability::CloudPaid => (
            owner,
            CommandNativeStatus::Gated {
                owner,
                gate: "cloud-or-paid-service",
            },
        ),
        MenuAvailability::InfrastructureOnly => {
            (owner, CommandNativeStatus::Infrastructure { owner })
        }
        MenuAvailability::Experimental => (
            owner,
            CommandNativeStatus::Gated {
                owner,
                gate: "experimental",
            },
        ),
        MenuAvailability::PlatformSpecific => (
            owner,
            CommandNativeStatus::Gated {
                owner,
                gate: "platform-specific",
            },
        ),
        MenuAvailability::Deprecated => (owner, CommandNativeStatus::Legacy { owner }),
        MenuAvailability::DeveloperOnly => (
            owner,
            CommandNativeStatus::Gated {
                owner,
                gate: "developer-only",
            },
        ),
    }
}

const MENU_SECTIONS: [(NativePlacement, &str); 11] = [
    (NativePlacement::WorkspaceShell, "Workspace"),
    (NativePlacement::GraphWorkspace, "Graph"),
    (NativePlacement::ExecutionDock, "Execution"),
    (NativePlacement::NodeLibrary, "Nodes"),
    (NativePlacement::AssetBrowser, "Assets"),
    (NativePlacement::MediaEditor, "Media"),
    (NativePlacement::Settings, "Settings"),
    (NativePlacement::HelpCenter, "Help"),
    (NativePlacement::Account, "Account"),
    (NativePlacement::ExtensionHost, "Extensions"),
    (NativePlacement::DesktopIntegration, "Desktop"),
];

pub fn native_menu_action_names() -> Result<Vec<&'static str>, MenuRegistryError> {
    menu_registry()
        .map(|registration| Ok(registration?.native_action.map(NativeAction::name)))
        .filter_map(Result::transpose)
        .collect()
}

pub fn try_sim_menu() -> Result<Menu, MenuRegistryError> {
    let registrations = menu_registry().collect::<Result<Vec<_>, _>>()?;
    let sections = MENU_SECTIONS.into_iter().filter_map(|(placement, name)| {
        let items = registrations
            .iter()
            .filter(|registration| registration.placement == placement)
            .filter_map(|registration| {
                registration
                    .native_action
                    .map(|action| action.menu_item(registration.label.clone()))
            })
            .collect::<Vec<_>>();
        (!items.is_empty()).then(|| MenuItem::submenu(Menu::new(name).items(items)))
    });
    Ok(Menu::new("Sim").items(sections))
}

pub fn sim_menu() -> Menu {
    match try_sim_menu() {
        Ok(menu) => menu,
        Err(error) => {
            log::error!("failed to build the canonical Sim menu registry: {error}");
            Menu::new("Sim").disabled(true)
        }
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;
    use crate::{
        CommandAvailability, GraphCopy, GraphSelectAll, GraphWorkspaceItem, GraphWorkspaceModel,
        GroupSelectedNodes, OpenWorkflow, QueuePrompt, ToggleMinimap, ToggleSelectedBypass,
        command_registry,
    };
    use gpui::{Focusable as _, KeyBinding, KeyContext, TestAppContext, WeakEntity};
    use serde_json::{Value, json};
    use settings::SettingsStore;
    use sha2::{Digest, Sha256};
    use sim_runtime::{
        GraphCommand, GraphDocument, GraphIdentifier, GraphNode, GraphPoint, NodeCreationSource,
        WorkflowStorageProvider,
    };
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
        path::PathBuf,
    };
    use uuid::Uuid;

    const COMMAND_CSV: &str =
        include_str!("../../../.agents/specs/comfy-parity/catalogs/frontend-commands.csv");
    const NATIVE_COMMAND_DISPOSITIONS_CSV: &str = include_str!(
        "../../../.agents/specs/comfy-parity/catalogs/native-command-dispositions.csv"
    );
    const KEYBINDING_CSV: &str =
        include_str!("../../../.agents/specs/comfy-parity/catalogs/frontend-keybindings.csv");
    const MENU_CSV: &str =
        include_str!("../../../.agents/specs/comfy-parity/catalogs/frontend-menus.csv");
    const NATIVE_MENU_DISPOSITIONS_CSV: &str =
        include_str!("../../../.agents/specs/comfy-parity/catalogs/native-menu-dispositions.csv");
    const COMPONENT_SURFACE_CSV: &str = include_str!(
        "../../../.agents/specs/comfy-parity/catalogs/frontend-component-surfaces.csv"
    );
    const NATIVE_COMPONENT_DISPOSITIONS_CSV: &str = include_str!(
        "../../../.agents/specs/comfy-parity/catalogs/native-component-dispositions.csv"
    );
    const NATIVE_SPEC_MAPPING: &str =
        include_str!("../../../.agents/specs/comfy-parity/catalogs/native-spec-mapping.json");
    const SHELL_TRACE_CLOSURE: &str = include_str!(
        "../../../.agents/specs/comfy-parity/catalogs/native-shell-trace-closure.json"
    );
    const PARITY_MATRIX: &str =
        include_str!("../../../.agents/specs/comfy-parity/parity-matrix.md");
    const TASKS: &str = include_str!("../../../.agents/specs/comfy-parity/tasks.md");
    const DEFAULT_KEYMAP: &str = include_str!("../../../assets/keymaps/default-sim.json");
    const GRAPH_SHELL_TASK_ID: &str = "comfy-parity-graph-shell-accessibility";

    fn csv_rows(source: &str) -> Result<Vec<BTreeMap<String, String>>, String> {
        let mut records = Vec::<Vec<String>>::new();
        let mut record = Vec::new();
        let mut field = String::new();
        let mut quoted = false;
        let mut characters = source.chars().peekable();
        while let Some(character) = characters.next() {
            match character {
                '"' if quoted && characters.peek() == Some(&'"') => {
                    field.push('"');
                    characters.next();
                }
                '"' => quoted = !quoted,
                ',' if !quoted => record.push(std::mem::take(&mut field)),
                '\n' if !quoted => {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                }
                '\r' if !quoted => {}
                _ => field.push(character),
            }
        }
        if quoted {
            return Err("CSV ended inside a quoted field".to_owned());
        }
        if !field.is_empty() || !record.is_empty() {
            record.push(field);
            records.push(record);
        }
        let headers = records
            .first()
            .cloned()
            .ok_or_else(|| "CSV has no header".to_owned())?;
        records
            .into_iter()
            .skip(1)
            .filter(|record| !record.iter().all(String::is_empty))
            .map(|record| {
                if record.len() != headers.len() {
                    return Err(format!(
                        "CSV row has {} columns; expected {}",
                        record.len(),
                        headers.len()
                    ));
                }
                Ok(headers.iter().cloned().zip(record).collect())
            })
            .collect()
    }

    fn field<'a>(row: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str, String> {
        row.get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("CSV row is missing `{name}`"))
    }

    fn digest(source: &str) -> String {
        format!("{:x}", Sha256::digest(source.as_bytes()))
    }

    fn collect_menu_action_names(items: &[MenuItem], names: &mut Vec<String>) {
        for item in items {
            match item {
                MenuItem::Action { action, .. } => names.push(action.name().to_owned()),
                MenuItem::Submenu(menu) => collect_menu_action_names(&menu.items, names),
                MenuItem::Separator | MenuItem::SystemMenu(_) => {}
            }
        }
    }

    fn registry_reconciliation() -> Result<Value, String> {
        let rows = csv_rows(COMMAND_CSV)?;
        let disposition_rows = csv_rows(NATIVE_COMMAND_DISPOSITIONS_CSV)?
            .into_iter()
            .map(|row| {
                let command_id = field(&row, "command_id")?.to_owned();
                Ok((command_id, row))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let registrations = command_registry().collect::<Vec<_>>();
        if rows.len() != 118
            || registrations.len() != rows.len()
            || disposition_rows.len() != rows.len()
        {
            return Err(format!(
                "command count mismatch: source={} ledger={} registry={}",
                rows.len(),
                disposition_rows.len(),
                registrations.len()
            ));
        }
        let mut identifiers = BTreeSet::new();
        let mut status_counts = BTreeMap::<String, usize>::new();
        let mut owner_counts = BTreeMap::<String, usize>::new();
        let mut canonical_dispositions = Vec::new();
        for (row, registration) in rows.iter().zip(&registrations) {
            let command_id = field(row, "command_id")?;
            if !identifiers.insert(command_id) {
                return Err(format!("duplicate command `{command_id}`"));
            }
            if registration.command_id != crate::native_command_id_from_catalog(command_id)
                || registration.label != field(row, "label")?
                || registration.availability.catalog_name() != field(row, "availability")?
            {
                return Err(format!("command registry drift for `{command_id}`"));
            }
            let disposition = disposition_rows
                .get(command_id)
                .ok_or_else(|| format!("command ledger is missing `{command_id}`"))?;
            let key = match registration.status {
                CommandNativeStatus::Executable => {
                    if registration.gpui_action.is_none() {
                        return Err(format!(
                            "executable command `{command_id}` has no registered GPUI action"
                        ));
                    }
                    "executable"
                }
                CommandNativeStatus::Infrastructure { owner } => {
                    if owner.is_empty() {
                        return Err(format!("`{command_id}` has an empty infrastructure owner"));
                    }
                    "infrastructure"
                }
                CommandNativeStatus::RequiresInput { owner, .. } => {
                    if owner.is_empty() {
                        return Err(format!("`{command_id}` has an empty input owner"));
                    }
                    "requires-input"
                }
                CommandNativeStatus::LaterOwned { owner } => {
                    if owner.is_empty() {
                        return Err(format!("`{command_id}` has an empty later owner"));
                    }
                    "later-owned"
                }
                CommandNativeStatus::Gated { owner, gate } => {
                    if owner.is_empty() || gate.is_empty() {
                        return Err(format!("`{command_id}` has an incomplete gate"));
                    }
                    "gated"
                }
                CommandNativeStatus::Legacy { owner } => {
                    if owner.is_empty() {
                        return Err(format!("`{command_id}` has an empty legacy owner"));
                    }
                    "legacy"
                }
            };
            let input_requirement = match registration.status {
                CommandNativeStatus::RequiresInput { input, .. } => input,
                _ => "",
            };
            let native_action = registration
                .gpui_action
                .map(|action| format!("{action:?}"))
                .unwrap_or_default();
            if registration.feature_id != field(disposition, "feature_id")?
                || format!("{:?}", registration.placement) != field(disposition, "placement")?
                || registration.owner != field(disposition, "owner_task_id")?
                || key != field(disposition, "disposition")?
                || native_action != field(disposition, "native_action")?
                || input_requirement != field(disposition, "input_requirement")?
            {
                return Err(format!(
                    "command `{command_id}` does not match its generated authoritative disposition"
                ));
            }
            *status_counts.entry(key.to_owned()).or_default() += 1;
            *owner_counts
                .entry(registration.owner.to_owned())
                .or_default() += 1;
            canonical_dispositions.push(json!({
                "id": registration.command_id,
                "status": registration.status,
                "owner": registration.owner,
                "placement": registration.placement,
                "graph_action": registration.graph_action,
                "gpui_action": registration.gpui_action,
            }));
        }
        let memory_planner_commands = [
            "Sim.Memory.UnloadModels",
            "Sim.Memory.UnloadModelsAndExecutionCache",
        ];
        for command_id in memory_planner_commands {
            let registration = registrations
                .iter()
                .find(|registration| registration.command_id == command_id)
                .ok_or_else(|| format!("missing memory command `{command_id}`"))?;
            if registration.placement != NativePlacement::ExecutionDock
                || !matches!(
                    registration.status,
                    CommandNativeStatus::LaterOwned { owner }
                        if owner == registration.owner
                )
            {
                return Err(format!(
                    "memory command `{command_id}` lost its execution-dock/memory-planner disposition"
                ));
            }
        }
        let disposition_bytes = serde_json::to_vec(&canonical_dispositions)
            .map_err(|error| format!("serialize command dispositions: {error}"))?;
        Ok(json!({
            "name": "118-command-registry-reconciliation",
            "passed": true,
            "row_count": rows.len(),
            "status_counts": status_counts,
            "owner_counts": owner_counts,
            "disposition_row_count": canonical_dispositions.len(),
            "disposition_digest": format!("{:x}", Sha256::digest(disposition_bytes)),
            "memory_planner_command_count": memory_planner_commands.len(),
            "digest": digest(COMMAND_CSV),
            "generated_ledger_digest": digest(NATIVE_COMMAND_DISPOSITIONS_CSV),
        }))
    }

    fn keybinding_reconciliation(cx: &TestAppContext) -> Result<Value, String> {
        let rows = csv_rows(KEYBINDING_CSV)?;
        let registrations = keybinding_registry()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("resolve keybinding registry: {error}"))?;
        if rows.len() != 34 || registrations.len() != rows.len() {
            return Err(format!(
                "keybinding count mismatch: source={} registry={}",
                rows.len(),
                registrations.len()
            ));
        }
        let asset: Value = serde_json::from_str(DEFAULT_KEYMAP)
            .map_err(|error| format!("default keymap JSON: {error}"))?;
        let sections = asset
            .as_array()
            .ok_or_else(|| "default keymap must be an array".to_owned())?;
        let section = sections
            .first()
            .and_then(Value::as_object)
            .ok_or_else(|| "default keymap has no object section".to_owned())?;
        if section.get("context").and_then(Value::as_str) != Some(SIM_KEYMAP_CONTEXT) {
            return Err(
                "default keymap is not scoped to non-editable SimGraph controls".to_owned(),
            );
        }
        let bindings = section
            .get("bindings")
            .and_then(Value::as_object)
            .ok_or_else(|| "default keymap has no bindings object".to_owned())?;
        if bindings.len() != 34 {
            return Err(format!("default keymap has {} bindings", bindings.len()));
        }
        let mut feature_ids = BTreeSet::new();
        let registrations = registrations
            .into_iter()
            .map(|registration| (registration.binding.feature_id, registration))
            .collect::<BTreeMap<_, _>>();
        if registrations.len() != rows.len() {
            return Err("keybinding registry contains duplicate feature identities".to_owned());
        }
        let mut owner_counts = BTreeMap::<String, usize>::new();
        let mut canonical_dispositions = Vec::new();
        for row in &rows {
            let feature_id = field(row, "feature_id")?;
            if !feature_ids.insert(feature_id) {
                return Err(format!("duplicate keybinding feature `{feature_id}`"));
            }
            let registration = registrations
                .get(feature_id)
                .ok_or_else(|| format!("keybinding registry is missing `{feature_id}`"))?;
            if registration.binding.source_combo != field(row, "combo")?
                || registration.binding.command_id
                    != crate::native_command_id_from_catalog(field(row, "command_id")?)
                || registration.binding.availability != CommandAvailability::Active
                || field(row, "availability")? != "active"
            {
                return Err(format!("keybinding registry drift for `{feature_id}`"));
            }
            let asset_action = bindings
                .get(registration.binding.keystroke)
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    format!(
                        "default keymap is missing `{}` as a string action",
                        registration.binding.keystroke
                    )
                })?;
            if asset_action != registration.native_action.name()
                || registration.binding.native_action != registration.native_action
            {
                return Err(format!(
                    "default keymap action drift for `{}`",
                    registration.binding.keystroke
                ));
            }
            cx.update(|cx| cx.build_action(registration.native_action.name(), None))
                .map_err(|error| {
                    format!(
                        "native keymap action `{}` is not registered: {error}",
                        registration.native_action.name()
                    )
                })?;
            *owner_counts
                .entry(registration.owner.to_owned())
                .or_default() += 1;
            canonical_dispositions.push(json!({
                "feature_id": registration.binding.feature_id,
                "command_id": registration.binding.command_id,
                "placement": registration.placement,
                "owner": registration.owner,
                "status": registration.status,
            }));
        }
        let disposition_bytes = serde_json::to_vec(&canonical_dispositions)
            .map_err(|error| format!("serialize keybinding dispositions: {error}"))?;
        Ok(json!({
            "name": "34-scoped-keybinding-reconciliation",
            "passed": true,
            "row_count": rows.len(),
            "context": SIM_KEYMAP_CONTEXT,
            "owner_counts": owner_counts,
            "disposition_digest": format!("{:x}", Sha256::digest(disposition_bytes)),
            "digest": digest(KEYBINDING_CSV),
        }))
    }

    fn menu_reconciliation() -> Result<Value, String> {
        let rows = csv_rows(MENU_CSV)?;
        let disposition_rows = csv_rows(NATIVE_MENU_DISPOSITIONS_CSV)?
            .into_iter()
            .map(|row| {
                let feature_id = field(&row, "feature_id")?.to_owned();
                Ok((feature_id, row))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        if rows.len() != 236 {
            return Err(format!("menu catalog has {} rows", rows.len()));
        }
        if disposition_rows.len() != rows.len() {
            return Err(format!(
                "native menu disposition ledger has {} unique rows, expected {}",
                disposition_rows.len(),
                rows.len()
            ));
        }
        let registrations = menu_registry()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("resolve menu registry: {error}"))?;
        if registrations.len() != rows.len() {
            return Err(format!(
                "menu registry has {} rows, expected {}",
                registrations.len(),
                rows.len()
            ));
        }
        let registrations = registrations
            .into_iter()
            .map(|registration| (registration.feature_id.clone(), registration))
            .collect::<BTreeMap<_, _>>();
        if registrations.len() != rows.len() {
            return Err("menu registry contains duplicate feature identities".to_owned());
        }
        let graph_context_registrations = registrations
            .values()
            .filter(|registration| registration.owner == GRAPH_CONTEXT_MENU_OWNER)
            .collect::<Vec<_>>();
        let graph_context_action_count = graph_context_registrations
            .iter()
            .filter(|registration| registration.context_action.is_some())
            .count();
        let graph_context_infrastructure_count = graph_context_registrations
            .iter()
            .filter(|registration| registration.context_infrastructure.is_some())
            .count();
        if graph_context_registrations.len() != 63
            || graph_context_action_count != 55
            || graph_context_infrastructure_count != 8
        {
            return Err(format!(
                "graph context registry has {} rows, {} actions, and {} infrastructure bindings",
                graph_context_registrations.len(),
                graph_context_action_count,
                graph_context_infrastructure_count
            ));
        }
        for registration in &graph_context_registrations {
            let action = registration.context_action.is_some();
            let infrastructure = registration.context_infrastructure.is_some();
            if action == infrastructure || registration.context_surface.is_none() {
                return Err(format!(
                    "graph context feature `{}` does not have exactly one typed binding and a surface",
                    registration.feature_id
                ));
            }
            if infrastructure
                && (registration.context_surface
                    != Some(GeneratedGraphContextSurface::Infrastructure)
                    || registration.availability != MenuAvailability::InfrastructureOnly
                    || !matches!(
                        registration.status,
                        CommandNativeStatus::Infrastructure { owner }
                            if owner == registration.owner
                    )
                    || registration.native_action.is_some()
                    || registration.command_id.is_some())
            {
                return Err(format!(
                    "graph context infrastructure feature `{}` is invokable or lacks its canonical owner",
                    registration.feature_id
                ));
            }
            if action
                && (!matches!(registration.status, CommandNativeStatus::Executable)
                    || registration.context_surface
                        == Some(GeneratedGraphContextSurface::Infrastructure))
            {
                return Err(format!(
                    "graph context action feature `{}` is not executable on a user surface",
                    registration.feature_id
                ));
            }
        }
        let mut feature_ids = BTreeSet::new();
        let mut placement_counts = BTreeMap::<String, usize>::new();
        let mut owner_counts = BTreeMap::<String, usize>::new();
        let mut status_counts = BTreeMap::<String, usize>::new();
        let mut canonical_dispositions = Vec::new();
        for row in &rows {
            let feature_id = field(row, "feature_id")?;
            if !feature_ids.insert(feature_id) {
                return Err(format!("duplicate menu feature `{feature_id}`"));
            }
            let reconciliation = registrations
                .get(feature_id)
                .ok_or_else(|| format!("menu registry is missing `{feature_id}`"))?;
            let generated = GENERATED_MENU_CATALOG
                .iter()
                .find(|row| row.feature_id == feature_id)
                .ok_or_else(|| format!("generated menu catalog is missing `{feature_id}`"))?;
            let disposition = disposition_rows
                .get(feature_id)
                .ok_or_else(|| format!("menu disposition ledger is missing `{feature_id}`"))?;
            if reconciliation.owner.is_empty() {
                return Err(format!("menu feature `{feature_id}` has no owner"));
            }
            let expected_label = reconciliation
                .command_id
                .as_deref()
                .and_then(command_registration)
                .map(|command| command.label)
                .unwrap_or(field(row, "label_or_action")?);
            if reconciliation.feature_id != feature_id
                || reconciliation.label != expected_label
                || generated.menu_surface != field(row, "menu_surface")?
            {
                return Err(format!("menu feature `{feature_id}` lost stable metadata"));
            }
            let catalog_availability = match reconciliation.availability {
                MenuAvailability::Active => "active",
                MenuAvailability::Conditional => "conditional",
                MenuAvailability::CloudPaid => "cloud/paid",
                MenuAvailability::InfrastructureOnly => "infrastructure-only",
                MenuAvailability::Experimental => "experimental",
                MenuAvailability::PlatformSpecific => "platform-specific",
                MenuAvailability::Deprecated => "deprecated/dead",
                MenuAvailability::DeveloperOnly => "developer-only",
            };
            if catalog_availability != field(row, "availability")? {
                return Err(format!("menu feature `{feature_id}` lost availability"));
            }
            if format!("{:?}", reconciliation.placement) != field(disposition, "placement")?
                || reconciliation.owner != field(disposition, "owner_task_id")?
            {
                return Err(format!(
                    "menu feature `{feature_id}` does not match its generated placement/owner ledger"
                ));
            }
            let context_action = reconciliation
                .context_action
                .map(|action| format!("{action:?}"))
                .unwrap_or_default();
            let context_infrastructure = reconciliation
                .context_infrastructure
                .map(|infrastructure| format!("{infrastructure:?}"))
                .unwrap_or_default();
            let context_surface = reconciliation
                .context_surface
                .map(|surface| format!("{surface:?}"))
                .unwrap_or_default();
            if context_action != field(disposition, "context_action")?
                || context_infrastructure != field(disposition, "context_infrastructure")?
                || context_surface != field(disposition, "context_surface")?
            {
                return Err(format!(
                    "menu feature `{feature_id}` does not match its generated graph context binding"
                ));
            }
            let generated_disposition = field(disposition, "disposition")?;
            let decision_kind = match generated_disposition {
                "canonical-command" | "native" => "place",
                "infrastructure" if reconciliation.context_infrastructure.is_some() => {
                    "place-prerequisite"
                }
                "deferred" | "gated-or-deferred" | "infrastructure" | "legacy" => "defer",
                value => {
                    return Err(format!(
                        "menu feature `{feature_id}` has unknown disposition `{value}`"
                    ));
                }
            };
            let parity_prefix = format!("| `{feature_id}` |");
            let parity_rows = PARITY_MATRIX
                .lines()
                .filter(|line| line.starts_with(&parity_prefix))
                .collect::<Vec<_>>();
            if parity_rows.len() != 1 {
                return Err(format!(
                    "menu feature `{feature_id}` has {} parity rows",
                    parity_rows.len()
                ));
            }
            let decision = parity_rows[0]
                .rsplit('|')
                .nth(1)
                .map(str::trim)
                .ok_or_else(|| format!("menu feature `{feature_id}` has no parity decision"))?;
            let exact_prefix = format!(
                "{decision_kind}:{};owner:{};",
                field(disposition, "placement")?,
                reconciliation.owner
            );
            if !decision.starts_with(&exact_prefix) {
                return Err(format!(
                    "menu feature `{feature_id}` parity decision `{decision}` does not start with `{exact_prefix}`"
                ));
            }
            if let Some(infrastructure) = reconciliation.context_infrastructure {
                let expected_decision =
                    format!("{exact_prefix}adapter:consumed-prerequisite:{infrastructure:?}");
                if decision != expected_decision {
                    return Err(format!(
                        "menu feature `{feature_id}` prerequisite decision `{decision}` does not equal `{expected_decision}`"
                    ));
                }
            }
            if matches!(
                reconciliation.status,
                CommandNativeStatus::LaterOwned { owner }
                    if owner == GRAPH_SHELL_TASK_ID
            ) {
                return Err(format!(
                    "menu feature `{feature_id}` defers to its completed registry owner"
                ));
            }
            if field(row, "item_kind")? == "command-action" {
                let expected = field(row, "action_or_target")?
                    .strip_prefix("execute command ")
                    .ok_or_else(|| format!("command menu `{feature_id}` has malformed target"))?;
                let canonical_expected = crate::native_command_id_from_catalog(expected);
                if reconciliation.command_id.as_deref() != Some(canonical_expected.as_ref()) {
                    return Err(format!(
                        "command menu `{feature_id}` resolved {:?}, expected `{expected}`",
                        reconciliation.command_id
                    ));
                }
            }
            let expected_native_action = match generated_disposition {
                "canonical-command" => reconciliation
                    .command_id
                    .as_deref()
                    .and_then(command_registration)
                    .and_then(|command| command.gpui_action)
                    .map(|action| format!("{action:?}"))
                    .unwrap_or_default(),
                "native" => field(disposition, "native_action")?.to_owned(),
                _ => String::new(),
            };
            if reconciliation
                .native_action
                .map(|action| format!("{action:?}"))
                .unwrap_or_default()
                != expected_native_action
            {
                return Err(format!(
                    "menu feature `{feature_id}` does not preserve its generated action binding"
                ));
            }
            *placement_counts
                .entry(format!("{:?}", reconciliation.placement))
                .or_default() += 1;
            *owner_counts
                .entry(reconciliation.owner.to_owned())
                .or_default() += 1;
            let status_name = match reconciliation.status {
                CommandNativeStatus::Executable => "executable",
                CommandNativeStatus::Infrastructure { .. } => "infrastructure",
                CommandNativeStatus::RequiresInput { .. } => "requires-input",
                CommandNativeStatus::LaterOwned { .. } => "later-owned",
                CommandNativeStatus::Gated { .. } => "gated",
                CommandNativeStatus::Legacy { .. } => "legacy",
            };
            *status_counts.entry(status_name.to_owned()).or_default() += 1;
            canonical_dispositions.push(json!({
                "feature_id": reconciliation.feature_id,
                "label": reconciliation.label,
                "availability": reconciliation.availability,
                "placement": reconciliation.placement,
                "owner": reconciliation.owner,
                "status": reconciliation.status,
                "command_id": reconciliation.command_id,
                "native_action": reconciliation.native_action,
                "context_action": reconciliation.context_action,
                "context_infrastructure": reconciliation.context_infrastructure,
                "context_surface": reconciliation.context_surface,
            }));
        }
        let menu = try_sim_menu().map_err(|error| format!("build production menu: {error}"))?;
        let mut actual_menu_actions = Vec::new();
        collect_menu_action_names(&menu.items, &mut actual_menu_actions);
        let mut expected_menu_actions = native_menu_action_names()
            .map_err(|error| format!("resolve production menu actions: {error}"))?
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        actual_menu_actions.sort();
        expected_menu_actions.sort();
        if actual_menu_actions != expected_menu_actions {
            return Err("production Sim menu diverges from its canonical registry".to_owned());
        }
        let disposition_bytes = serde_json::to_vec(&canonical_dispositions)
            .map_err(|error| format!("serialize menu dispositions: {error}"))?;
        Ok(json!({
            "name": "236-menu-placement-reconciliation",
            "passed": true,
            "row_count": rows.len(),
            "placement_counts": placement_counts,
            "owner_counts": owner_counts,
            "status_counts": status_counts,
            "graph_context_row_count": graph_context_registrations.len(),
            "graph_context_action_count": graph_context_action_count,
            "graph_context_infrastructure_count": graph_context_infrastructure_count,
            "disposition_row_count": canonical_dispositions.len(),
            "disposition_digest": format!("{:x}", Sha256::digest(disposition_bytes)),
            "exact_owner_row_count": registrations.len(),
            "heuristic_fallback_count": 0,
            "production_action_count": actual_menu_actions.len(),
            "digest": digest(MENU_CSV),
            "generated_ledger_digest": digest(NATIVE_MENU_DISPOSITIONS_CSV),
        }))
    }

    fn trace_closure_reconciliation() -> Result<Value, String> {
        let expected_feature_ids = [COMMAND_CSV, KEYBINDING_CSV, MENU_CSV, COMPONENT_SURFACE_CSV]
            .into_iter()
            .map(csv_rows)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .map(|row| field(&row, "feature_id").map(str::to_owned))
            .collect::<Result<BTreeSet<_>, _>>()?;
        if expected_feature_ids.len() != 1_193 {
            return Err(format!(
                "shell source catalogs contain {} unique features, expected 1193",
                expected_feature_ids.len()
            ));
        }

        let closure: Value = serde_json::from_str(SHELL_TRACE_CLOSURE)
            .map_err(|error| format!("shell trace closure JSON: {error}"))?;
        if closure["feature_count"] != 1_193
            || closure["task_id"] != GRAPH_SHELL_TASK_ID
            || closure["validation_id"] != "VAL-GPUI-012"
        {
            return Err("shell trace closure header drifted".to_owned());
        }
        let closure_rows = closure["rows"]
            .as_array()
            .ok_or_else(|| "shell trace closure rows are not an array".to_owned())?;
        let closure_feature_ids = closure_rows
            .iter()
            .map(|row| {
                let feature_id = row["feature_id"]
                    .as_str()
                    .ok_or_else(|| "shell trace closure row has no feature ID".to_owned())?;
                let validations = row["validation_ids"]
                    .as_array()
                    .ok_or_else(|| format!("{feature_id} has no validation array"))?;
                if !validations.iter().any(|value| value == "VAL-GPUI-012") {
                    return Err(format!("{feature_id} is missing VAL-GPUI-012"));
                }
                let criteria = row["requirement_criteria"]
                    .as_array()
                    .ok_or_else(|| format!("{feature_id} has no criterion array"))?;
                if criteria.is_empty() {
                    return Err(format!("{feature_id} has no mapped criterion"));
                }
                Ok(feature_id.to_owned())
            })
            .collect::<Result<BTreeSet<_>, String>>()?;
        if closure_feature_ids != expected_feature_ids {
            return Err(
                "shell trace closure does not exactly match the four source catalogs".to_owned(),
            );
        }

        let mapping: Value = serde_json::from_str(NATIVE_SPEC_MAPPING)
            .map_err(|error| format!("native spec mapping JSON: {error}"))?;
        let special = mapping["special_feature_tasks"]
            .as_object()
            .ok_or_else(|| "native spec mapping has no special feature tasks".to_owned())?;
        let mapped_feature_ids = special
            .iter()
            .filter_map(|(feature_id, tasks)| {
                tasks
                    .as_array()
                    .is_some_and(|tasks| {
                        tasks
                            .iter()
                            .any(|task| task.as_str() == Some(GRAPH_SHELL_TASK_ID))
                    })
                    .then_some(feature_id.clone())
            })
            .collect::<BTreeSet<_>>();
        if mapped_feature_ids != expected_feature_ids {
            return Err(
                "native spec mapping does not map exactly the shell catalogs to Task 17".to_owned(),
            );
        }

        Ok(json!({
            "name": "1193-feature-forward-reverse-trace-closure",
            "passed": true,
            "feature_count": closure_feature_ids.len(),
            "closure_digest": digest(SHELL_TRACE_CLOSURE),
            "mapping_digest": digest(NATIVE_SPEC_MAPPING),
        }))
    }

    fn component_surface_reconciliation() -> Result<Value, String> {
        let rows = csv_rows(COMPONENT_SURFACE_CSV)?;
        let disposition_rows = csv_rows(NATIVE_COMPONENT_DISPOSITIONS_CSV)?
            .into_iter()
            .map(|row| {
                let feature_id = field(&row, "feature_id")?.to_owned();
                Ok((feature_id, row))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let registrations = component_surface_registry()
            .map(|registration| (registration.feature_id, registration))
            .collect::<BTreeMap<_, _>>();
        if rows.len() != 805
            || disposition_rows.len() != rows.len()
            || registrations.len() != rows.len()
        {
            return Err(format!(
                "component surface count mismatch: source={} ledger={} registry={}",
                rows.len(),
                disposition_rows.len(),
                registrations.len()
            ));
        }
        let task_ids = TASKS
            .lines()
            .filter_map(|line| line.trim().strip_prefix("- _id: "))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        let mut canonical_dispositions = Vec::new();
        for row in &rows {
            let feature_id = field(row, "feature_id")?;
            let registration = registrations
                .get(feature_id)
                .ok_or_else(|| format!("component registry is missing `{feature_id}`"))?;
            let disposition = disposition_rows
                .get(feature_id)
                .ok_or_else(|| format!("component ledger is missing `{feature_id}`"))?;
            let kind = match registration.disposition {
                GeneratedComponentDisposition::Place => "place",
                GeneratedComponentDisposition::Defer => "defer",
            };
            let target = format!("{:?}", registration.placement);
            if registration.domain != field(row, "domain")?
                || kind != field(disposition, "disposition")?
                || target != field(disposition, "placement")?
                || registration.owner != field(disposition, "owner_task_id")?
            {
                return Err(format!(
                    "component surface `{feature_id}` does not match its generated authoritative disposition"
                ));
            }
            let prefix = format!("| `{feature_id}` |");
            let matches = PARITY_MATRIX
                .lines()
                .filter(|line| line.starts_with(&prefix))
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err(format!(
                    "component surface `{feature_id}` has {} parity rows",
                    matches.len()
                ));
            }
            let decision = matches[0]
                .rsplit('|')
                .nth(1)
                .map(str::trim)
                .ok_or_else(|| {
                    format!("component surface `{feature_id}` has no decision column")
                })?;
            let exact_decision = format!("{kind}:{target};owner:{}", registration.owner);
            if decision != exact_decision {
                return Err(format!(
                    "component surface `{feature_id}` parity decision `{decision}` does not equal `{exact_decision}`"
                ));
            }
            if registration.owner.is_empty() || !task_ids.contains(registration.owner) {
                return Err(format!(
                    "component surface `{feature_id}` has unknown owner `{}`",
                    registration.owner
                ));
            }
            canonical_dispositions.push(json!({
                "feature_id": feature_id,
                "kind": kind,
                "target": target,
                "owner": registration.owner,
            }));
        }
        let disposition_bytes = serde_json::to_vec(&canonical_dispositions)
            .map_err(|error| format!("serialize component dispositions: {error}"))?;
        Ok(json!({
            "name": "805-component-surface-placement-decisions",
            "passed": true,
            "row_count": rows.len(),
            "catalog_digest": digest(COMPONENT_SURFACE_CSV),
            "generated_ledger_digest": digest(NATIVE_COMPONENT_DISPOSITIONS_CSV),
            "matrix_digest": digest(PARITY_MATRIX),
            "disposition_row_count": canonical_dispositions.len(),
            "disposition_digest": format!("{:x}", Sha256::digest(disposition_bytes)),
            "generated_registry_row_count": registrations.len(),
            "heuristic_fallback_count": 0,
        }))
    }

    fn graph_fixture() -> Result<GraphWorkspaceModel, String> {
        let mut document = GraphDocument::default();
        document.document_identity = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1201);
        let bytes = document
            .to_workflow_bytes()
            .map_err(|error| format!("encode graph fixture: {error}"))?;
        let mut model = GraphWorkspaceModel::open(
            "Shell validation",
            "val-gpui-012-shell-fixture",
            WorkflowStorageProvider::Draft,
            bytes,
        )
        .map_err(|error| format!("open graph fixture: {error}"))?;
        for (identifier, x) in [("source", 100.0), ("target", 420.0)] {
            model
                .apply(GraphCommand::AddNode {
                    node: GraphNode::new(
                        GraphIdentifier::from(identifier),
                        "NativeShellFixture",
                        identifier,
                        GraphPoint { x, y: 100.0 },
                    ),
                    source: NodeCreationSource::Library,
                })
                .map_err(|error| format!("add `{identifier}`: {error}"))?;
        }
        Ok(model)
    }

    fn interaction_reconciliation(cx: &mut TestAppContext) -> Result<Value, String> {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let (item, cx) = cx.add_window_view(|_, cx| {
            GraphWorkspaceItem::new(
                graph_fixture().expect("validated graph fixture"),
                WeakEntity::new_invalid(),
                cx,
            )
        });
        item.update_in(cx, |item, window, cx| {
            item.focus_graph(window, cx);
            if !item.focus_handle(cx).is_focused(window) {
                return Err("graph did not receive focus".to_owned());
            }
            Ok(())
        })?;

        cx.dispatch_action(GraphSelectAll);
        let selected = item.read_with(cx, |item, _| {
            item.model()
                .selection()
                .map(|selection| selection.nodes.len())
        });
        if selected != Some(2) {
            return Err(format!("select-all selected {selected:?} nodes"));
        }
        cx.dispatch_action(GraphCopy);
        let clipboard = cx
            .read_from_clipboard()
            .and_then(|item| item.text())
            .ok_or_else(|| "copy action did not write graph text".to_owned())?;
        if !clipboard.starts_with(crate::GRAPH_CLIPBOARD_MEDIA_TYPE) {
            return Err("copy action wrote an untyped payload".to_owned());
        }

        let binding = KeyBinding::new("ctrl-b", ToggleSelectedBypass, Some(SIM_KEYMAP_CONTEXT));
        let predicate = binding
            .predicate()
            .ok_or_else(|| "graph keybinding has no context predicate".to_owned())?;
        let graph_context = KeyContext::parse(SIM_GRAPH_KEY_CONTEXT)
            .map_err(|error| format!("parse graph context: {error}"))?;
        let text_input_context = KeyContext::parse(SIM_TEXT_INPUT_KEY_CONTEXT)
            .map_err(|error| format!("parse text-input context: {error}"))?;
        if !predicate.eval(std::slice::from_ref(&graph_context))
            || predicate.eval(&[graph_context, text_input_context])
            || predicate.eval(&[KeyContext::default()])
        {
            return Err("graph keybinding predicate is not graph-exclusive".to_owned());
        }
        cx.cx.update(|cx| cx.bind_keys([binding]));
        let before_keybinding = item
            .read_with(cx, |item, _| item.model().encode())
            .map_err(|error| format!("encode pre-keybinding graph: {error}"))?;
        cx.simulate_keystrokes("ctrl-b");
        let after_keybinding = item
            .read_with(cx, |item, _| item.model().encode())
            .map_err(|error| format!("encode post-keybinding graph: {error}"))?;
        if before_keybinding == after_keybinding {
            return Err("scoped bypass keybinding produced no state transition".to_owned());
        }

        cx.dispatch_action(GroupSelectedNodes);
        let group_count = item.read_with(cx, |item, _| {
            item.model()
                .document()
                .and_then(|document| document.active_graph().ok())
                .map(|graph| graph.groups.len())
        });
        if group_count != Some(1) {
            return Err(format!("group action created {group_count:?} groups"));
        }
        cx.dispatch_action(ToggleMinimap);
        let minimap_visible = item.read_with(cx, |item, _| {
            item.model()
                .document()
                .and_then(|document| document.active_graph().ok())
                .is_some_and(|graph| graph.viewport.minimap_visible)
        });
        if !minimap_visible {
            return Err("minimap action did not update viewport state".to_owned());
        }

        cx.dispatch_action(QueuePrompt);
        let execution_error = item
            .read_with(cx, |item, _| item.model().last_error.clone())
            .ok_or_else(|| "execution action produced no visible error".to_owned())?;
        if !execution_error.contains("execution profile") {
            return Err("execution action did not expose its missing-profile rejection".to_owned());
        }
        let trace_length = item.read_with(cx, |item, _| item.shell_dispatch_trace_for_test().len());
        cx.dispatch_action(OpenWorkflow);
        let later_owned_action = item.read_with(cx, |item, _| {
            let trace = item.shell_dispatch_trace_for_test();
            (
                trace.len() == trace_length + 1
                    && trace
                        .last()
                        .is_some_and(|command_id| command_id == "Sim.OpenWorkflow"),
                item.model().last_error.clone(),
            )
        });
        if !later_owned_action.0 {
            return Err(
                "later-owned Open Workflow action did not reach the shell registry".to_owned(),
            );
        }
        if !later_owned_action.1.is_some_and(|error| {
            error.contains("Sim.OpenWorkflow") && error.contains("implementation owner")
        }) {
            return Err(
                "later-owned Open Workflow action did not produce visible owner feedback"
                    .to_owned(),
            );
        }
        let typed_outcomes = item.update(cx, |item, cx| {
            [
                item.dispatch_shell_command("Sim.QueuePrompt", cx),
                item.dispatch_shell_command("Sim.Graph.ConvertToSubgraph", cx),
                item.dispatch_shell_command("Sim.OpenWorkflow", cx),
                item.dispatch_shell_command("Sim.ToggleAssetAPI", cx),
                item.dispatch_shell_command(
                    "Sim.Manager.CustomNodesManager.ShowLegacyCustomNodesMenu",
                    cx,
                ),
                item.dispatch_shell_command("Sim.DoesNotExist", cx),
            ]
        });
        if !matches!(
            &typed_outcomes[0],
            CommandDispatchOutcome::Rejected { command_id, error }
                if command_id == "Sim.QueuePrompt" && error.contains("execution profile")
        ) || !matches!(
            typed_outcomes[1],
            CommandDispatchOutcome::RequiresInput { .. }
        ) || !matches!(
            &typed_outcomes[2],
            CommandDispatchOutcome::LaterOwned { command_id, .. }
                if command_id == "Sim.OpenWorkflow"
        ) || !matches!(typed_outcomes[3], CommandDispatchOutcome::Gated { .. })
            || !matches!(typed_outcomes[4], CommandDispatchOutcome::Legacy { .. })
            || !matches!(typed_outcomes[5], CommandDispatchOutcome::Unknown { .. })
        {
            return Err(format!("typed shell outcome mismatch: {typed_outcomes:?}"));
        }

        let executable_registrations = command_registry()
            .filter(|registration| registration.status == CommandNativeStatus::Executable)
            .collect::<Vec<_>>();
        for registration in &executable_registrations {
            let native_action = registration.gpui_action.ok_or_else(|| {
                format!(
                    "executable command `{}` has no GPUI action",
                    registration.command_id
                )
            })?;
            let action_name = native_action.name();
            let action = cx
                .cx
                .update(|cx| cx.build_action(action_name, None))
                .map_err(|error| format!("build `{action_name}`: {error}"))?;
            let trace_length =
                item.read_with(cx, |item, _| item.shell_dispatch_trace_for_test().len());
            cx.update(|window, cx| window.dispatch_action(action, cx));
            let reached = item.read_with(cx, |item, _| {
                let trace = item.shell_dispatch_trace_for_test();
                trace.len() == trace_length + 1
                    && trace
                        .last()
                        .is_some_and(|command_id| command_id == registration.command_id)
            });
            if !reached {
                return Err(format!(
                    "registered action `{action_name}` did not reach command `{}`",
                    registration.command_id
                ));
            }
            let semantic_outcome = item.update(cx, |item, cx| {
                item.dispatch_shell_command(registration.command_id, cx)
            });
            match &semantic_outcome {
                CommandDispatchOutcome::Executed { command_id }
                    if command_id == registration.command_id => {}
                CommandDispatchOutcome::Rejected { command_id, error }
                    if command_id == registration.command_id && !error.is_empty() => {}
                CommandDispatchOutcome::RequiresInput { command_id, input }
                    if command_id == registration.command_id
                        && registration.command_id == "Sim.PublishSubgraph"
                        && !input.is_empty() => {}
                _ => {
                    return Err(format!(
                        "executable command `{}` returned non-executable semantic outcome {semantic_outcome:?}",
                        registration.command_id
                    ));
                }
            }
        }
        let final_state = item
            .read_with(cx, |item, _| item.model().encode())
            .map_err(|error| format!("encode final shell state: {error}"))?;
        Ok(json!({
            "name": "real-gpui-action-focus-clipboard-and-state-transitions",
            "passed": true,
            "selected_node_count": selected,
            "group_count": group_count,
            "typed_outcomes": typed_outcomes,
            "executable_reachability_count": executable_registrations.len(),
            "clipboard_digest": digest(&clipboard),
            "pre_keybinding_digest": format!("{:x}", Sha256::digest(before_keybinding)),
            "post_state_digest": format!("{:x}", Sha256::digest(final_state)),
        }))
    }

    fn write_artifact(cases: Vec<Value>) -> Result<(), String> {
        if cases.iter().any(|case| case["passed"] != true) {
            return Err("VAL-GPUI-012 contains a failing case".to_owned());
        }
        let artifact = json!({
            "validation_id": "VAL-GPUI-012",
            "environment": {
                "backend": "gpui-test",
                "platform": "mock-window",
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "feature": "test-support",
                "scheduler_seed": 16012,
                "iterations": std::env::var("ITERATIONS").unwrap_or_else(|_| "1".to_owned()),
            },
            "fixture_digests": {
                "commands": digest(COMMAND_CSV),
                "native_command_dispositions": digest(NATIVE_COMMAND_DISPOSITIONS_CSV),
                "keybindings": digest(KEYBINDING_CSV),
                "menus": digest(MENU_CSV),
                "native_menu_dispositions": digest(NATIVE_MENU_DISPOSITIONS_CSV),
                "component_surfaces": digest(COMPONENT_SURFACE_CSV),
                "native_component_dispositions": digest(NATIVE_COMPONENT_DISPOSITIONS_CSV),
                "native_spec_mapping": digest(NATIVE_SPEC_MAPPING),
                "shell_trace_closure": digest(SHELL_TRACE_CLOSURE),
                "parity_matrix": digest(PARITY_MATRIX),
                "default_keymap": digest(DEFAULT_KEYMAP),
            },
            "catalog_counts": {
                "commands": 118,
                "keybindings": 34,
                "menus": 236,
                "component_surfaces": 805,
            },
            "cases": cases,
            "skipped": [],
        });
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"))
            .join("comfy-parity");
        fs::create_dir_all(&target)
            .map_err(|error| format!("create artifact directory: {error}"))?;
        fs::write(
            target.join("val-gpui-012.json"),
            serde_json::to_vec_pretty(&artifact)
                .map_err(|error| format!("serialize artifact: {error}"))?,
        )
        .map_err(|error| format!("write artifact: {error}"))
    }

    #[test]
    fn graph_context_registry_separates_actions_from_infrastructure() {
        let registrations = graph_context_registry()
            .collect::<Result<Vec<_>, _>>()
            .expect("resolve generated graph context registry");
        assert_eq!(registrations.len(), 63);
        assert_eq!(
            registrations
                .iter()
                .filter(|registration| registration.context_action.is_some())
                .count(),
            55
        );
        let infrastructure = registrations
            .iter()
            .filter_map(|registration| {
                registration
                    .context_infrastructure
                    .map(|binding| (binding, registration))
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(infrastructure.len(), 8);
        assert_eq!(
            infrastructure.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([
                GeneratedGraphContextInfrastructure::CommandAdapter,
                GeneratedGraphContextInfrastructure::ContextMenuConverter,
                GeneratedGraphContextInfrastructure::CoreMenuLoader,
                GeneratedGraphContextInfrastructure::DropdownRenderer,
                GeneratedGraphContextInfrastructure::MergedMoreOptions,
                GeneratedGraphContextInfrastructure::NativeContextRenderer,
                GeneratedGraphContextInfrastructure::RegisterMenuGroup,
                GeneratedGraphContextInfrastructure::TranslatedRegistryItems,
            ])
        );
        for registration in infrastructure.values() {
            assert_eq!(registration.context_action, None);
            assert_eq!(
                registration.context_surface,
                Some(GeneratedGraphContextSurface::Infrastructure)
            );
            assert_eq!(
                registration.availability,
                MenuAvailability::InfrastructureOnly
            );
            assert_eq!(
                registration.status,
                CommandNativeStatus::Infrastructure {
                    owner: GRAPH_CONTEXT_MENU_OWNER,
                }
            );
            assert_eq!(registration.native_action, None);
            assert_eq!(registration.command_id, None);
        }
    }

    #[gpui::test(seed = 16012)]
    fn val_gpui_012(cx: &mut TestAppContext) {
        let cases = vec![
            registry_reconciliation().expect("reconcile command catalog"),
            keybinding_reconciliation(cx).expect("reconcile keybinding catalog"),
            menu_reconciliation().expect("reconcile menu catalog"),
            component_surface_reconciliation().expect("reconcile component surfaces"),
            trace_closure_reconciliation().expect("reconcile shell trace closure"),
            interaction_reconciliation(cx).expect("exercise graph shell interactions"),
        ];
        write_artifact(cases).expect("write VAL-GPUI-012 artifact");
    }
}
