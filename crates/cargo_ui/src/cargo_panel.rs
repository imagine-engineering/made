use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::Context as _;
use db::kvp::KeyValueStore;
use feature_flags::{
    FeatureFlagAppExt as _, FeatureFlagStore, RustNamedCargoPresetsExperimentalFeatureFlag,
};
use gpui::{
    Action, Anchor, App, AsyncWindowContext, Context, DismissEvent, Entity, EventEmitter,
    FocusHandle, Focusable, IntoElement, Pixels, Point, Render, ScrollStrategy, Subscription,
    TaskExt as _, WeakEntity, Window, actions, anchored, deferred, div,
};
use language_tools::language_tool_tree::{
    self, LanguageToolNode, LanguageToolNodeId, LanguageToolProviderStatus, LanguageToolSnapshot,
    LanguageToolTreeHost, LanguageToolTreeStatus, language_tool_tree, status_message,
};
use project::{
    Project, ProjectPath,
    cargo_workspace::{
        CargoConfigurationDiagnosticCategory, CargoDependencyKind, CargoDependencySourceKind,
        CargoFeatureEnabled, CargoHostCompilerStatus, CargoPackageModel, CargoProfileOrigin,
        CargoSnapshotCompleteness, CargoTargetConfiguration, CargoTargetKind, CargoToolchainFormat,
        CargoWorkspaceConfiguration, CargoWorkspaceSnapshot,
    },
    cargo_workspace_store::{CargoWorkspaceRemoteError, CargoWorkspaceRemoteErrorKind},
};
use settings::{DockSide, Settings, SettingsLocation, SettingsStore};
use ui::{ContextMenu, IconName, Tooltip, prelude::*};
use util::{ResultExt as _, rel_path::RelPath};
use workspace::{
    Panel, Workspace,
    dock::{DockPosition, PanelEvent},
    workspace_scoped_state_key,
};

use crate::{
    BenchSelected, BuildSelected, CargoAction, CargoActionNodeKind, CargoActionRuntime,
    CargoActionSelection, CargoActionTargetKind, CargoPanelSettings, CargoPreset,
    CargoPresetEditor, CargoPresetSaveScope, CargoPresetScope, CargoPresetSettings,
    CargoPresetValidationContext, CargoPresetWorkspaceState, CargoSafeSelectionState,
    CheckSelected, CleanSelected, ClippySelected, DebugSelected, DocSelected, FmtSelected,
    ProfileSelected, RunSelected, RunWithCoverageSelected, TestSelected, TreeSelected,
    WorkspaceCargoActionDispatcher, apply_preset_save, cargo_action_availability_with_preset,
    dispatch_cargo_action, plan_cargo_action_with_confirmation, recover_workspace_state,
};

actions!(cargo_panel, [ToggleCargoPanel, SelectPreset]);

const CARGO_PANEL_KEY: &str = "CargoPanel";
const CARGO_PRESET_STATE_KEY: &str = "cargo-preset-state-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoActiveConfiguration {
    pub scope: String,
    pub profile: Option<String>,
    pub native_profiling: bool,
    pub selected_features: Vec<String>,
    pub default_features: Option<bool>,
    pub explicit_target: Option<String>,
    pub target_selector: Option<String>,
    pub environment_keys: Vec<String>,
    pub notice: Option<String>,
}

impl Default for CargoActiveConfiguration {
    fn default() -> Self {
        Self {
            scope: "selected workspace or package".to_string(),
            profile: None,
            native_profiling: false,
            selected_features: Vec::new(),
            default_features: None,
            explicit_target: None,
            target_selector: None,
            environment_keys: Vec::new(),
            notice: None,
        }
    }
}

pub struct CargoTreeProvider {
    navigation: HashMap<LanguageToolNodeId, ProjectPath>,
    action_selections: HashMap<LanguageToolNodeId, CargoActionSelection>,
}

impl CargoTreeProvider {
    pub fn project(snapshot: &CargoWorkspaceSnapshot) -> (LanguageToolSnapshot, Self) {
        Self::project_with_configuration(snapshot, &CargoActiveConfiguration::default())
    }

    pub fn project_with_configuration(
        snapshot: &CargoWorkspaceSnapshot,
        active_configuration: &CargoActiveConfiguration,
    ) -> (LanguageToolSnapshot, Self) {
        let mut navigation = HashMap::new();
        let mut action_selections = HashMap::new();
        let mut roots = Vec::new();
        for workspace in &snapshot.workspaces {
            let workspace_id = id(&format!(
                "workspace:{}:{}",
                workspace.key.root.worktree_id.to_proto(),
                workspace.key.root.path.as_unix_str()
            ));
            if let Some(path) = workspace.root_manifest.clone() {
                navigation.insert(workspace_id.clone(), path);
            }
            if let Some(workspace_manifest) = workspace.root_manifest.clone() {
                action_selections.insert(
                    workspace_id.clone(),
                    CargoActionSelection {
                        node_kind: CargoActionNodeKind::Workspace,
                        worktree_id: workspace_manifest.worktree_id,
                        workspace_name: workspace.display_name.clone(),
                        workspace_manifest,
                        package_name: None,
                        package_manifest: None,
                        target: None,
                        has_bench_targets: workspace.members.iter().any(|package| {
                            package
                                .targets
                                .iter()
                                .any(|target| target.kind == CargoTargetKind::Bench)
                        }),
                    },
                );
            }
            let mut members = vec![configuration_node(
                &workspace_id,
                &workspace.configuration,
                workspace.root_manifest.as_ref(),
                active_configuration,
                &mut navigation,
            )];
            for package in &workspace.members {
                if let Some(workspace_manifest) = workspace.root_manifest.as_ref() {
                    members.push(package_node(
                        &workspace_id,
                        &workspace.display_name,
                        workspace_manifest,
                        package,
                        &mut navigation,
                        &mut action_selections,
                    ));
                } else {
                    members.push(package_node(
                        &workspace_id,
                        &workspace.display_name,
                        &package.manifest_path,
                        package,
                        &mut navigation,
                        &mut action_selections,
                    ));
                }
            }
            roots.push(LanguageToolNode {
                id: workspace_id,
                label: workspace.display_name.clone(),
                secondary_label: Some(workspace.key.root.path.as_unix_str().to_string()),
                icon: Some(IconName::FileRust),
                accessibility_label: format!("Cargo workspace {}", workspace.display_name),
                children: members,
                enabled: workspace.root_manifest.is_some(),
                activation_label: workspace
                    .root_manifest
                    .as_ref()
                    .map(|_| "Open Cargo.toml".to_string()),
            });
        }
        for failure in &snapshot.failures {
            let failure_label = match failure.category {
                project::cargo_workspace::CargoWorkspaceErrorCategory::Restricted => {
                    "Cargo access restricted"
                }
                project::cargo_workspace::CargoWorkspaceErrorCategory::CargoNotFound => {
                    "Cargo not found"
                }
                project::cargo_workspace::CargoWorkspaceErrorCategory::UnsupportedMetadata => {
                    "Unsupported Cargo metadata"
                }
                project::cargo_workspace::CargoWorkspaceErrorCategory::Disconnected => {
                    "Cargo host disconnected"
                }
                project::cargo_workspace::CargoWorkspaceErrorCategory::Cancelled => {
                    "Cargo refresh cancelled"
                }
                project::cargo_workspace::CargoWorkspaceErrorCategory::CargoFailed
                | project::cargo_workspace::CargoWorkspaceErrorCategory::InvalidMetadata
                | project::cargo_workspace::CargoWorkspaceErrorCategory::Internal => {
                    "Cargo metadata error"
                }
            };
            roots.push(LanguageToolNode {
                id: id(&format!(
                    "failure:{}:{}",
                    failure.manifest_path.worktree_id.to_proto(),
                    failure.manifest_path.path.as_unix_str()
                )),
                label: failure_label.to_string(),
                secondary_label: Some(format!(
                    "{} · {}",
                    failure.manifest_path.path.as_unix_str(),
                    failure.message
                )),
                icon: Some(IconName::CircleHelp),
                accessibility_label: format!("Cargo metadata error: {}", failure.message),
                children: Vec::new(),
                enabled: false,
                activation_label: None,
            });
        }
        let status = if roots.is_empty() && snapshot.failures.is_empty() {
            LanguageToolProviderStatus::Empty(
                "No Cargo.toml files were found in visible worktrees.".to_string(),
            )
        } else if snapshot.workspaces.is_empty()
            && !snapshot.failures.is_empty()
            && snapshot.failures.iter().all(|failure| {
                failure.category
                    == project::cargo_workspace::CargoWorkspaceErrorCategory::Restricted
            })
        {
            LanguageToolProviderStatus::Restricted(
                "Cargo metadata is disabled for restricted worktrees. Trust the worktree and refresh."
                    .to_string(),
            )
        } else if snapshot.completeness == CargoSnapshotCompleteness::Partial {
            let message = if snapshot.failures.is_empty()
                && snapshot.workspaces.iter().any(|workspace| {
                    workspace.configuration.completeness
                        == project::cargo_workspace::CargoConfigurationCompleteness::Partial
                }) {
                "Cargo metadata loaded, but some configuration facts are unavailable. Refresh to retry."
            } else {
                "Some Cargo workspaces could not be loaded. Refresh to retry."
            };
            LanguageToolProviderStatus::Partial(message.to_string())
        } else {
            LanguageToolProviderStatus::Current
        };
        (
            LanguageToolSnapshot { roots, status },
            Self {
                navigation,
                action_selections,
            },
        )
    }

    pub fn navigation(&self, id: &LanguageToolNodeId) -> Option<&ProjectPath> {
        self.navigation.get(id)
    }

    pub fn action_selection(&self, id: &LanguageToolNodeId) -> Option<&CargoActionSelection> {
        self.action_selections.get(id)
    }
}

fn configuration_node(
    workspace_id: &LanguageToolNodeId,
    configuration: &CargoWorkspaceConfiguration,
    root_manifest: Option<&ProjectPath>,
    active: &CargoActiveConfiguration,
    navigation: &mut HashMap<LanguageToolNodeId, ProjectPath>,
) -> LanguageToolNode {
    let configuration_id = id(&format!("{}:section:configuration", workspace_id.0));
    let active_id = id(&format!("{}:active", configuration_id.0));
    let mut environment_keys = active.environment_keys.clone();
    environment_keys.sort();
    environment_keys.dedup();
    let selected_features = if active.selected_features.is_empty() {
        "Cargo default".to_string()
    } else {
        active.selected_features.join(", ")
    };
    let environment_key_summary = if environment_keys.is_empty() {
        "none".to_string()
    } else {
        environment_keys.join(", ")
    };
    let active_children = vec![
        configuration_leaf(
            &active_id,
            "scope",
            "Scope",
            &active.scope,
            None,
            navigation,
        ),
        configuration_leaf(
            &active_id,
            "profile",
            "Profile",
            active.profile.as_deref().unwrap_or("Cargo default"),
            None,
            navigation,
        ),
        configuration_leaf(
            &active_id,
            "features",
            "Features",
            &selected_features,
            None,
            navigation,
        ),
        configuration_leaf(
            &active_id,
            "default-features",
            "Default features",
            match active.default_features {
                Some(true) => "enabled",
                Some(false) => "disabled",
                None => "Cargo default",
            },
            None,
            navigation,
        ),
        configuration_leaf(
            &active_id,
            "explicit-target",
            "Explicit target",
            active.explicit_target.as_deref().unwrap_or("none"),
            None,
            navigation,
        ),
        configuration_leaf(
            &active_id,
            "target-selector",
            "Target selector",
            active.target_selector.as_deref().unwrap_or("Cargo default"),
            None,
            navigation,
        ),
        configuration_leaf(
            &active_id,
            "environment-keys",
            "Environment keys",
            &environment_key_summary,
            None,
            navigation,
        ),
    ];
    let mut active_children = active_children;
    if active.native_profiling {
        active_children.push(configuration_leaf(
            &active_id,
            "profiling-mode",
            "Profiling mode",
            "macOS native",
            None,
            navigation,
        ));
    }
    if let Some(notice) = active.notice.as_deref() {
        active_children.push(configuration_leaf(
            &active_id,
            "preset-notice",
            "Preset notice",
            notice,
            None,
            navigation,
        ));
    }
    let mut active_summary = [
        active
            .profile
            .as_ref()
            .map(|profile| format!("profile {profile}"))
            .unwrap_or_else(|| "Cargo default profile".to_string()),
        active
            .explicit_target
            .as_ref()
            .map(|target| format!("target {target}"))
            .unwrap_or_else(|| "no explicit target".to_string()),
    ]
    .join(" · ");
    if active.native_profiling {
        active_summary.push_str(" · macOS native profiling");
    }
    let mut children = vec![LanguageToolNode {
        id: active_id,
        label: "Active configuration".to_string(),
        secondary_label: Some(active_summary.clone()),
        icon: Some(IconName::Settings),
        accessibility_label: format!("Cargo active configuration, {active_summary}"),
        children: active_children,
        enabled: false,
        activation_label: None,
    }];

    let profiles_id = id(&format!("{}:profiles", configuration_id.0));
    let profile_children = configuration
        .profiles
        .iter()
        .map(|profile| {
            configuration_leaf(
                &profiles_id,
                &profile.name,
                &profile.name,
                match profile.origin {
                    CargoProfileOrigin::Implicit => "implicit Cargo profile",
                    CargoProfileOrigin::Declared => "declared in Cargo.toml",
                },
                root_manifest.cloned(),
                navigation,
            )
        })
        .collect::<Vec<_>>();
    children.push(LanguageToolNode {
        id: profiles_id,
        label: "Profiles".to_string(),
        secondary_label: Some(configuration.profiles.len().to_string()),
        icon: None,
        accessibility_label: format!("Cargo profiles, {} entries", configuration.profiles.len()),
        children: profile_children,
        enabled: false,
        activation_label: None,
    });

    let toolchain_id = id(&format!("{}:toolchain", configuration_id.0));
    if let Some(toolchain) = &configuration.declared_toolchain {
        let toolchain_format = match toolchain.format {
            CargoToolchainFormat::Toml => "rust-toolchain.toml",
            CargoToolchainFormat::Legacy => "legacy rust-toolchain",
        };
        let components = if toolchain.components.is_empty() {
            "none declared".to_string()
        } else {
            toolchain.components.join(", ")
        };
        let targets = if toolchain.targets.is_empty() {
            "none declared".to_string()
        } else {
            toolchain.targets.join(", ")
        };
        let mut toolchain_children = vec![configuration_leaf(
            &toolchain_id,
            "channel",
            "Channel",
            toolchain.channel.as_deref().unwrap_or("not declared"),
            Some(toolchain.source_path.clone()),
            navigation,
        )];
        toolchain_children.push(configuration_leaf(
            &toolchain_id,
            "components",
            "Components",
            &components,
            Some(toolchain.source_path.clone()),
            navigation,
        ));
        toolchain_children.push(configuration_leaf(
            &toolchain_id,
            "targets",
            "Declared targets",
            &targets,
            Some(toolchain.source_path.clone()),
            navigation,
        ));
        children.push(LanguageToolNode {
            id: toolchain_id,
            label: "Declared toolchain".to_string(),
            secondary_label: Some(
                toolchain
                    .channel
                    .clone()
                    .unwrap_or_else(|| toolchain_format.to_string()),
            ),
            icon: Some(IconName::Settings),
            accessibility_label: format!("Declared Cargo toolchain, {toolchain_format}"),
            children: toolchain_children,
            enabled: true,
            activation_label: Some("Open toolchain declaration".to_string()),
        });
        navigation.insert(
            id(&format!("{}:toolchain", configuration_id.0)),
            toolchain.source_path.clone(),
        );
    } else {
        children.push(configuration_leaf(
            &configuration_id,
            "toolchain",
            "Declared toolchain",
            "none",
            None,
            navigation,
        ));
    }

    let host_status = match configuration.host_compiler.status {
        CargoHostCompilerStatus::Available => match (
            configuration.host_compiler.release.as_deref(),
            configuration.host_compiler.host_target.as_deref(),
        ) {
            (Some(release), Some(target)) => format!("rustc {release} · {target}"),
            (Some(release), None) => format!("rustc {release} · host target unknown"),
            (None, Some(target)) => format!("release unknown · {target}"),
            (None, None) => "available · details unknown".to_string(),
        },
        CargoHostCompilerStatus::Restricted => "restricted".to_string(),
        CargoHostCompilerStatus::Missing => "rustc not found".to_string(),
        CargoHostCompilerStatus::Failed => "probe failed".to_string(),
        CargoHostCompilerStatus::Unknown => "unknown".to_string(),
    };
    let host_status = if configuration.host_compiler.stale {
        format!("{host_status} · stale")
    } else {
        host_status
    };
    children.push(configuration_leaf(
        &configuration_id,
        "host-compiler",
        "Host compiler",
        &host_status,
        None,
        navigation,
    ));
    let cargo_target = match configuration.cargo_target {
        CargoTargetConfiguration::UnresolvedCargoDefault => "unresolved Cargo default",
    };
    children.push(configuration_leaf(
        &configuration_id,
        "cargo-target",
        "Cargo target resolution",
        cargo_target,
        None,
        navigation,
    ));

    if !configuration.diagnostics.is_empty() {
        let diagnostics_id = id(&format!("{}:diagnostics", configuration_id.0));
        let diagnostics = configuration
            .diagnostics
            .iter()
            .enumerate()
            .map(|(index, diagnostic)| {
                configuration_leaf(
                    &diagnostics_id,
                    &format!(
                        "{}-{index}",
                        configuration_diagnostic_label(diagnostic.category)
                    ),
                    configuration_diagnostic_label(diagnostic.category),
                    &diagnostic.message,
                    diagnostic.source_path.clone(),
                    navigation,
                )
            })
            .collect();
        children.push(LanguageToolNode {
            id: diagnostics_id,
            label: "Configuration diagnostics".to_string(),
            secondary_label: Some(configuration.diagnostics.len().to_string()),
            icon: Some(IconName::CircleHelp),
            accessibility_label: format!(
                "Cargo configuration diagnostics, {} entries",
                configuration.diagnostics.len()
            ),
            children: diagnostics,
            enabled: false,
            activation_label: None,
        });
    }

    LanguageToolNode {
        id: configuration_id,
        label: "Configuration".to_string(),
        secondary_label: Some(active_summary),
        icon: Some(IconName::Settings),
        accessibility_label: "Cargo workspace configuration".to_string(),
        children,
        enabled: false,
        activation_label: None,
    }
}

fn configuration_leaf(
    parent: &LanguageToolNodeId,
    discriminator: &str,
    label: &str,
    value: &str,
    navigation_path: Option<ProjectPath>,
    navigation: &mut HashMap<LanguageToolNodeId, ProjectPath>,
) -> LanguageToolNode {
    let node_id = id(&format!("{}:{discriminator}", parent.0));
    if let Some(path) = navigation_path.clone() {
        navigation.insert(node_id.clone(), path);
    }
    LanguageToolNode {
        id: node_id,
        label: label.to_string(),
        secondary_label: Some(value.to_string()),
        icon: None,
        accessibility_label: format!("Cargo configuration {label}, {value}"),
        children: Vec::new(),
        enabled: navigation_path.is_some(),
        activation_label: navigation_path.map(|_| "Open configuration source".to_string()),
    }
}

fn configuration_diagnostic_label(category: CargoConfigurationDiagnosticCategory) -> &'static str {
    match category {
        CargoConfigurationDiagnosticCategory::Manifest => "Manifest",
        CargoConfigurationDiagnosticCategory::Toolchain => "Toolchain",
        CargoConfigurationDiagnosticCategory::CompilerProbe => "Compiler probe",
    }
}

fn package_node(
    workspace_id: &LanguageToolNodeId,
    workspace_name: &str,
    workspace_manifest: &ProjectPath,
    package: &CargoPackageModel,
    navigation: &mut HashMap<LanguageToolNodeId, ProjectPath>,
    action_selections: &mut HashMap<LanguageToolNodeId, CargoActionSelection>,
) -> LanguageToolNode {
    let package_id = id(&format!(
        "{}:member:{}",
        workspace_id.0,
        package.manifest_path.path.as_unix_str()
    ));
    navigation.insert(package_id.clone(), package.manifest_path.clone());
    let has_bench_targets = package
        .targets
        .iter()
        .any(|target| target.kind == CargoTargetKind::Bench);
    action_selections.insert(
        package_id.clone(),
        CargoActionSelection {
            node_kind: CargoActionNodeKind::Package,
            worktree_id: package.manifest_path.worktree_id,
            workspace_name: workspace_name.to_string(),
            workspace_manifest: workspace_manifest.clone(),
            package_name: Some(package.name.clone()),
            package_manifest: Some(package.manifest_path.clone()),
            target: None,
            has_bench_targets,
        },
    );
    let mut sections = Vec::new();
    if !package.targets.is_empty() {
        let children = package
            .targets
            .iter()
            .map(|target| {
                let target_id = id(&format!(
                    "{}:target:{:?}:{}:{}",
                    package_id.0,
                    target.kind,
                    target.name,
                    target.source_display_path.as_deref().unwrap_or("")
                ));
                if let Some(path) = target.source_path.clone() {
                    navigation.insert(target_id.clone(), path);
                }
                action_selections.insert(
                    target_id.clone(),
                    CargoActionSelection {
                        node_kind: CargoActionNodeKind::Target(action_target_kind(&target.kind)),
                        worktree_id: package.manifest_path.worktree_id,
                        workspace_name: workspace_name.to_string(),
                        workspace_manifest: workspace_manifest.clone(),
                        package_name: Some(package.name.clone()),
                        package_manifest: Some(package.manifest_path.clone()),
                        target: target_selector(&target.kind, &target.name),
                        has_bench_targets: target.kind == CargoTargetKind::Bench,
                    },
                );
                LanguageToolNode {
                    id: target_id,
                    label: format!("{} {}", target_kind_label(&target.kind), target.name),
                    secondary_label: Some(
                        [
                            target.source_display_path.clone(),
                            (!target.crate_types.is_empty()).then(|| target.crate_types.join(", ")),
                            (!target.required_features.is_empty()).then(|| {
                                format!("requires {}", target.required_features.join(", "))
                            }),
                            Some(format!("edition {}", target.edition)),
                        ]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                        .join(" · "),
                    ),
                    icon: Some(IconName::FileRust),
                    accessibility_label: format!(
                        "Cargo target {} {}",
                        target_kind_label(&target.kind),
                        target.name
                    ),
                    children: Vec::new(),
                    enabled: target.source_path.is_some(),
                    activation_label: target
                        .source_path
                        .as_ref()
                        .map(|_| "Open target source".to_string()),
                }
            })
            .collect();
        sections.push(section(&package_id, "targets", "Targets", children));
    }
    if !package.features.is_empty() {
        let children = package
            .features
            .iter()
            .map(|feature| {
                let feature_id = id(&format!("{}:feature:{}", package_id.0, feature.name));
                navigation.insert(feature_id.clone(), package.manifest_path.clone());
                let enabled = match feature.enabled {
                    CargoFeatureEnabled::Enabled if feature.defined => "enabled",
                    CargoFeatureEnabled::Enabled => "enabled implicit",
                    CargoFeatureEnabled::Disabled => "disabled",
                    CargoFeatureEnabled::Unknown => "unknown",
                };
                LanguageToolNode {
                    id: feature_id,
                    label: feature.name.clone(),
                    secondary_label: Some(enabled.to_string()),
                    icon: None,
                    accessibility_label: format!("Cargo feature {}, {enabled}", feature.name),
                    children: Vec::new(),
                    enabled: true,
                    activation_label: Some("Open Cargo.toml".to_string()),
                }
            })
            .collect();
        sections.push(section(
            &package_id,
            "features",
            "Features — Cargo default resolution",
            children,
        ));
    }
    if !package.dependencies.is_empty() {
        let mut groups = Vec::new();
        for (kind, label) in [
            (CargoDependencyKind::Normal, "Normal"),
            (CargoDependencyKind::Development, "Development"),
            (CargoDependencyKind::Build, "Build"),
            (CargoDependencyKind::Unknown, "Other"),
        ] {
            let children = package
                .dependencies
                .iter()
                .enumerate()
                .filter(|(_, dependency)| dependency.kind == kind)
                .map(|(index, dependency)| {
                    let display_name = dependency
                        .rename
                        .as_deref()
                        .unwrap_or(&dependency.declaration_name);
                    let dependency_id = id(&format!(
                        "{}:dependency:{kind:?}:{display_name}:{}:{}:{index}",
                        package_id.0,
                        dependency.target.as_deref().unwrap_or(""),
                        dependency.version_requirement
                    ));
                    let navigation_path = dependency
                        .resolved_workspace_member
                        .clone()
                        .or_else(|| dependency.local_manifest.clone())
                        .unwrap_or_else(|| package.manifest_path.clone());
                    navigation.insert(dependency_id.clone(), navigation_path);
                    let mut annotations = vec![dependency.version_requirement.clone()];
                    annotations.push(
                        match dependency.source_kind {
                            CargoDependencySourceKind::Path => "path",
                            CargoDependencySourceKind::Registry => "registry",
                            CargoDependencySourceKind::Git => "git",
                            CargoDependencySourceKind::Other => "other",
                        }
                        .to_string(),
                    );
                    if dependency.optional {
                        annotations.push("optional".to_string());
                    }
                    if !dependency.uses_default_features {
                        annotations.push("default features off".to_string());
                    }
                    if !dependency.requested_features.is_empty() {
                        annotations.push(format!(
                            "features {}",
                            dependency.requested_features.join(", ")
                        ));
                    }
                    if let Some(target) = &dependency.target {
                        annotations.push(target.clone());
                    }
                    if let Some(version) = &dependency.resolved_version {
                        annotations.push(format!("resolved {version}"));
                    }
                    let insight = crate::cargo_dependency_insight(&dependency_id, dependency);
                    for (node_id, path) in insight.navigation {
                        navigation.insert(node_id, path);
                    }
                    LanguageToolNode {
                        id: dependency_id,
                        label: display_name.to_string(),
                        secondary_label: Some(annotations.join(" · ")),
                        icon: None,
                        accessibility_label: format!("Cargo dependency {display_name}"),
                        children: insight.children,
                        enabled: true,
                        activation_label: Some("Open relevant Cargo.toml".to_string()),
                    }
                })
                .collect::<Vec<_>>();
            if !children.is_empty() {
                groups.push(section(
                    &package_id,
                    &format!("dependency-{label}"),
                    label,
                    children,
                ));
            }
        }
        sections.push(section(&package_id, "dependencies", "Dependencies", groups));
    }
    LanguageToolNode {
        id: package_id,
        label: format!("{} {}", package.name, package.version),
        secondary_label: Some(package.manifest_path.path.as_unix_str().to_string()),
        icon: Some(IconName::Box),
        accessibility_label: format!("Cargo package {} {}", package.name, package.version),
        children: sections,
        enabled: true,
        activation_label: Some("Open Cargo.toml".to_string()),
    }
}

fn section(
    parent: &LanguageToolNodeId,
    discriminator: &str,
    label: &str,
    children: Vec<LanguageToolNode>,
) -> LanguageToolNode {
    LanguageToolNode {
        id: id(&format!("{}:section:{discriminator}", parent.0)),
        label: label.to_string(),
        secondary_label: None,
        icon: None,
        accessibility_label: label.to_string(),
        children,
        enabled: false,
        activation_label: None,
    }
}

fn target_kind_label(kind: &CargoTargetKind) -> &str {
    match kind {
        CargoTargetKind::Library => "lib",
        CargoTargetKind::Binary => "bin",
        CargoTargetKind::Example => "example",
        CargoTargetKind::Test => "test",
        CargoTargetKind::Bench => "bench",
        CargoTargetKind::BuildScript => "build script",
        CargoTargetKind::Other(value) => value,
    }
}

fn action_target_kind(kind: &CargoTargetKind) -> CargoActionTargetKind {
    match kind {
        CargoTargetKind::Library => CargoActionTargetKind::Library,
        CargoTargetKind::Binary => CargoActionTargetKind::Binary,
        CargoTargetKind::Example => CargoActionTargetKind::Example,
        CargoTargetKind::Test => CargoActionTargetKind::Test,
        CargoTargetKind::Bench => CargoActionTargetKind::Bench,
        CargoTargetKind::BuildScript => CargoActionTargetKind::BuildScript,
        CargoTargetKind::Other(_) => CargoActionTargetKind::Other,
    }
}

fn target_selector(kind: &CargoTargetKind, name: &str) -> Option<crate::CargoTargetSelector> {
    match kind {
        CargoTargetKind::Library => Some(crate::CargoTargetSelector::Library),
        CargoTargetKind::Binary => Some(crate::CargoTargetSelector::Binary(name.to_string())),
        CargoTargetKind::Example => Some(crate::CargoTargetSelector::Example(name.to_string())),
        CargoTargetKind::Test => Some(crate::CargoTargetSelector::Test(name.to_string())),
        CargoTargetKind::Bench => Some(crate::CargoTargetSelector::Bench(name.to_string())),
        CargoTargetKind::BuildScript | CargoTargetKind::Other(_) => None,
    }
}

fn id(value: &str) -> LanguageToolNodeId {
    LanguageToolNodeId(value.to_string())
}

fn retain_stale_workspaces(
    mut current: CargoWorkspaceSnapshot,
    previous: Option<&CargoWorkspaceSnapshot>,
) -> CargoWorkspaceSnapshot {
    let Some(previous) = previous else {
        return current;
    };
    for workspace in &mut current.workspaces {
        if let Some(previous_workspace) = previous
            .workspaces
            .iter()
            .find(|previous| previous.key == workspace.key)
        {
            workspace
                .configuration
                .retain_stale_safe_facts(&previous_workspace.configuration);
        }
    }
    for failure in &mut current.failures {
        let stale_workspace = previous.workspaces.iter().find(|workspace| {
            workspace.root_manifest.as_ref() == Some(&failure.manifest_path)
                || workspace
                    .members
                    .iter()
                    .any(|member| member.manifest_path == failure.manifest_path)
        });
        let Some(stale_workspace) = stale_workspace else {
            continue;
        };
        failure.has_stale_model = true;
        if !current
            .workspaces
            .iter()
            .any(|workspace| workspace.key == stale_workspace.key)
        {
            current.workspaces.push(stale_workspace.clone());
        }
    }
    current
        .workspaces
        .sort_by(|left, right| left.key.cmp(&right.key));
    current
}

fn active_configuration_from_preset_state(
    state: &CargoPresetWorkspaceState,
    settings: &CargoPresetSettings,
    notice: Option<String>,
) -> CargoActiveConfiguration {
    let Some(preset) = state
        .active_preset_id
        .as_ref()
        .and_then(|identifier| settings.presets.get(identifier))
    else {
        return CargoActiveConfiguration {
            notice,
            ..CargoActiveConfiguration::default()
        };
    };
    let scope = state.selection.scope.unwrap_or(preset.scope);
    let package = state.selection.package.as_ref().or(preset.package.as_ref());
    let target = state.selection.target.as_ref().or(preset.target.as_ref());
    CargoActiveConfiguration {
        scope: match (scope, package) {
            (CargoPresetScope::Workspace, _) => "workspace".to_string(),
            (CargoPresetScope::Package, Some(package)) => format!("package {package}"),
            (CargoPresetScope::Package, None) => "selected package".to_string(),
        },
        profile: preset.profile.clone(),
        native_profiling: preset.native_profiler.is_some(),
        selected_features: preset.features.clone(),
        default_features: preset.default_features,
        explicit_target: preset.target_triple.clone(),
        target_selector: target.map(|target| target.summary()),
        environment_keys: preset.environment_keys(),
        notice,
    }
}

fn persist_preset_state(key: String, state: CargoPresetWorkspaceState, cx: &App) -> gpui::Task<()> {
    let key_value_store = KeyValueStore::global(cx);
    cx.background_spawn(async move {
        let result = async {
            let serialized = serde_json::to_string(&state)?;
            key_value_store.write_kvp(key, serialized).await?;
            anyhow::Ok(())
        }
        .await;
        if let Err(error) = result {
            log::warn!("failed to persist Cargo preset state: {error}");
        }
    })
}

#[cfg(test)]
#[derive(Default)]
struct CargoProjectionTimings {
    refresh: Duration,
    admission: Duration,
    application: Duration,
    background_nanoseconds: Arc<std::sync::atomic::AtomicU64>,
}

pub struct CargoPanel {
    project: Entity<Project>,
    workspace: WeakEntity<Workspace>,
    focus_handle: FocusHandle,
    active: bool,
    host: LanguageToolTreeHost,
    provider: CargoTreeProvider,
    active_configuration: CargoActiveConfiguration,
    preset_state: CargoPresetWorkspaceState,
    preset_serialization_key: Option<String>,
    preset_notice: Option<String>,
    pending_preset_serialization: Option<gpui::Task<()>>,
    last_snapshot: Option<Arc<CargoWorkspaceSnapshot>>,
    input_snapshot: Option<Arc<CargoWorkspaceSnapshot>>,
    projection_task: gpui::Task<()>,
    projection_generation: u64,
    #[cfg(test)]
    projection_timings: CargoProjectionTimings,
    refresh_pending: bool,
    refresh_error: Option<LanguageToolProviderStatus>,
    execution_host_available: bool,
    input_host_available: bool,
    pending_clean_confirmation: Option<CargoActionSelection>,
    context_menu: Option<(Entity<ContextMenu>, Point<Pixels>, Subscription)>,
    _subscriptions: Vec<Subscription>,
}

fn cargo_preset_settings<'a>(
    project: &Entity<Project>,
    manifest: Option<&ProjectPath>,
    cx: &'a App,
) -> &'a CargoPresetSettings {
    let location = manifest
        .map(|manifest| SettingsLocation {
            worktree_id: manifest.worktree_id,
            path: &manifest.path,
        })
        .or_else(|| {
            project
                .read(cx)
                .visible_worktrees(cx)
                .next()
                .map(|worktree| SettingsLocation {
                    worktree_id: worktree.read(cx).id(),
                    path: RelPath::empty(),
                })
        });
    CargoPresetSettings::get(location, cx)
}

impl CargoPanel {
    pub async fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> anyhow::Result<Entity<Self>> {
        let workspace_handle = workspace.clone();
        workspace.update_in(&mut cx, |workspace, _window, cx| {
            let project = workspace.project().clone();
            let preset_serialization_key = workspace_scoped_state_key(
                CARGO_PRESET_STATE_KEY,
                workspace.database_id(),
                workspace.session_id().as_deref(),
            );
            let preset_settings = cargo_preset_settings(&project, None, cx);
            let serialized_preset_state = preset_serialization_key.as_ref().and_then(|key| {
                KeyValueStore::global(cx)
                    .read_kvp(key)
                    .map_err(|error| {
                        log::warn!("failed to restore Cargo preset state: {error}");
                        error
                    })
                    .ok()
                    .flatten()
            });
            let mut recovered = recover_workspace_state(
                serialized_preset_state.as_deref(),
                &preset_settings.presets,
            );
            if !cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>() {
                recovered.state.active_preset_id = None;
            }
            let active_configuration = active_configuration_from_preset_state(
                &recovered.state,
                preset_settings,
                recovered.notice.clone(),
            );
            cx.new(|cx: &mut Context<CargoPanel>| {
                let store = project.read(cx).cargo_workspace_store().clone();
                let subscription = cx.subscribe(&store, |panel, _, _, cx| {
                    panel.invalidate(cx);
                });
                let focus_handle = cx.focus_handle();
                let settings_subscription = cx.observe_global::<SettingsStore>(|panel, cx| {
                    panel.reconcile_preset_settings(cx);
                });
                let flags_subscription = cx.observe_global::<FeatureFlagStore>(|panel, cx| {
                    panel.context_menu.take();
                    panel.reconcile_preset_settings(cx);
                });
                let pending_preset_serialization = if recovered.rewrite {
                    preset_serialization_key
                        .as_ref()
                        .map(|key| persist_preset_state(key.clone(), recovered.state.clone(), cx))
                } else {
                    None
                };
                CargoPanel {
                    project,
                    workspace: workspace_handle,
                    focus_handle: focus_handle.clone(),
                    active: false,
                    host: LanguageToolTreeHost::with_focus_handle(focus_handle),
                    provider: CargoTreeProvider {
                        navigation: HashMap::new(),
                        action_selections: HashMap::new(),
                    },
                    active_configuration,
                    preset_state: recovered.state,
                    preset_serialization_key,
                    preset_notice: recovered.notice,
                    pending_preset_serialization,
                    last_snapshot: None,
                    input_snapshot: None,
                    projection_task: gpui::Task::ready(()),
                    projection_generation: 0,
                    #[cfg(test)]
                    projection_timings: CargoProjectionTimings::default(),
                    refresh_pending: false,
                    refresh_error: None,
                    execution_host_available: false,
                    input_host_available: false,
                    pending_clean_confirmation: None,
                    context_menu: None,
                    _subscriptions: vec![subscription, settings_subscription, flags_subscription],
                }
            })
        })
    }

    pub fn toggle_focus(
        workspace: &mut Workspace,
        _: &ToggleCargoPanel,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        workspace.toggle_panel_focus::<Self>(window, cx);
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        #[cfg(test)]
        let started = std::time::Instant::now();
        self.host.cancel_debounce();
        let generation = self.host.start_refresh();
        self.cancel_projection();
        self.refresh_pending = true;
        self.execution_host_available = false;
        self.input_host_available = false;
        let store = self.project.read(cx).cargo_workspace_store().clone();
        let task = store.update(cx, |store, cx| store.refresh(cx));
        let refresh_task = cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |panel, cx| {
                if panel.host.refresh_generation() != generation {
                    return;
                }
                panel.refresh_pending = false;
                match result {
                    Ok(snapshot) => {
                        panel.accept_snapshot(generation, snapshot, cx);
                    }
                    Err(error) => {
                        panel.apply_refresh_error(generation, error, cx);
                    }
                }
                cx.notify();
            })
            .log_err();
        });
        self.host.replace_refresh_task(refresh_task);
        cx.notify();
        #[cfg(test)]
        {
            self.projection_timings.refresh += started.elapsed();
        }
    }

    fn cancel_projection(&mut self) {
        self.projection_generation = self.projection_generation.wrapping_add(1);
        self.projection_task = gpui::Task::ready(());
    }

    fn accept_snapshot(
        &mut self,
        generation: u64,
        snapshot: CargoWorkspaceSnapshot,
        cx: &mut Context<Self>,
    ) {
        if self.host.refresh_generation() != generation {
            return;
        }
        #[cfg(test)]
        let started = std::time::Instant::now();
        self.refresh_pending = false;
        self.refresh_error = None;
        self.execution_host_available = false;
        self.input_host_available = true;
        self.input_snapshot = Some(Arc::new(snapshot));
        self.rebuild_projection(cx);
        #[cfg(test)]
        {
            self.projection_timings.admission += started.elapsed();
        }
    }

    fn apply_refresh_error(
        &mut self,
        generation: u64,
        error: anyhow::Error,
        cx: &mut Context<Self>,
    ) {
        if self.host.refresh_generation() != generation {
            return;
        }
        self.cancel_projection();
        self.refresh_pending = false;
        self.execution_host_available = false;
        self.input_host_available = false;
        let status = if let Some(remote_error) = error.downcast_ref::<CargoWorkspaceRemoteError>() {
            match remote_error.kind {
                CargoWorkspaceRemoteErrorKind::UnsupportedHost => {
                    LanguageToolProviderStatus::Unsupported(remote_error.to_string())
                }
                CargoWorkspaceRemoteErrorKind::Disconnected => {
                    LanguageToolProviderStatus::Disconnected(remote_error.to_string())
                }
                CargoWorkspaceRemoteErrorKind::Restricted => {
                    LanguageToolProviderStatus::Restricted(remote_error.to_string())
                }
            }
        } else {
            LanguageToolProviderStatus::Error(error.to_string())
        };
        self.refresh_error = Some(status.clone());
        self.host.apply_provider_error(generation, status);
        if self.host.take_dirty() {
            self.schedule_debounced_refresh(cx);
        }
    }

    fn rebuild_projection(&mut self, cx: &mut Context<Self>) {
        self.cancel_projection();
        let Some(input) = self.input_snapshot.clone() else {
            return;
        };
        let previous = self.last_snapshot.clone();
        let active = self.active_configuration.clone();
        let generation = self.projection_generation;
        let refresh_generation = self.host.refresh_generation();
        let refresh_pending = self.refresh_pending;
        let refresh_error = self.refresh_error.clone();
        #[cfg(test)]
        let background_executor = cx.background_executor().clone();
        #[cfg(test)]
        let background_nanoseconds = self.projection_timings.background_nanoseconds.clone();
        let projected = cx.background_spawn(async move {
            #[cfg(test)]
            let started = std::time::Instant::now();
            #[cfg(test)]
            assert!(
                !background_executor.is_main_thread(),
                "Cargo projection must run in the background"
            );
            let snapshot = Arc::new(retain_stale_workspaces(
                (*input).clone(),
                previous.as_deref(),
            ));
            let (mut tree, provider) =
                CargoTreeProvider::project_with_configuration(&snapshot, &active);
            if refresh_pending {
                tree.status = LanguageToolProviderStatus::Loading;
            } else if let Some(error) = refresh_error {
                tree.status = error;
            }
            #[cfg(test)]
            background_nanoseconds.store(
                started
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .expect("bounded projection duration"),
                std::sync::atomic::Ordering::SeqCst,
            );
            (snapshot, tree, provider)
        });
        self.projection_task = cx.spawn(async move |this, cx| {
            let (snapshot, tree, provider) = projected.await;
            this.update(cx, |panel, cx| {
                panel.apply_projection(
                    generation,
                    refresh_generation,
                    snapshot,
                    tree,
                    provider,
                    cx,
                );
            })
            .log_err();
        });
    }

    fn apply_projection(
        &mut self,
        generation: u64,
        refresh_generation: u64,
        snapshot: Arc<CargoWorkspaceSnapshot>,
        tree: LanguageToolSnapshot,
        provider: CargoTreeProvider,
        cx: &mut Context<Self>,
    ) {
        if self.projection_generation != generation
            || self.host.refresh_generation() != refresh_generation
        {
            return;
        }
        #[cfg(test)]
        let started = std::time::Instant::now();
        let covers_current_inputs = snapshot.input_fingerprint
            == self
                .project
                .read(cx)
                .cargo_workspace_store()
                .read(cx)
                .current_input_fingerprint(cx);
        self.last_snapshot = Some(snapshot);
        self.provider = provider;
        self.execution_host_available =
            self.input_host_available && !self.refresh_pending && self.refresh_error.is_none();
        self.host.apply_refresh(refresh_generation, Ok(tree));
        if self.host.take_dirty() && !covers_current_inputs {
            self.schedule_debounced_refresh(cx);
        }
        cx.notify();
        #[cfg(test)]
        {
            self.projection_timings.application += started.elapsed();
        }
    }

    fn invalidate(&mut self, cx: &mut Context<Self>) {
        self.cancel_projection();
        self.execution_host_available = false;
        self.input_host_available = false;
        if !self.active {
            self.host.mark_dirty();
            return;
        }
        if self.refresh_pending {
            self.host.mark_dirty();
            return;
        }
        self.schedule_debounced_refresh(cx);
    }

    fn preset_manifest(&self) -> Option<&ProjectPath> {
        self.host
            .selected()
            .and_then(|id| self.provider.action_selection(id))
            .map(|selection| &selection.workspace_manifest)
            .or_else(|| {
                let index = self.host.selected_index()?;
                let root = self
                    .host
                    .visible_rows()
                    .get(..=index)?
                    .iter()
                    .rev()
                    .find(|row| row.parent_id.is_none())?;
                self.provider
                    .action_selection(&root.id)
                    .map(|selection| &selection.workspace_manifest)
            })
            .or_else(|| {
                self.input_snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.workspaces.first())
                    .and_then(|workspace| workspace.root_manifest.as_ref())
            })
    }

    fn preset_settings<'a>(&self, cx: &'a App) -> &'a CargoPresetSettings {
        cargo_preset_settings(&self.project, self.preset_manifest(), cx)
    }

    fn update_selection(
        &mut self,
        select: impl FnOnce(&mut LanguageToolTreeHost),
        cx: &mut Context<Self>,
    ) {
        let previous = self.preset_manifest().cloned();
        select(&mut self.host);
        if self.preset_manifest() != previous.as_ref() {
            self.reconcile_preset_settings(cx);
        }
    }

    fn reconcile_preset_settings(&mut self, cx: &mut Context<Self>) {
        let preset_settings = self.preset_settings(cx);
        let recovered = recover_workspace_state(
            serde_json::to_string(&self.preset_state).ok().as_deref(),
            &preset_settings.presets,
        );
        let mut state = recovered.state;
        if !cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>() {
            state.active_preset_id = None;
        }
        let changed = state != self.preset_state;
        self.preset_state = state;
        if recovered.notice.is_some() {
            self.preset_notice = recovered.notice;
        }
        self.active_configuration = active_configuration_from_preset_state(
            &self.preset_state,
            preset_settings,
            self.preset_notice.clone(),
        );
        if changed || recovered.rewrite {
            self.persist_preset_state(cx);
        }
        self.invalidate(cx);
        cx.notify();
    }

    pub fn set_active_preset(
        &mut self,
        active_preset_id: Option<String>,
        selection: CargoSafeSelectionState,
        cx: &mut Context<Self>,
    ) {
        if active_preset_id.is_some()
            && !cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>()
        {
            self.set_action_notice(
                "Named Cargo presets require the rust-named-cargo-presets-experimental opt-in",
                cx,
            );
            return;
        }
        self.preset_state.active_preset_id = active_preset_id;
        self.preset_state.selection = selection;
        self.preset_notice = None;
        self.reconcile_preset_settings(cx);
        self.persist_preset_state(cx);
    }

    fn persist_preset_state(&mut self, cx: &mut Context<Self>) {
        let Some(key) = self.preset_serialization_key.clone() else {
            return;
        };
        self.pending_preset_serialization =
            Some(persist_preset_state(key, self.preset_state.clone(), cx));
    }

    fn schedule_debounced_refresh(&mut self, cx: &mut Context<Self>) {
        let debounce_task = cx.spawn(async move |this, cx| {
            cx.background_executor()
                .timer(Duration::from_millis(150))
                .await;
            this.update(cx, |panel, cx| panel.refresh(cx)).ok();
        });
        self.host.replace_debounce_task(debounce_task);
    }

    fn activate_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.host.can_activate() {
            return;
        }
        let Some(path) = self
            .host
            .selected()
            .and_then(|id| self.provider.navigation(id))
            .cloned()
        else {
            return;
        };
        if let Some(workspace) = self.workspace.upgrade() {
            workspace
                .update(cx, |workspace, cx| {
                    workspace.open_path(path, None, true, window, cx)
                })
                .detach_and_log_err(cx);
        }
    }

    fn reveal_selection(&self, strategy: ScrollStrategy) {
        self.host.reveal_selection(strategy);
    }

    fn action_runtime(&self, selection: &CargoActionSelection, cx: &App) -> CargoActionRuntime {
        let project = self.project.read(cx);
        let matching_failure = self.last_snapshot.as_ref().and_then(|snapshot| {
            snapshot
                .failures
                .iter()
                .find(|failure| failure.manifest_path == selection.workspace_manifest)
        });
        CargoActionRuntime {
            trusted: !matching_failure.is_some_and(|failure| {
                failure.category
                    == project::cargo_workspace::CargoWorkspaceErrorCategory::Restricted
            }),
            connected: !project.is_disconnected(cx),
            writable: !project.is_read_only(cx),
            cargo_available: !matching_failure.is_some_and(|failure| {
                failure.category
                    == project::cargo_workspace::CargoWorkspaceErrorCategory::CargoNotFound
            }),
            host_capable: self.execution_host_available && project.supports_terminal(cx),
        }
    }

    fn set_action_notice(&mut self, notice: impl Into<String>, cx: &mut Context<Self>) {
        let notice = notice.into();
        self.preset_notice = Some(notice.clone());
        self.active_configuration.notice = Some(notice);
        self.rebuild_projection(cx);
        cx.notify();
    }

    fn execute_selected_action(
        &mut self,
        action: CargoAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.execute_selected_action_with_preset(action, None, window, cx);
    }

    pub(crate) fn execute_selected_action_with_preset(
        &mut self,
        action: CargoAction,
        preset_override: Option<CargoPreset>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(selection) = self
            .host
            .selected()
            .and_then(|id| self.provider.action_selection(id))
            .cloned()
        else {
            self.set_action_notice("Select a Cargo workspace, package, or target first", cx);
            return;
        };
        let preset = preset_override.or_else(|| {
            self.preset_state
                .active_preset_id
                .as_ref()
                .and_then(|identifier| self.preset_settings(cx).presets.get(identifier))
                .cloned()
        });
        if let Err(error) = crate::ensure_experimental_cargo_action(action, preset.as_ref(), cx) {
            self.set_action_notice(error.to_string(), cx);
            return;
        }
        let runtime = self.action_runtime(&selection, cx);
        let Some(availability) =
            cargo_action_availability_with_preset(&selection, runtime, preset.as_ref())
                .into_iter()
                .find(|availability| availability.action == action)
        else {
            self.set_action_notice("The selected Cargo action is unavailable", cx);
            return;
        };
        if !availability.enabled {
            self.set_action_notice(
                availability
                    .reason
                    .unwrap_or_else(|| "The selected Cargo action is unavailable".to_string()),
                cx,
            );
            return;
        }
        let clean_confirmed = if action == CargoAction::Clean {
            if self.pending_clean_confirmation.as_ref() != Some(&selection) {
                self.pending_clean_confirmation = Some(selection);
                self.set_action_notice(
                    "Clean removes Cargo build artifacts. Invoke Clean again on this item to confirm",
                    cx,
                );
                return;
            }
            self.pending_clean_confirmation.take();
            true
        } else {
            self.pending_clean_confirmation = None;
            false
        };
        let Some(workspace) = self.workspace.upgrade() else {
            self.set_action_notice("The workspace is no longer available", cx);
            return;
        };
        let task_contexts = workspace.update(cx, |workspace, cx| {
            tasks_ui::task_contexts(workspace, window, cx)
        });
        let workspace_handle = workspace.downgrade();
        cx.spawn_in(window, async move |panel, cx| {
            let task_contexts = task_contexts.await;
            let result = async {
                let base_context = task_contexts
                    .task_context_for_worktree_id(selection.worktree_id)
                    .ok_or_else(|| {
                        anyhow::anyhow!("No task context exists for the selected worktree")
                    })?;
                panel.read_with(cx, |_, cx| {
                    crate::ensure_experimental_cargo_action(action, preset.as_ref(), cx)
                })??;
                let plan = plan_cargo_action_with_confirmation(
                    action,
                    &selection,
                    preset.as_ref(),
                    base_context,
                    clean_confirmed,
                )?;
                panel.update(cx, |panel, cx| {
                    crate::ensure_experimental_cargo_action(action, preset.as_ref(), cx)?;
                    let current_selection = panel
                        .host
                        .selected()
                        .and_then(|id| panel.provider.action_selection(id));
                    anyhow::ensure!(
                        current_selection == Some(&selection),
                        "Cargo selection changed while preparing the action; invoke it again"
                    );
                    let availability = cargo_action_availability_with_preset(
                        &selection,
                        panel.action_runtime(&selection, cx),
                        preset.as_ref(),
                    )
                    .into_iter()
                    .find(|entry| entry.action == action)
                    .ok_or_else(|| anyhow::anyhow!("Cargo action is no longer available"))?;
                    anyhow::ensure!(
                        availability.enabled,
                        "{}",
                        availability
                            .reason
                            .unwrap_or_else(|| "Cargo action is no longer available".to_string())
                    );
                    anyhow::Ok(())
                })??;
                workspace_handle.update_in(cx, |workspace, window, cx| {
                    let mut dispatcher = WorkspaceCargoActionDispatcher::new(workspace, window, cx);
                    dispatch_cargo_action(plan, &mut dispatcher);
                })?;
                anyhow::Ok(())
            }
            .await;
            if let Err(error) = result {
                panel
                    .update(cx, |panel, cx| {
                        panel.set_action_notice(format!("Cargo action failed: {error}"), cx)
                    })
                    .ok();
            }
        })
        .detach();
    }

    fn open_preset_editor(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>() {
            self.set_action_notice(
                "Named Cargo presets require the rust-named-cargo-presets-experimental opt-in",
                cx,
            );
            return;
        }
        let preset = self
            .preset_state
            .active_preset_id
            .as_ref()
            .and_then(|identifier| self.preset_settings(cx).presets.get(identifier))
            .cloned()
            .unwrap_or_else(|| {
                crate::CargoPreset::ephemeral_default(crate::CargoSubcommand::Build)
            });
        let mut validation_context = CargoPresetValidationContext::default();
        if let Some(snapshot) = &self.last_snapshot {
            for workspace in &snapshot.workspaces {
                validation_context.profiles.extend(
                    workspace
                        .configuration
                        .profiles
                        .iter()
                        .map(|profile| profile.name.clone()),
                );
                for package in &workspace.members {
                    validation_context.packages.insert(package.name.clone());
                    validation_context
                        .targets
                        .extend(package.targets.iter().map(|target| target.name.clone()));
                }
            }
        }
        let Some(workspace) = self.workspace.upgrade() else {
            self.set_action_notice("The workspace is no longer available", cx);
            return;
        };
        let panel = cx.weak_entity();
        workspace.update(cx, |workspace, cx| {
            workspace.toggle_modal(window, cx, move |window, cx| {
                CargoPresetEditor::new(panel, preset, validation_context, window, cx)
            });
        });
    }

    pub(crate) fn save_project_preset(
        &mut self,
        preset: CargoPreset,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
            "Named Cargo presets require the rust-named-cargo-presets-experimental opt-in"
        );
        let selection = self
            .host
            .selected()
            .and_then(|id| self.provider.action_selection(id))
            .cloned()
            .context("Select a Cargo workspace, package, or target before saving for Project")?;
        let runtime = self.action_runtime(&selection, cx);
        if !runtime.trusted {
            anyhow::bail!("Trust this worktree before saving a shared Cargo preset");
        }
        if !runtime.connected {
            anyhow::bail!("The authoritative project host is disconnected");
        }
        if !runtime.writable {
            anyhow::bail!("This project is read-only");
        }

        let settings_path = Arc::<RelPath>::from(
            RelPath::from_unix_str(".zed/settings.json")
                .context("project settings path should be valid")?,
        );
        let settings_directory = Arc::<RelPath>::from(
            RelPath::from_unix_str(".zed").context("settings directory should be valid")?,
        );
        let (revoked, experimental_subscription) =
            crate::cargo_preset_editor::named_preset_save_revocation(cx);
        let project = self.project.clone();
        let panel = cx.weak_entity();
        cx.spawn(async move |_panel, cx| {
            let worktree = project
                .read_with(cx, |project, cx| {
                    project.worktree_for_id(selection.worktree_id, cx)
                })
                .context("Selected Cargo worktree is unavailable")?;
            let (directory_exists, file_exists) = worktree.read_with(cx, |worktree, _| {
                (
                    worktree.entry_for_path(&settings_directory).is_some(),
                    worktree.entry_for_path(&settings_path).is_some(),
                )
            });
            project.read_with(cx, |_, cx| {
                anyhow::ensure!(
                    !revoked.load(std::sync::atomic::Ordering::SeqCst)
                        && cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
                    "Named Cargo preset opt-in was revoked before creating settings"
                );
                anyhow::Ok(())
            })?;
            if !directory_exists {
                project
                    .update(cx, |project, cx| {
                        project.create_entry(
                            ProjectPath {
                                worktree_id: selection.worktree_id,
                                path: settings_directory.clone(),
                            },
                            true,
                            cx,
                        )
                    })
                    .await?;
            }
            project.read_with(cx, |_, cx| {
                anyhow::ensure!(
                    !revoked.load(std::sync::atomic::Ordering::SeqCst)
                        && cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
                    "Named Cargo preset opt-in was revoked before creating settings"
                );
                anyhow::Ok(())
            })?;
            if !file_exists {
                project
                    .update(cx, |project, cx| {
                        project.create_entry(
                            ProjectPath {
                                worktree_id: selection.worktree_id,
                                path: settings_path.clone(),
                            },
                            false,
                            cx,
                        )
                    })
                    .await?;
            }
            project.read_with(cx, |_, cx| {
                anyhow::ensure!(
                    !revoked.load(std::sync::atomic::Ordering::SeqCst)
                        && cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
                    "Named Cargo preset opt-in was revoked before opening settings"
                );
                anyhow::Ok(())
            })?;
            let buffer = project
                .update(cx, |project, cx| {
                    project.open_buffer(
                        ProjectPath {
                            worktree_id: selection.worktree_id,
                            path: settings_path,
                        },
                        cx,
                    )
                })
                .await?;
            let old_text = buffer.read_with(cx, |buffer, _| buffer.text());
            project.read_with(cx, |_, cx| {
                anyhow::ensure!(
                    !revoked.load(std::sync::atomic::Ordering::SeqCst)
                        && cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
                    "Named Cargo preset opt-in was revoked before saving"
                );
                anyhow::Ok(())
            })?;
            let new_text = cx.read_global(|store: &SettingsStore, _| {
                store.new_text_for_update(old_text, |settings| {
                    let cargo = settings.project.cargo.get_or_insert_default();
                    if let Err(error) =
                        apply_preset_save(cargo, &preset, CargoPresetSaveScope::Project, true, true)
                    {
                        log::error!("Cargo project preset update failed: {error:#}");
                    }
                })
            })?;
            buffer.update(cx, |buffer, cx| {
                anyhow::ensure!(
                    !revoked.load(std::sync::atomic::Ordering::SeqCst)
                        && cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
                    "Named Cargo preset opt-in was revoked before saving"
                );
                let _edit_timestamp = buffer.set_text(new_text, cx);
                anyhow::Ok(())
            })?;
            project
                .update(cx, |project, cx| project.save_buffer(buffer, cx))
                .await?;
            project.read_with(cx, |_, cx| {
                anyhow::ensure!(
                    !revoked.load(std::sync::atomic::Ordering::SeqCst)
                        && cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
                    "Named Cargo preset opt-in was revoked before save completion"
                );
                anyhow::Ok(())
            })?;
            drop(experimental_subscription);
            panel.update(cx, |panel, cx| {
                panel.set_action_notice(
                    "Saved Cargo preset to .zed/settings.json without environment values",
                    cx,
                );
            })?;
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
        Ok(())
    }

    fn deploy_preset_menu(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>() {
            self.set_action_notice(
                "Named Cargo presets require the rust-named-cargo-presets-experimental opt-in",
                cx,
            );
            return;
        }
        let panel = cx.entity();
        let presets = self
            .preset_settings(cx)
            .presets
            .values()
            .map(|preset| (preset.id.clone(), preset.label.clone()))
            .collect::<Vec<_>>();
        let menu = ContextMenu::build(window, cx, |mut menu, window, _| {
            menu = menu
                .context(self.focus_handle.clone())
                .header("Cargo preset")
                .entry(
                    "Cargo default",
                    None,
                    window.handler_for(&panel, |panel, _, cx| {
                        panel.set_active_preset(None, CargoSafeSelectionState::default(), cx);
                    }),
                );
            for (identifier, label) in presets {
                menu = menu.entry(
                    format!("{label} ({identifier})"),
                    None,
                    window.handler_for(&panel, move |panel, _, cx| {
                        panel.set_active_preset(
                            Some(identifier.clone()),
                            CargoSafeSelectionState::default(),
                            cx,
                        );
                    }),
                );
            }
            menu
        });
        window.focus(&menu.focus_handle(cx), cx);
        let subscription = cx.subscribe(&menu, |panel, _, _: &DismissEvent, cx| {
            panel.context_menu.take();
            cx.notify();
        });
        self.context_menu = Some((menu, window.mouse_position(), subscription));
        cx.notify();
    }

    fn deploy_context_menu(
        &mut self,
        position: Point<Pixels>,
        id: LanguageToolNodeId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.update_selection(|host| host.select(id), cx);
        let can_open = self.host.can_activate()
            && self
                .host
                .selected()
                .is_some_and(|id| self.provider.navigation(id).is_some());
        let active_preset = self
            .preset_state
            .active_preset_id
            .as_ref()
            .and_then(|identifier| self.preset_settings(cx).presets.get(identifier))
            .cloned();
        let action_entries = self
            .host
            .selected()
            .and_then(|id| self.provider.action_selection(id))
            .map(|selection| {
                cargo_action_availability_with_preset(
                    selection,
                    self.action_runtime(selection, cx),
                    active_preset.as_ref(),
                )
            })
            .unwrap_or_default();
        let menu = ContextMenu::build(window, cx, |mut menu, _, cx| {
            menu = menu.context(self.focus_handle.clone());
            if !action_entries.is_empty() {
                menu = menu.header("Cargo actions");
                for availability in action_entries {
                    if crate::ensure_experimental_cargo_action(
                        availability.action,
                        active_preset.as_ref(),
                        cx,
                    )
                    .is_err()
                    {
                        continue;
                    }
                    let label = availability.accessibility_label;
                    menu = match availability.action {
                        CargoAction::Build => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(BuildSelected),
                        ),
                        CargoAction::Check => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(CheckSelected),
                        ),
                        CargoAction::Run => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(RunSelected),
                        ),
                        CargoAction::RunWithCoverage => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(RunWithCoverageSelected),
                        ),
                        CargoAction::Profile => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(ProfileSelected),
                        ),
                        CargoAction::Test => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(TestSelected),
                        ),
                        CargoAction::Bench => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(BenchSelected),
                        ),
                        CargoAction::Debug => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(DebugSelected),
                        ),
                        CargoAction::Doc => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(DocSelected),
                        ),
                        CargoAction::Clippy => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(ClippySelected),
                        ),
                        CargoAction::Fmt => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(FmtSelected),
                        ),
                        CargoAction::Clean => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(CleanSelected),
                        ),
                        CargoAction::Tree => menu.action_disabled_when(
                            !availability.enabled,
                            label,
                            Box::new(TreeSelected),
                        ),
                    };
                }
                menu = menu.separator();
            }
            menu.when(can_open, |menu| {
                menu.action("Open", Box::new(language_tool_tree::ActivateSelected))
            })
            .action("Refresh", Box::new(language_tool_tree::Refresh))
            .separator()
            .action("Expand All", Box::new(language_tool_tree::ExpandAll))
            .action("Collapse All", Box::new(language_tool_tree::CollapseAll))
        });
        window.focus(&menu.focus_handle(cx), cx);
        let subscription = cx.subscribe(&menu, |panel, _, _: &DismissEvent, cx| {
            panel.context_menu.take();
            cx.notify();
        });
        self.context_menu = Some((menu, position, subscription));
        cx.notify();
    }
}

impl Panel for CargoPanel {
    fn persistent_name() -> &'static str {
        "Cargo"
    }

    fn panel_key() -> &'static str {
        CARGO_PANEL_KEY
    }

    fn position(&self, _: &Window, cx: &App) -> DockPosition {
        match CargoPanelSettings::get_global(cx).dock {
            DockSide::Left => DockPosition::Left,
            DockSide::Right => DockPosition::Right,
        }
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        matches!(position, DockPosition::Left | DockPosition::Right)
    }

    fn set_position(&mut self, position: DockPosition, _: &mut Window, cx: &mut Context<Self>) {
        settings::update_settings_file(<dyn fs::Fs>::global(cx), cx, move |settings, _| {
            settings.cargo_panel.get_or_insert_default().dock = Some(match position {
                DockPosition::Right => DockSide::Right,
                DockPosition::Left | DockPosition::Bottom => DockSide::Left,
            });
        });
    }

    fn default_size(&self, _: &Window, cx: &App) -> Pixels {
        CargoPanelSettings::get_global(cx).default_width
    }

    fn icon(&self, _: &Window, cx: &App) -> Option<IconName> {
        CargoPanelSettings::get_global(cx)
            .button
            .then_some(IconName::FileRust)
    }

    fn icon_tooltip(&self, _: &Window, _: &App) -> Option<&'static str> {
        Some("Cargo")
    }

    fn toggle_action(&self) -> Box<dyn Action> {
        Box::new(ToggleCargoPanel)
    }

    fn starts_open(&self, _: &Window, cx: &App) -> bool {
        CargoPanelSettings::get_global(cx).starts_open
    }

    fn set_active(&mut self, active: bool, _: &mut Window, cx: &mut Context<Self>) {
        let was_active = self.active;
        self.active = active;
        if active && !was_active {
            if matches!(self.host.status(), LanguageToolTreeStatus::Dormant)
                || self.host.take_dirty()
            {
                self.refresh(cx);
            }
        }
    }

    fn activation_priority(&self) -> u32 {
        7
    }

    fn hide_button_setting(&self, _: &App) -> Option<workspace::HideStatusItem> {
        Some(workspace::HideStatusItem::new(|settings| {
            settings.cargo_panel.get_or_insert_default().button = Some(false);
        }))
    }
}

impl Focusable for CargoPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<PanelEvent> for CargoPanel {}

impl Render for CargoPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = self.host.visible_rows_snapshot();
        let selected = self.host.selected().cloned();
        let status = status_message(self.host.status());
        let can_expand_all = self.host.can_expand_all();
        let can_collapse_all = self.host.can_collapse_all();
        let can_refresh = self.host.can_refresh();
        let click_panel = cx.weak_entity();
        let toggle_panel = cx.weak_entity();
        let context_panel = cx.weak_entity();
        v_flex()
            .id("cargo-panel")
            .key_context("CargoPanel")
            .track_focus(&self.focus_handle)
            .size_full()
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::Refresh, _, cx| panel.refresh(cx)),
            )
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
                    panel.update_selection(|host| host.select_next(), cx);
                    panel.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectPrevious, _, cx| {
                    panel.update_selection(|host| host.select_previous(), cx);
                    panel.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectFirst, _, cx| {
                    panel.update_selection(|host| host.select_first(), cx);
                    panel.reveal_selection(ScrollStrategy::Top);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectLast, _, cx| {
                    panel.update_selection(|host| host.select_last(), cx);
                    panel.reveal_selection(ScrollStrategy::Bottom);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectParent, _, cx| {
                    panel.update_selection(|host| host.select_parent(), cx);
                    panel.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
            .on_action(
                cx.listener(|panel, _: &language_tool_tree::SelectFirstChild, _, cx| {
                    panel.update_selection(|host| host.select_first_child(), cx);
                    panel.reveal_selection(ScrollStrategy::Center);
                    cx.notify();
                }),
            )
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
            .on_action(cx.listener(|panel, _: &BuildSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Build, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &CheckSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Check, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &RunSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Run, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &ProfileSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Profile, window, cx)
            }))
            .on_action(
                cx.listener(|panel, _: &RunWithCoverageSelected, window, cx| {
                    panel.execute_selected_action(CargoAction::RunWithCoverage, window, cx)
                }),
            )
            .on_action(cx.listener(|panel, _: &TestSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Test, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &BenchSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Bench, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &SelectPreset, window, cx| {
                panel.deploy_preset_menu(window, cx)
            }))
            .on_action(cx.listener(|panel, _: &DebugSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Debug, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &DocSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Doc, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &ClippySelected, window, cx| {
                panel.execute_selected_action(CargoAction::Clippy, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &FmtSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Fmt, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &CleanSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Clean, window, cx)
            }))
            .on_action(cx.listener(|panel, _: &TreeSelected, window, cx| {
                panel.execute_selected_action(CargoAction::Tree, window, cx)
            }))
            .child(
                div()
                    .min_h_9()
                    .px_2()
                    .py_1()
                    .flex()
                    .flex_wrap()
                    .gap_1()
                    .items_center()
                    .justify_between()
                    .child("Cargo")
                    .child(
                        h_flex()
                            .min_w_0()
                            .flex_wrap()
                            .gap_1()
                            .child(
                                IconButton::new("cargo-select-preset", IconName::ListTodo)
                                    .disabled(!cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>())
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Select Cargo preset")
                                    .tooltip(Tooltip::for_action_title(
                                        "Select Cargo preset",
                                        &SelectPreset,
                                    ))
                                    .on_click(cx.listener(|panel, _, window, cx| {
                                        panel.deploy_preset_menu(window, cx)
                                    })),
                            )
                            .child(
                                IconButton::new("cargo-edit-preset", IconName::Pencil)
                                    .disabled(!cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>())
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Edit Cargo preset")
                                    .tooltip(Tooltip::text("Edit Cargo preset"))
                                    .on_click(cx.listener(|panel, _, window, cx| {
                                        panel.open_preset_editor(window, cx)
                                    })),
                            )
                            .child(
                                IconButton::new("cargo-expand-all", IconName::ExpandVertical)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Expand All")
                                    .tooltip(Tooltip::text("Expand All"))
                                    .disabled(!can_expand_all)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.host.expand_all();
                                        cx.notify();
                                    })),
                            )
                            .child(
                                IconButton::new("cargo-collapse-all", IconName::ListCollapse)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Collapse All")
                                    .tooltip(Tooltip::text("Collapse All"))
                                    .disabled(!can_collapse_all)
                                    .on_click(cx.listener(|panel, _, _, cx| {
                                        panel.host.collapse_all();
                                        cx.notify();
                                    })),
                            )
                            .child(
                                IconButton::new("cargo-refresh", IconName::RefreshTitle)
                                    .shape(ui::IconButtonShape::Square)
                                    .aria_label("Refresh")
                                    .tooltip(Tooltip::text("Refresh"))
                                    .disabled(!can_refresh)
                                    .on_click(cx.listener(|panel, _, _, cx| panel.refresh(cx))),
                            ),
                    ),
            )
            .child(language_tool_tree(
                rows,
                selected,
                status,
                self.host.scroll_handle().clone(),
                move |id, click_count, window, cx| {
                    click_panel
                        .update(cx, |panel, cx| {
                            window.focus(&panel.focus_handle, cx);
                            panel.update_selection(|host| host.select(id.clone()), cx);
                            if click_count > 1 {
                                panel.activate_selected(window, cx);
                            }
                            cx.notify();
                        })
                        .ok();
                },
                move |id, _, cx| {
                    toggle_panel
                        .update(cx, |panel, cx| {
                            panel.update_selection(|host| host.select(id.clone()), cx);
                            panel.host.toggle(&id);
                            cx.notify();
                        })
                        .ok();
                },
                move |id, position, window, cx| {
                    context_panel
                        .update(cx, |panel, cx| {
                            panel.deploy_context_menu(position, id, window, cx)
                        })
                        .ok();
                },
            ))
            .children(self.context_menu.as_ref().map(|(menu, position, _)| {
                deferred(
                    anchored()
                        .position(*position)
                        .anchor(Anchor::TopLeft)
                        .child(menu.clone()),
                )
                .with_priority(1)
            }))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::Arc,
        time::{Duration, Instant},
    };

    use gpui::{TestAppContext, UpdateGlobal};

    use project::cargo_workspace::{
        CargoCandidateFailure, CargoConfigurationCompleteness, CargoDependencyDeclarationOrigin,
        CargoDependencyFeatureCausality, CargoDependencyLockStatus, CargoDependencyModel,
        CargoDependencySourceKind, CargoFeatureModel, CargoHostCompilerModel,
        CargoHostCompilerStatus, CargoPackageModel, CargoProfileModel, CargoProfileOrigin,
        CargoResolvedDependencyModel, CargoSnapshotCompleteness, CargoTargetModel,
        CargoToolchainFormat, CargoToolchainModel, CargoWorkspaceConfiguration,
        CargoWorkspaceErrorCategory, CargoWorkspaceKey, CargoWorkspaceModel,
    };
    use settings::WorktreeId;
    use util::rel_path::RelPath;

    use super::*;

    fn path(value: &str) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_usize(1),
            path: Arc::from(
                RelPath::from_unix_str(&value).expect("fixture path should be relative"),
            ),
        }
    }

    fn snapshot() -> CargoWorkspaceSnapshot {
        let manifest = path("member/Cargo.toml");
        CargoWorkspaceSnapshot {
            revision: 1,
            input_fingerprint: 1,
            completeness: CargoSnapshotCompleteness::Complete,
            failures: Vec::new(),
            workspaces: vec![CargoWorkspaceModel {
                key: CargoWorkspaceKey { root: path("") },
                root_manifest: Some(path("Cargo.toml")),
                display_name: "fixture".to_string(),
                is_virtual: true,
                dependency_graph: None,
                configuration: CargoWorkspaceConfiguration::unresolved(),
                members: vec![CargoPackageModel {
                    id: "member 0.1.0".to_string(),
                    name: "member".to_string(),
                    version: "0.1.0".to_string(),
                    manifest_path: manifest,
                    is_default_member: true,
                    targets: vec![CargoTargetModel {
                        name: "member".to_string(),
                        kind: CargoTargetKind::Binary,
                        crate_types: vec!["bin".to_string()],
                        source_path: Some(path("member/src/main.rs")),
                        source_display_path: Some("member/src/main.rs".to_string()),
                        required_features: Vec::new(),
                        edition: "2024".to_string(),
                    }],
                    features: vec![CargoFeatureModel {
                        name: "default".to_string(),
                        defined: true,
                        enabled: CargoFeatureEnabled::Enabled,
                        expands: Vec::new(),
                    }],
                    dependencies: vec![CargoDependencyModel {
                        declaration_name: "serde".to_string(),
                        rename: None,
                        kind: CargoDependencyKind::Normal,
                        version_requirement: "^1".to_string(),
                        optional: false,
                        uses_default_features: true,
                        requested_features: vec!["derive".to_string()],
                        target: None,
                        source_kind: CargoDependencySourceKind::Registry,
                        resolved_name: Some("serde".to_string()),
                        resolved_version: Some("1.0.0".to_string()),
                        resolved_workspace_member: None,
                        local_manifest: None,
                        declaration_manifest: Some(path("member/Cargo.toml")),
                        declaration_origin: CargoDependencyDeclarationOrigin::Direct,
                        resolved_instances: vec![CargoResolvedDependencyModel {
                            name: "serde".to_string(),
                            version: "1.0.0".to_string(),
                            source_kind: CargoDependencySourceKind::Registry,
                            enabled_features: vec!["derive".to_string()],
                            lock_status: CargoDependencyLockStatus::Locked,
                            workspace_member: None,
                            local_manifest: None,
                        }],
                        resolution_truncated: false,
                        feature_causality: CargoDependencyFeatureCausality::Validated,
                        cycle_detected: false,
                    }],
                }],
            }],
        }
    }

    #[gpui::test]
    async fn cargo_coverage_action_publishes_real_llvm3_report_after_task_success(
        cx: &mut TestAppContext,
    ) {
        use std::{cell::RefCell, process::ExitStatus, rc::Rc};

        struct Terminal {
            calls: Rc<RefCell<Vec<task::SpawnInTerminal>>>,
            fs: Arc<project::FakeFs>,
        }

        impl workspace::TerminalProvider for Terminal {
            fn spawn_prepared(
                &self,
                launch: task::SpawnInTerminal,
                handle: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<anyhow::Result<ExitStatus>>> {
                let terminal = Self {
                    calls: self.calls.clone(),
                    fs: self.fs.clone(),
                };
                window.to_async(cx).spawn(async move |cx| {
                    let original = launch.clone();
                    let prepared = match prepare(launch, cx).await {
                        Ok(prepared) => prepared,
                        Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                        Err(error) => return Some(Err(error)),
                    };
                    if let Err(error) = prepared.validate_task_identity(&original) {
                        return Some(Err(error));
                    }
                    let result = match cx.update(|window, cx| match handle {
                        Some(handle) => {
                            terminal.spawn_structured(prepared.launch, handle, window, cx)
                        }
                        None => terminal.spawn(prepared.launch, window, cx),
                    }) {
                        Ok(result) => result,
                        Err(error) => return Some(Err(error)),
                    };
                    match prepared.completion {
                        Some(complete) => complete(result, cx).await,
                        None => result.await,
                    }
                })
            }

            fn spawn(
                &self,
                task: task::SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<anyhow::Result<ExitStatus>>> {
                let artifact = task.cwd.as_ref().expect("coverage launch directory").join(
                    task.args
                        .get(
                            task.args
                                .iter()
                                .position(|argument| argument == "--output-path")
                                .expect("coverage output flag")
                                + 1,
                        )
                        .expect("coverage output argument"),
                );
                self.calls.borrow_mut().push(task);
                let fs = self.fs.clone();
                cx.background_spawn(async move {
                    let report =
                        include_str!("../../project/test_data/source_coverage/llvm3-run.json")
                            .replace("/workspace/", "/fixture/");
                    fs.insert_file(&artifact, report.into_bytes()).await;
                    Some(Ok(ExitStatus::default()))
                })
            }
        }

        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/fixture",
            serde_json::json!({
                "Cargo.toml": "[workspace]\nmembers = [\"member\"]",
                "member": {"Cargo.toml": "[package]\nname = \"member\"\nversion = \"0.1.0\"", "src": {"main.rs": "fn main() {}"}},
                "src": {"bin": {"fixture.rs": "fn main() {}"}},
                "target": {}
            }),
        )
        .await;
        let project = Project::test(fs.clone(), ["/fixture".as_ref()], cx).await;
        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("fixture worktree")
                .read(cx)
                .id()
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = CargoPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Cargo panel");
        let calls = Rc::new(RefCell::new(Vec::new()));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(Terminal {
                calls: calls.clone(),
                fs,
            })
        });
        let mut snapshot = snapshot();
        let root = snapshot.workspaces.first_mut().expect("workspace model");
        root.key.root.worktree_id = worktree_id;
        root.root_manifest.as_mut().expect("manifest").worktree_id = worktree_id;
        let member = root.members.first_mut().expect("member");
        member.manifest_path.worktree_id = worktree_id;
        member.dependencies.clear();
        let (tree, provider) = CargoTreeProvider::project(&snapshot);
        let selected = provider
            .action_selections
            .iter()
            .find(|(_, selection)| {
                matches!(
                    selection.node_kind,
                    CargoActionNodeKind::Target(CargoActionTargetKind::Binary)
                )
            })
            .map(|(id, _)| id.clone())
            .expect("binary action node");
        panel.update(cx, |panel, _| {
            panel.host.replace_snapshot(tree);
            panel.host.expand_all();
            panel.host.select(selected);
            panel.provider = provider;
            let snapshot = Arc::new(snapshot);
            panel.input_snapshot = Some(snapshot.clone());
            panel.last_snapshot = Some(snapshot);
            panel.execution_host_available = true;
            panel.input_host_available = true;
        });
        let mut preset = CargoPreset::ephemeral_default(crate::CargoSubcommand::Run);
        preset.presentation.save = task::SaveStrategy::None;
        panel.update_in(cx, |panel, window, cx| {
            panel.execute_selected_action_with_preset(
                CargoAction::RunWithCoverage,
                Some(preset),
                window,
                cx,
            );
        });
        cx.run_until_parked();

        let calls = calls.borrow();
        assert_eq!(calls.len(), 1);
        let task = calls.first().expect("one coverage Task");
        assert_eq!(task.command.as_deref(), Some("cargo"));
        assert_eq!(
            task.args
                .iter()
                .take(5)
                .map(String::as_str)
                .collect::<Vec<_>>(),
            [
                "llvm-cov",
                "--json",
                "--output-path",
                task.args.get(3).expect("bound collector output").as_str(),
                "run"
            ]
        );
        let artifact = task.args.get(3).expect("bound collector output");
        assert!(artifact.starts_with("target/zed-coverage-"));
        assert!(artifact.ends_with(".json"));
        assert_ne!(artifact, crate::CARGO_COVERAGE_ARTIFACT_PATH);
        let history = workspace.read_with(cx, |workspace, cx| {
            workspace
                .project()
                .read(cx)
                .task_store()
                .read(cx)
                .task_inventory()
                .cloned()
                .expect("Task history")
        });
        let (_, cached) = history.read_with(cx, |history, _| {
            history
                .last_scheduled_task(Some(&task.id))
                .expect("scheduled coverage Task")
        });
        assert_eq!(
            cached
                .resolved_artifact_group()
                .expect("bound coverage group")
                .files
                .get("report")
                .expect("bound declaration")
                .path,
            *artifact
        );
        assert_eq!(
            cached
                .original_task()
                .artifact_group
                .as_ref()
                .expect("unbound template")
                .files
                .get("report")
                .expect("unbound coverage report")
                .path,
            crate::CARGO_COVERAGE_ARTIFACT_PATH
        );
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        store.read_with(cx, |store, _| {
            let selected = store
                .selected_snapshot()
                .expect("successful coverage report");
            assert_eq!(
                selected.status,
                project::source_coverage::SourceCoverageStatus::Current,
                "{selected:?}"
            );
            assert_eq!(selected.files.len(), 1);
            assert_eq!(
                selected.files.first().expect("mapped source").covered_lines,
                3
            );
        });
        assert!(
            workspace.read_with(cx, |workspace, cx| {
                workspace.item_of_type::<editor::Editor>(cx).is_none()
            }),
            "a coverage Data artifact should populate coverage state without opening JSON"
        );
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn cargo_profile_action_panel_dispatches_and_rejects_unavailable_host(
        cx: &mut TestAppContext,
    ) {
        use std::{cell::RefCell, rc::Rc};
        struct Terminal(Rc<RefCell<Vec<task::SpawnInTerminal>>>);
        impl workspace::TerminalProvider for Terminal {
            fn spawn_prepared(
                &self,
                launch: task::SpawnInTerminal,
                handle: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
                let terminal = Self(self.0.clone());
                window.to_async(cx).spawn(async move |cx| {
                    let original = launch.clone();
                    let prepared = match prepare(launch, cx).await {
                        Ok(prepared) => prepared,
                        Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                        Err(error) => return Some(Err(error)),
                    };
                    if let Err(error) = prepared.validate_task_identity(&original) {
                        return Some(Err(error));
                    }
                    let result = match cx.update(|window, cx| match handle {
                        Some(handle) => {
                            terminal.spawn_structured(prepared.launch, handle, window, cx)
                        }
                        None => terminal.spawn(prepared.launch, window, cx),
                    }) {
                        Ok(result) => result,
                        Err(error) => return Some(Err(error)),
                    };
                    match prepared.completion {
                        Some(complete) => complete(result, cx).await,
                        None => result.await,
                    }
                })
            }

            fn spawn(
                &self,
                task: task::SpawnInTerminal,
                _: &mut Window,
                _: &mut App,
            ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
                self.0.borrow_mut().push(task);
                gpui::Task::ready(Some(Err(anyhow::anyhow!("configured profiler not found"))))
            }
        }
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/fixture", serde_json::json!({
            "Cargo.toml": "[workspace]\nmembers = [\"member\"]",
            "member": {"Cargo.toml": "[package]\nname = \"member\"\nversion = \"0.1.0\"", "src": {"main.rs": "fn main() {}"}},
            "target": {"profile.svg": "<svg/>"}
        })).await;
        let project = Project::test(fs, ["/fixture".as_ref()], cx).await;
        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("worktree")
                .read(cx)
                .id()
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = CargoPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Cargo panel");
        let calls = Rc::new(RefCell::new(Vec::new()));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(Terminal(calls.clone()))
        });
        let mut snapshot = snapshot();
        let root = snapshot.workspaces.first_mut().expect("workspace model");
        root.key.root.worktree_id = worktree_id;
        root.root_manifest.as_mut().expect("manifest").worktree_id = worktree_id;
        let member = root.members.first_mut().expect("member");
        member.manifest_path.worktree_id = worktree_id;
        member.dependencies.clear();
        let (tree, provider) = CargoTreeProvider::project(&snapshot);
        let selected = provider
            .action_selections
            .iter()
            .find(|(_, selection)| {
                matches!(
                    selection.node_kind,
                    CargoActionNodeKind::Target(CargoActionTargetKind::Binary)
                )
            })
            .map(|(id, _)| id.clone())
            .expect("binary action node");
        panel.update(cx, |panel, _| {
            panel.host.replace_snapshot(tree);
            panel.host.expand_all();
            panel.host.select(selected.clone());
            panel.provider = provider;
            let snapshot = Arc::new(snapshot);
            panel.input_snapshot = Some(snapshot.clone());
            panel.last_snapshot = Some(snapshot);
            panel.execution_host_available = false;
        });
        let mut preset = CargoPreset::ephemeral_default(crate::CargoSubcommand::Run);
        preset.presentation.save = task::SaveStrategy::None;
        preset.profiler = Some(
            crate::CargoProfiler::from_settings(&settings::CargoProfilerSettingsContent {
                command: Some("missing-profiler".into()),
                args: Some(vec!["record".into(), "argument with spaces".into()]),
                artifact_path: Some("target/profile.svg".into()),
                artifact_kind: Some("svg".into()),
                artifact_max_bytes: Some(1024),
            })
            .expect("profiler declaration"),
        );
        panel.update_in(cx, |panel, window, cx| {
            panel.execute_selected_action_with_preset(
                CargoAction::Profile,
                Some(preset.clone()),
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert!(
            calls.borrow().is_empty(),
            "unavailable hosts must not start a Task"
        );
        panel.update_in(cx, |panel, window, cx| {
            panel.execution_host_available = true;
            panel.input_host_available = true;
            panel.execute_selected_action_with_preset(
                CargoAction::Profile,
                Some(preset.clone()),
                window,
                cx,
            );
            panel.host.select_first();
        });
        cx.run_until_parked();
        assert!(
            calls.borrow().is_empty(),
            "a changed selection must not start a stale Task"
        );
        panel.update_in(cx, |panel, window, cx| {
            panel.host.select(selected);
            panel.execute_selected_action_with_preset(
                CargoAction::Profile,
                Some(preset),
                window,
                cx,
            );
        });
        cx.run_until_parked();
        let calls = calls.borrow();
        assert_eq!(calls.len(), 1);
        let task = calls.first().expect("one explicit profile Task");
        assert_eq!(task.command.as_deref(), Some("missing-profiler"));
        assert_eq!(task.args, vec!["record", "argument with spaces"]);
        assert_eq!(task.cwd.as_deref(), Some(std::path::Path::new("/fixture")));
        assert!(
            workspace.read_with(cx, |workspace, cx| workspace
                .item_of_type::<editor::Editor>(cx)
                .is_none()),
            "a failed profiler must not open even an existing artifact"
        );
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn queued_project_preset_save_off_then_on_requires_fresh_operation(
        cx: &mut TestAppContext,
    ) {
        use fs::Fs as _;
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
        });
        let filesystem = project::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/fixture",
                serde_json::json!({
                    "Cargo.toml": "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
                    "src": {"main.rs": "fn main() {}"},
                    ".zed": {"settings.json": "{}"}
                }),
            )
            .await;
        let project = Project::test(filesystem.clone(), ["/fixture".as_ref()], cx).await;
        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("fixture worktree")
                .read(cx)
                .id()
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_context = cx.update(|window, cx| window.to_async(cx));
        let panel = CargoPanel::load(workspace.downgrade(), async_context)
            .await
            .expect("Cargo panel");
        let mut source = snapshot();
        let model = source.workspaces.first_mut().expect("fixture workspace");
        model.key.root.worktree_id = worktree_id;
        model
            .root_manifest
            .as_mut()
            .expect("root manifest")
            .worktree_id = worktree_id;
        let (tree, provider) = CargoTreeProvider::project(&source);
        let selected = provider
            .action_selections
            .iter()
            .find(|(_, selection)| selection.node_kind == CargoActionNodeKind::Workspace)
            .map(|(id, _)| id.clone())
            .expect("workspace action node");
        panel.update(cx, |panel, _| {
            panel.host.replace_snapshot(tree);
            panel.host.select(selected);
            panel.provider = provider;
            let source = Arc::new(source);
            panel.input_snapshot = Some(source.clone());
            panel.last_snapshot = Some(source);
        });
        let mut preset = CargoPreset::ephemeral_default(crate::CargoSubcommand::Run);
        preset.id = "queued-project-experiment".into();
        panel.update(cx, |panel, cx| {
            panel
                .save_project_preset(preset.clone(), cx)
                .expect("queue actual project save");
        });
        cx.update(|_, cx| crate::set_experimental_cargo_opt_ins_for_test(false, cx));
        cx.update(|_, cx| crate::set_experimental_cargo_opt_ins_for_test(true, cx));
        assert!(cx.update(|_, cx| cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>()));
        cx.run_until_parked();
        let settings_path = std::path::Path::new("/fixture/.zed/settings.json");
        let saved = filesystem
            .load(settings_path)
            .await
            .expect("existing project settings");
        assert_eq!(
            saved, "{}",
            "revocation permanently retires the original queued save"
        );
        assert!(!panel.read_with(cx, |panel, _| {
            panel
                .preset_notice
                .as_deref()
                .is_some_and(|notice| notice.starts_with("Saved Cargo preset"))
        }));
        panel.update(cx, |panel, cx| {
            panel
                .save_project_preset(preset, cx)
                .expect("fresh project save");
        });
        let deadline = cx.background_executor.timer(Duration::from_secs(10));
        let save_completed = async {
            loop {
                cx.run_until_parked();
                let saved = filesystem
                    .load(settings_path)
                    .await
                    .expect("actual project settings");
                if saved.contains("queued-project-experiment") {
                    break;
                }
                cx.background_executor
                    .timer(Duration::from_millis(10))
                    .await;
            }
        };
        match futures::future::select(Box::pin(save_completed), Box::pin(deadline)).await {
            futures::future::Either::Left(((), _)) => {}
            futures::future::Either::Right(_) => {
                panic!("fresh project preset did not reach the actual save owner")
            }
        }
        cx.run_until_parked();
        assert!(panel.read_with(cx, |panel, _| {
            panel
                .preset_notice
                .as_deref()
                .is_some_and(|notice| notice.starts_with("Saved Cargo preset"))
        }));
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn cargo_panel_preset_menu_selects_project_precedence_and_restores_defaults(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
            SettingsStore::update_global(cx, |store, cx| {
                store.set_user_settings(r#"{
                    "cargo": {"schema_version": 2, "presets": {
                        "shared": {"label": "User preset", "subcommand": "run",
                            "profile": "dev", "environment": {"USER_ONLY": "private user value"}}
                    }}
                }"#, cx).result().expect("user preset");
                store.set_local_settings(
                    settings::WorktreeId::from_usize(1),
                    settings::LocalSettingsPath::InWorktree(RelPath::empty().into()),
                    settings::LocalSettingsKind::Settings,
                    Some(r#"{
                        "cargo": {"schema_version": 2, "presets": {
                            "shared": {"label": "Project preset", "profile": "release",
                                "toolchain": "1.97.1", "environment": {"PROJECT_ONLY": "private project value"}}
                        }}
                    }"#), cx,
                ).expect("project preset");
            });
        });
        cx.update(|cx| crate::set_experimental_cargo_opt_ins_for_test(true, cx));
        let project = Project::test(project::FakeFs::new(cx.executor()), [], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_context = cx.update(|window, cx| window.to_async(cx));
        let panel = CargoPanel::load(workspace.downgrade(), async_context)
            .await
            .expect("Cargo panel");
        let mut source = snapshot();
        let mut second = source.workspaces.first().expect("first workspace").clone();
        second.key.root.worktree_id = settings::WorktreeId::from_usize(2);
        second
            .root_manifest
            .as_mut()
            .expect("second manifest")
            .worktree_id = settings::WorktreeId::from_usize(2);
        source.workspaces.push(second);
        let source = Arc::new(source);
        let (tree, provider) = CargoTreeProvider::project(&source);
        panel.update(cx, |panel, _| {
            panel.host.replace_snapshot(tree);
            panel.host.select_first();
            panel.provider = provider;
            panel.input_snapshot = Some(source.clone());
            panel.last_snapshot = Some(source);
        });
        let menu = panel.update_in(cx, |panel, window, cx| {
            panel.deploy_preset_menu(window, cx);
            panel.context_menu.as_ref().expect("preset menu").0.clone()
        });
        menu.update_in(cx, |menu, window, cx| {
            menu.select_first(&menu::SelectFirst, window, cx);
            menu.select_next(&menu::SelectNext, window, cx);
            menu.confirm(&menu::Confirm, window, cx);
        });
        panel.read_with(cx, |panel, cx| {
            assert_eq!(
                panel.preset_state.active_preset_id.as_deref(),
                Some("shared")
            );
            assert_eq!(
                panel.active_configuration.profile.as_deref(),
                Some("release")
            );
            assert!(panel.context_menu.is_none());
            let preset = panel
                .preset_settings(cx)
                .presets
                .get("shared")
                .expect("project preset");
            assert_eq!(preset.label, "Project preset");
            assert_eq!(preset.toolchain.as_deref(), Some("1.97.1"));
            let compiled = crate::compile_preset(
                preset,
                &crate::CargoCompileContext {
                    workspace_name: Some("fixture".into()),
                    workspace_cwd: Some("/fixture".into()),
                    package_name: None,
                    package_cwd: None,
                },
                None,
            )
            .expect("selected preset compiles");
            assert_eq!(
                compiled.task_template.args,
                ["+1.97.1", "run", "--workspace", "--profile", "release"]
            );
            assert_eq!(
                compiled
                    .task_template
                    .env
                    .get("USER_ONLY")
                    .map(String::as_str),
                Some("private user value")
            );
            assert_eq!(
                compiled
                    .task_template
                    .env
                    .get("PROJECT_ONLY")
                    .map(String::as_str),
                Some("private project value")
            );
            let serialized = serde_json::to_string(&panel.preset_state).expect("safe state");
            assert!(!serialized.contains("private"));
            assert!(!serialized.contains("USER_ONLY"));
            assert!(!serialized.contains("PROJECT_ONLY"));
            assert_eq!(
                CargoPresetSettings::get_global(cx)
                    .presets
                    .get("shared")
                    .expect("user preset")
                    .profile
                    .as_deref(),
                Some("dev")
            );
        });
        panel.update(cx, |panel, cx| {
            panel.update_selection(|host| host.select_last(), cx);
            assert_eq!(panel.active_configuration.profile.as_deref(), Some("dev"));
            assert!(
                !panel
                    .active_configuration
                    .environment_keys
                    .contains(&"PROJECT_ONLY".to_string())
            );
            panel.host.expand_all();
            panel.update_selection(|host| host.select_next(), cx);
            assert_eq!(panel.active_configuration.profile.as_deref(), Some("dev"));
            panel.update_selection(|host| host.select_first(), cx);
            assert_eq!(
                panel.active_configuration.profile.as_deref(),
                Some("release")
            );
        });
        let menu = panel.update_in(cx, |panel, window, cx| {
            panel.deploy_preset_menu(window, cx);
            panel.context_menu.as_ref().expect("preset menu").0.clone()
        });
        menu.update_in(cx, |menu, window, cx| {
            menu.select_first(&menu::SelectFirst, window, cx);
            menu.confirm(&menu::Confirm, window, cx);
        });
        panel.read_with(cx, |panel, _| {
            assert!(panel.preset_state.active_preset_id.is_none());
            assert!(panel.active_configuration.profile.is_none());
            assert!(panel.active_configuration.environment_keys.is_empty());
        });
        panel.update(cx, |panel, cx| {
            panel.set_active_preset(
                Some("shared".into()),
                CargoSafeSelectionState::default(),
                cx,
            )
        });
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel.preset_state.active_preset_id.as_deref(),
                Some("shared")
            )
        });
        cx.update(|_, cx| crate::set_experimental_cargo_opt_ins_for_test(false, cx));
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(panel.preset_state.active_preset_id.is_none());
            assert!(panel.active_configuration.profile.is_none());
        });
        panel.update_in(cx, |panel, window, cx| {
            panel.set_active_preset(
                Some("shared".into()),
                CargoSafeSelectionState::default(),
                cx,
            );
            panel.deploy_preset_menu(window, cx);
            panel.open_preset_editor(window, cx);
            assert!(panel.preset_state.active_preset_id.is_none());
            assert!(panel.context_menu.is_none());
        });
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn cargo_panel_settings_moves_preserve_shared_dock_focus(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
        });
        let project = Project::test(project::FakeFs::new(cx.executor()), [], cx).await;
        let (container, cx) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_context = cx.update(|window, cx| window.to_async(cx));
        let panel = CargoPanel::load(workspace.downgrade(), async_context)
            .await
            .expect("Cargo panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace
                .right_dock()
                .update(cx, |dock, cx| dock.set_open(true, window, cx));
            workspace.focus_panel::<CargoPanel>(window, cx);
        });
        for side in [DockSide::Left, DockSide::Right] {
            cx.update(|_, cx| {
                let mut settings = *CargoPanelSettings::get_global(cx);
                settings.dock = side;
                CargoPanelSettings::override_global(settings, cx);
            });
            cx.run_until_parked();
            workspace.update_in(cx, |workspace, window, cx| {
                let (selected, previous) = match side {
                    DockSide::Left => (workspace.left_dock(), workspace.right_dock()),
                    DockSide::Right => (workspace.right_dock(), workspace.left_dock()),
                };
                assert!(selected.read(cx).is_open());
                assert!(selected.read(cx).panel_for_id(panel.entity_id()).is_some());
                assert_eq!(selected.read(cx).panels_len(), 1);
                assert!(previous.read(cx).panel_for_id(panel.entity_id()).is_none());
                assert!(panel.read(cx).focus_handle.is_focused(window));
            });
            panel.update(cx, |_, cx| cx.notify());
            cx.run_until_parked();
            workspace.read_with(cx, |workspace, cx| {
                assert_eq!(
                    workspace.left_dock().read(cx).panels_len()
                        + workspace.right_dock().read(cx).panels_len(),
                    1
                );
            });
        }
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn cargo_dashboard_production_projection_rejects_stale_configuration_and_host_results(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
        });
        let project = Project::test(project::FakeFs::new(cx.executor()), [], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = CargoPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Cargo panel");
        panel.update(cx, |panel, cx| panel.refresh(cx));
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(
                panel
                    .last_snapshot
                    .as_ref()
                    .expect("empty project refresh")
                    .workspaces
                    .is_empty()
            )
        });
        let mut source = snapshot();
        source.input_fingerprint = project.read_with(cx, |project, cx| {
            project
                .cargo_workspace_store()
                .read(cx)
                .current_input_fingerprint(cx)
        });
        let (old_generation, old_projection) = panel.update(cx, |panel, cx| {
            let old_generation = panel.host.start_refresh();
            panel.accept_snapshot(old_generation, source.clone(), cx);
            let old_projection = panel.projection_generation;
            assert!(
                panel
                    .last_snapshot
                    .as_ref()
                    .expect("previous result")
                    .workspaces
                    .is_empty(),
                "projection must not run inline"
            );
            let current_generation = panel.host.start_refresh();
            let mut current = source.clone();
            current.revision = 2;
            panel.accept_snapshot(current_generation, current, cx);
            panel.active_configuration.profile = Some("ship".into());
            panel.set_action_notice("obsolete notice", cx);
            panel.set_action_notice("current notice", cx);
            panel.accept_snapshot(old_generation, source.clone(), cx);
            (old_generation, old_projection)
        });
        cx.run_until_parked();
        let selected = panel.update(cx, |panel, cx| {
            assert_eq!(
                panel
                    .last_snapshot
                    .as_ref()
                    .expect("current result")
                    .revision,
                2
            );
            assert!(panel.execution_host_available);
            panel.host.expand_all();
            assert!(
                panel
                    .host
                    .visible_rows()
                    .iter()
                    .any(|row| row.secondary_label.as_deref() == Some("current notice"))
            );
            assert!(
                !panel
                    .host
                    .visible_rows()
                    .iter()
                    .any(|row| row.secondary_label.as_deref() == Some("obsolete notice"))
            );
            for (generation, refresh_generation) in [
                (old_projection, panel.host.refresh_generation()),
                (panel.projection_generation, old_generation),
            ] {
                let (tree, provider) = CargoTreeProvider::project(&source);
                panel.apply_projection(
                    generation,
                    refresh_generation,
                    Arc::new(source.clone()),
                    tree,
                    provider,
                    cx,
                );
                assert_eq!(
                    panel
                        .last_snapshot
                        .as_ref()
                        .expect("late result rejected")
                        .revision,
                    2
                );
            }
            panel.host.select_first();
            let selected = panel.host.selected().cloned();
            panel.set_active_preset(None, CargoSafeSelectionState::default(), cx);
            let mut current = source.clone();
            current.revision = 3;
            panel.accept_snapshot(panel.host.refresh_generation(), current, cx);
            selected
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            assert_eq!(panel.host.selected(), selected.as_ref());
            assert!(
                panel.host.visible_rows().len() > 1,
                "expansion survives configuration changes"
            );
            assert!(panel.host.visible_rows().iter().any(|row| {
                row.label == "Active configuration"
                    && row
                        .secondary_label
                        .as_deref()
                        .is_some_and(|label| label.contains("Cargo default profile"))
            }));
            panel.set_action_notice("invalidated notice", cx);
        });
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        store.update(cx, |_, cx| {
            cx.emit(project::cargo_workspace_store::CargoWorkspaceStoreEvent::Invalidated)
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            assert!(!panel.execution_host_available);
            assert!(
                !panel
                    .host
                    .visible_rows()
                    .iter()
                    .any(|row| row.secondary_label.as_deref() == Some("invalidated notice"))
            );
            panel.accept_snapshot(panel.host.refresh_generation(), source.clone(), cx);
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            panel.apply_refresh_error(
                panel.host.refresh_generation(),
                CargoWorkspaceRemoteError {
                    kind: CargoWorkspaceRemoteErrorKind::Restricted,
                }
                .into(),
                cx,
            );
            panel.set_action_notice("host remains restricted", cx);
        });
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            assert!(matches!(
                panel.host.status(),
                LanguageToolTreeStatus::Restricted(_)
            ));
            assert!(
                !panel.execution_host_available,
                "notice projection cannot restore host authority"
            );
            let generation = panel.host.start_refresh();
            panel.accept_snapshot(generation, source, cx);
        });
        cx.run_until_parked();
        panel.read_with(cx, |panel, _| {
            assert!(panel.execution_host_available);
            assert!(!matches!(
                panel.host.status(),
                LanguageToolTreeStatus::Restricted(_)
            ));
        });
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[test]
    fn projection_is_stable_finite_and_direct_only() {
        let snapshot = snapshot();
        let (first, first_provider) = CargoTreeProvider::project(&snapshot);
        let (second, _) = CargoTreeProvider::project(&snapshot);
        let first_ids = collect_ids(&first.roots);
        let second_ids = collect_ids(&second.roots);
        assert_eq!(first_ids, second_ids);
        assert!(first_ids.iter().all(|id| !id.0.contains("/Users/")));

        let dependencies = find_node(&first.roots, "Dependencies")
            .expect("Dependencies section should be projected");
        let dependency = dependencies
            .children
            .first()
            .and_then(|group| group.children.first())
            .expect("direct dependency should be projected under its kind");
        assert_eq!(dependency.children.len(), 4);
        assert!(
            dependency
                .children
                .iter()
                .all(|section| { section.children.iter().all(|fact| fact.children.is_empty()) })
        );
        assert_eq!(
            first_provider.navigation(&dependency.id),
            Some(&path("member/Cargo.toml"))
        );
        assert_eq!(<CargoPanel as Panel>::persistent_name(), "Cargo");
    }

    #[test]
    fn rust_workspace_projects_one_thousand_packages_with_stable_finite_ids() {
        let mut large_snapshot = snapshot();
        let package_template = large_snapshot.workspaces[0].members[0].clone();
        large_snapshot.workspaces[0].members = (0..1_000)
            .map(|index| {
                let package_name = format!("package-{index:04}");
                let package_root = format!("members/{package_name}");
                let mut package = package_template.clone();
                package.id = format!("{package_name} 0.1.0");
                package.name = package_name.clone();
                package.manifest_path = path(&format!("{package_root}/Cargo.toml"));
                package.is_default_member = index == 0;
                package.targets[0].name = package_name;
                package.targets[0].source_path = Some(path(&format!("{package_root}/src/main.rs")));
                package.targets[0].source_display_path =
                    Some(format!("{package_root}/src/main.rs"));
                package
            })
            .collect();

        let (first, _) = CargoTreeProvider::project(&large_snapshot);
        let (second, _) = CargoTreeProvider::project(&large_snapshot);
        let first_ids = collect_ids(&first.roots);
        let second_ids = collect_ids(&second.roots);
        let unique_ids = first_ids.iter().collect::<HashSet<_>>();

        assert_eq!(large_snapshot.workspaces[0].members.len(), 1_000);
        assert_eq!(first_ids, second_ids);
        assert_eq!(unique_ids.len(), first_ids.len());
        assert!(first_ids.len() >= 20_000);
        assert!(
            first_ids.len() < 25_000,
            "1,000-package projection produced {} rows",
            first_ids.len()
        );
        assert!(first_ids.iter().all(|id| !id.0.contains("/Users/")));
    }

    const DASHBOARD_ROW_COUNT: usize = 10_000;

    async fn dashboard_projection(cx: &mut TestAppContext) -> LanguageToolSnapshot {
        const VISIBLE_ROW_LIMIT: usize = DASHBOARD_ROW_COUNT;

        let mut large_snapshot = snapshot();
        let package_template = large_snapshot.workspaces[0].members[0].clone();
        large_snapshot.workspaces[0].members = (0..400)
            .map(|index| scaled_package(&package_template, index))
            .collect();

        let initial_projection = CargoTreeProvider::project(&large_snapshot).0;
        let initial_count = collect_ids(&initial_projection.roots).len();
        assert!(
            initial_count < VISIBLE_ROW_LIMIT,
            "initial projection produced {initial_count} rows"
        );
        let remaining_rows = VISIBLE_ROW_LIMIT - initial_count;
        large_snapshot.workspaces[0]
            .members
            .extend((0..remaining_rows).map(|offset| {
                let index = 400 + offset;
                let mut package = scaled_package(&package_template, index);
                package.targets.clear();
                package.features.clear();
                package.dependencies.clear();
                package
            }));

        let projection_task = cx
            .background_executor
            .spawn(async move { CargoTreeProvider::project(&large_snapshot).0 });
        cx.background_executor.timer(Duration::from_millis(1)).await;
        let projected = projection_task.await;
        assert_eq!(collect_ids(&projected.roots).len(), VISIBLE_ROW_LIMIT);

        projected
    }

    #[gpui::test]
    async fn cargo_dashboard_reconciles_ten_thousand_rows(cx: &mut TestAppContext) {
        let projected = dashboard_projection(cx).await;
        let expected_ids = collect_ids(&projected.roots);
        assert_eq!(
            expected_ids.iter().collect::<HashSet<_>>().len(),
            DASHBOARD_ROW_COUNT
        );
        let mut host = LanguageToolTreeHost::default();
        host.replace_snapshot(projected.clone());
        host.expand_all();
        let visible_ids = |host: &LanguageToolTreeHost| {
            host.visible_rows()
                .iter()
                .map(|row| row.id.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(visible_ids(&host), expected_ids);
        for range in [
            0..25,
            4_000..4_025,
            9_990..10_010,
            10_000..10_025,
            10_010..10_020,
        ] {
            let actual = host
                .visible_rows_in_range(range.clone())
                .iter()
                .map(|row| row.id.clone())
                .collect::<Vec<_>>();
            let start = range.start.min(DASHBOARD_ROW_COUNT);
            let end = range.end.min(DASHBOARD_ROW_COUNT);
            assert_eq!(actual, expected_ids[start..end]);
        }
        let selected = expected_ids.get(4_012).expect("selected row").clone();
        host.select(selected.clone());
        let expanded = host.expanded().clone();
        host.replace_snapshot(projected.clone());
        assert_eq!(host.selected(), Some(&selected));
        assert_eq!(host.expanded(), &expanded);
        assert_eq!(visible_ids(&host), expected_ids);
        let mut empty = projected;
        empty.roots.clear();
        host.replace_snapshot(empty);
        assert!(host.visible_rows().is_empty());
        assert!(host.visible_rows_in_range(4_000..4_025).is_empty());
        assert!(host.selected().is_none());
        assert!(host.expanded().is_empty());
    }

    #[gpui::test]
    async fn cargo_dashboard_foreground_budget(cx: &mut TestAppContext) {
        const FOREGROUND_BUDGET: Duration = Duration::from_millis(250);
        const WARMUP_RUNS: usize = 3;
        const SAMPLE_COUNT: usize = 21;
        let projected = dashboard_projection(cx).await;
        let mut samples = Vec::with_capacity(SAMPLE_COUNT);
        // CI runs this measurement alone, after the concurrent correctness suite.
        // Fixture cloning is outside the original foreground reconciliation boundary.
        for iteration in 0..WARMUP_RUNS + SAMPLE_COUNT {
            let projected = projected.clone();
            let started = Instant::now();
            let mut host = LanguageToolTreeHost::default();
            host.replace_snapshot(projected);
            host.expand_all();
            let elapsed = started.elapsed();
            assert_eq!(host.visible_rows().len(), DASHBOARD_ROW_COUNT);
            assert_eq!(host.visible_rows_in_range(4_000..4_025).len(), 25);
            std::hint::black_box(host.visible_rows());
            if iteration >= WARMUP_RUNS {
                samples.push(elapsed);
            }
        }
        samples.sort_unstable();
        let median = samples.get(SAMPLE_COUNT / 2).expect("median sample");
        eprintln!(
            "cargo-dashboard-foreground-budget rows={DASHBOARD_ROW_COUNT} samples={SAMPLE_COUNT} min={:?} median={median:?} p95={:?} max={:?}",
            samples.first().expect("minimum sample"),
            samples.get(19).expect("p95 sample"),
            samples.last().expect("maximum sample"),
        );
        assert!(
            *median <= FOREGROUND_BUDGET,
            "10,000-row foreground reconciliation median {median:?} exceeds {FOREGROUND_BUDGET:?} across {SAMPLE_COUNT} samples"
        );
    }

    fn scaled_package(template: &CargoPackageModel, index: usize) -> CargoPackageModel {
        let package_name = format!("package-{index:04}");
        let package_root = format!("members/{package_name}");
        let mut package = template.clone();
        package.id = format!("{package_name} 0.1.0");
        package.name = package_name.clone();
        package.manifest_path = path(&format!("{package_root}/Cargo.toml"));
        package.is_default_member = index == 0;
        package.targets[0].name = package_name;
        package.targets[0].source_path = Some(path(&format!("{package_root}/src/main.rs")));
        package.targets[0].source_display_path = Some(format!("{package_root}/src/main.rs"));
        package
    }

    #[gpui::test]
    async fn cargo_dashboard_production_foreground_budget(cx: &mut TestAppContext) {
        use project::cargo_workspace_store::{
            CargoMetadataRequest, CargoMetadataRunner, CargoWorkspaceStore,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct MetadataRunner {
            output: Vec<u8>,
            calls: Arc<AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl CargoMetadataRunner for MetadataRunner {
            async fn run(&self, _: CargoMetadataRequest) -> anyhow::Result<Vec<u8>> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(self.output.clone())
            }
        }

        impl project::rust_test_provider::RustTestDiscoveryRunner for MetadataRunner {
            fn discover(
                &self,
                _: project::rust_test_provider::RustTestDiscoveryRequest,
            ) -> futures::future::BoxFuture<
                'static,
                anyhow::Result<project::rust_test_provider::RustTestDiscoveryCapture>,
            > {
                Box::pin(async {
                    Ok(project::rust_test_provider::RustTestDiscoveryCapture {
                        toolchain: "fixture".into(),
                        cargo_messages: String::new(),
                        listings: Vec::new(),
                    })
                })
            }
        }

        const FOREGROUND_BUDGET: Duration = Duration::from_millis(250);
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
            let mut preset = CargoPreset::ephemeral_default(crate::CargoSubcommand::Build);
            preset.id = "budget-release".into();
            preset.profile = Some("release".into());
            CargoPresetSettings::override_global(
                CargoPresetSettings {
                    presets: [(preset.id.clone(), preset)].into_iter().collect(),
                    diagnostics: Vec::new(),
                },
                cx,
            );
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/workspace",
            serde_json::json!({
                "Cargo.toml": "[workspace]\nmembers = [\"members/*\"]",
            }),
        )
        .await;
        let project = Project::test(fs, ["/workspace".as_ref()], cx).await;
        let mut metadata: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../project/test_data/cargo_workspace/workspace-v1.json"
        ))
        .expect("metadata fixture");
        let template = metadata["packages"][0].clone();
        let packages: Vec<_> = (0..1_000)
            .map(|index| {
                let mut package = template.clone();
                let name = format!("package-{index:04}");
                package["name"] = name.clone().into();
                package["id"] = format!("path+file:///workspace/members/{name}#0.1.0").into();
                package["manifest_path"] = format!("/workspace/members/{name}/Cargo.toml").into();
                package["dependencies"] = serde_json::json!([]);
                package["targets"] = serde_json::json!([{
                    "kind": ["lib"], "crate_types": ["lib"], "name": name,
                    "src_path": format!("/workspace/members/{name}/src/lib.rs"),
                    "edition": "2024", "doc": true, "doctest": true, "test": true
                }]);
                package["features"] = serde_json::json!({
                    "first": [], "second": [], "third": [], "fourth": [],
                    "fifth": [], "sixth": [], "seventh": [], "eighth": []
                });
                package
            })
            .collect();
        let members: Vec<_> = packages
            .iter()
            .map(|package| package["id"].clone())
            .collect();
        metadata["workspace_members"] = members.clone().into();
        metadata["workspace_default_members"] = members.into();
        metadata["packages"] = packages.into();
        metadata["resolve"] = serde_json::Value::Null;
        let calls = Arc::new(AtomicUsize::new(0));
        let runner = Arc::new(MetadataRunner {
            output: serde_json::to_vec(&metadata).expect("serialize metadata"),
            calls: calls.clone(),
        });
        project.update(cx, |project, cx| {
            let store = CargoWorkspaceStore::local_with_runner(
                project.worktree_store(),
                project.environment().clone(),
                runner.clone(),
            );
            // Rust discovery observes this metadata too; its process runner must not
            // probe the real filesystem from this deterministic panel fixture.
            project.rust_test_provider_store().update(cx, |store, _| {
                store
                    .set_discovery_runner_for_test(runner)
                    .expect("deterministic Rust discovery runner");
            });
            project
                .cargo_workspace_store()
                .update(cx, |current, _| *current = store);
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = CargoPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("Cargo panel");
        cx.run_until_parked();
        // Project startup can queue Rust test discovery that also reads Cargo metadata.
        // Settle that owner before measuring panel-specific refreshes.
        cx.background_executor
            .timer(Duration::from_millis(151))
            .await;
        cx.run_until_parked();
        let baseline_calls = calls.load(Ordering::SeqCst);
        let mut baseline_ids = None;
        for phase in ["refresh", "notice", "configuration"] {
            panel.update(cx, |panel, _| {
                panel.projection_timings = CargoProjectionTimings::default()
            });
            let started = Instant::now();
            let scheduling_started = Instant::now();
            panel.update(cx, |panel, cx| match phase {
                "refresh" => panel.refresh(cx),
                "notice" => panel.set_action_notice("budget notice", cx),
                "configuration" => {
                    panel.active = true;
                    panel.set_active_preset(
                        Some("budget-release".into()),
                        CargoSafeSelectionState::default(),
                        cx,
                    );
                }
                _ => unreachable!(),
            });
            let scheduling = scheduling_started.elapsed();
            cx.background_executor
                .timer(Duration::from_millis(151))
                .await;
            cx.run_until_parked();
            let expected_calls = baseline_calls
                + match phase {
                    "refresh" | "notice" => 1,
                    "configuration" => 2,
                    _ => unreachable!(),
                };
            assert_eq!(
                calls.load(Ordering::SeqCst),
                expected_calls,
                "{phase}: only explicit refresh and preset changes should request metadata"
            );
            let paint_started = Instant::now();
            let (row_count, ids) = panel.update(cx, |panel, cx| {
                assert!(
                    panel.execution_host_available,
                    "{phase} did not apply its result"
                );
                assert_eq!(
                    panel.last_snapshot.as_ref().expect("snapshot").workspaces[0]
                        .members
                        .len(),
                    1_000
                );
                assert_eq!(
                    panel.active_configuration.profile.as_deref(),
                    (phase == "configuration").then_some("release")
                );
                panel.host.expand_all();
                panel.host.select_last();
                panel.reveal_selection(ScrollStrategy::Bottom);
                cx.notify();
                (
                    panel.host.visible_rows().len(),
                    panel
                        .host
                        .visible_rows()
                        .iter()
                        .filter(|row| !row.id.0.ends_with(":preset-notice"))
                        .map(|row| row.id.clone())
                        .collect::<Vec<_>>(),
                )
            });
            cx.draw(
                gpui::point(px(0.0), px(0.0)),
                gpui::size(px(360.0), px(720.0)),
                |_, _| panel.clone().into_any_element(),
            );
            let painting = paint_started.elapsed();
            let elapsed = started.elapsed();
            let (foreground, background) = panel.read_with(cx, |panel, _| {
                let timings = &panel.projection_timings;
                (
                    scheduling
                        + timings.refresh
                        + timings.admission
                        + timings.application
                        + painting,
                    Duration::from_nanos(timings.background_nanoseconds.load(Ordering::SeqCst)),
                )
            });
            panel.read_with(cx, |panel, _| {
                let scroll = panel.host.scroll_handle().0.borrow();
                let measured = scroll.last_item_size.expect("painted list dimensions");
                assert!(measured.item.height > px(0.0) && measured.item.height <= px(720.0),
                    "{phase}: unbounded viewport {:?}", measured.item);
                assert!(measured.contents.height > measured.item.height);
                let offset = scroll.base_handle.offset().y;
                let bottom = measured.item.height - measured.contents.height;
                assert!((offset - bottom).abs() <= px(1.0),
                    "{phase}: bottom viewport was not painted: offset={offset:?}, expected={bottom:?}");
            });
            assert!(row_count >= 10_000, "fixture has only {row_count} rows");
            if let Some(expected) = &baseline_ids {
                assert!(
                    ids == *expected,
                    "stable rows changed identity during {phase}"
                );
            } else {
                baseline_ids = Some(ids);
            }
            assert!(
                foreground <= FOREGROUND_BUDGET,
                "{phase}: production foreground scheduling, admission, reconciliation and paint took {foreground:?}, exceeding {FOREGROUND_BUDGET:?}; background={background:?}, total={elapsed:?}"
            );
            eprintln!(
                "cargo-dashboard-production-budget phase={phase} rows={row_count} foreground_us={} background_us={} total_ms={}",
                foreground.as_micros(),
                background.as_micros(),
                elapsed.as_millis()
            );
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            baseline_calls + 2,
            "refresh and preset changes must reach the metadata runner"
        );
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[test]
    fn configuration_projection_distinguishes_active_host_and_cargo_targets() {
        let mut snapshot = snapshot();
        snapshot.workspaces[0].configuration = CargoWorkspaceConfiguration {
            profiles: vec![
                CargoProfileModel {
                    name: "dev".to_string(),
                    origin: CargoProfileOrigin::Implicit,
                },
                CargoProfileModel {
                    name: "ship".to_string(),
                    origin: CargoProfileOrigin::Declared,
                },
            ],
            declared_toolchain: Some(CargoToolchainModel {
                source_path: path("rust-toolchain.toml"),
                format: CargoToolchainFormat::Toml,
                channel: Some("stable".to_string()),
                components: vec!["clippy".to_string()],
                targets: vec!["wasm32-unknown-unknown".to_string()],
            }),
            host_compiler: CargoHostCompilerModel {
                status: CargoHostCompilerStatus::Available,
                release: Some("1.90.0".to_string()),
                host_target: Some("aarch64-apple-darwin".to_string()),
                stale: false,
            },
            cargo_target: CargoTargetConfiguration::UnresolvedCargoDefault,
            diagnostics: Vec::new(),
            completeness: CargoConfigurationCompleteness::Complete,
        };
        let active = CargoActiveConfiguration {
            scope: "package member".to_string(),
            profile: Some("ship".to_string()),
            native_profiling: false,
            selected_features: vec!["serde".to_string()],
            default_features: Some(false),
            explicit_target: Some("wasm32-unknown-unknown".to_string()),
            target_selector: Some("bin member".to_string()),
            environment_keys: vec!["RUSTFLAGS".to_string(), "CARGO_HOME".to_string()],
            notice: None,
        };
        let (first, provider) = CargoTreeProvider::project_with_configuration(&snapshot, &active);
        let (second, _) = CargoTreeProvider::project_with_configuration(&snapshot, &active);
        assert_eq!(collect_ids(&first.roots), collect_ids(&second.roots));

        let explicit = find_node(&first.roots, "Explicit target")
            .expect("explicit target row should be projected");
        assert_eq!(
            explicit.secondary_label.as_deref(),
            Some("wasm32-unknown-unknown")
        );
        let host = find_node(&first.roots, "Host compiler")
            .expect("host compiler row should be projected");
        assert!(
            host.secondary_label
                .as_deref()
                .is_some_and(|label| label.contains("aarch64-apple-darwin"))
        );
        let cargo_target = find_node(&first.roots, "Cargo target resolution")
            .expect("Cargo target row should be projected");
        assert_eq!(
            cargo_target.secondary_label.as_deref(),
            Some("unresolved Cargo default")
        );
        let environment = find_node(&first.roots, "Environment keys")
            .expect("environment-key row should be projected");
        assert_eq!(
            environment.secondary_label.as_deref(),
            Some("CARGO_HOME, RUSTFLAGS")
        );
        let toolchain = find_node(&first.roots, "Declared toolchain")
            .expect("toolchain row should be projected");
        assert_eq!(
            provider.navigation(&toolchain.id),
            Some(&path("rust-toolchain.toml"))
        );
        assert!(
            collect_ids(&first.roots)
                .iter()
                .all(|id| !id.0.contains("RUSTFLAGS"))
        );
    }

    #[test]
    fn configuration_probe_failure_retains_safe_host_facts_as_stale() {
        let mut previous = snapshot();
        previous.workspaces[0].configuration.host_compiler = CargoHostCompilerModel {
            status: CargoHostCompilerStatus::Available,
            release: Some("1.90.0".to_string()),
            host_target: Some("aarch64-apple-darwin".to_string()),
            stale: false,
        };
        let mut current = snapshot();
        current.revision = 2;
        current.completeness = CargoSnapshotCompleteness::Partial;
        current.workspaces[0].configuration.host_compiler.status = CargoHostCompilerStatus::Failed;
        current.workspaces[0].configuration.completeness = CargoConfigurationCompleteness::Partial;
        current = retain_stale_workspaces(current, Some(&previous));
        assert!(current.workspaces[0].configuration.host_compiler.stale);
        assert_eq!(
            current.workspaces[0]
                .configuration
                .host_compiler
                .host_target
                .as_deref(),
            Some("aarch64-apple-darwin")
        );
        let (projected, _) = CargoTreeProvider::project(&current);
        assert!(matches!(
            projected.status,
            LanguageToolProviderStatus::Partial(message)
                if message.contains("configuration facts")
        ));
    }

    #[test]
    fn partial_refresh_retains_matching_workspace_as_stale() {
        let previous = snapshot();
        let mut current = CargoWorkspaceSnapshot {
            revision: 2,
            input_fingerprint: 2,
            completeness: CargoSnapshotCompleteness::Partial,
            workspaces: Vec::new(),
            failures: vec![CargoCandidateFailure {
                manifest_path: path("Cargo.toml"),
                category: CargoWorkspaceErrorCategory::CargoFailed,
                message: "dependency resolution failed".to_string(),
                has_stale_model: false,
            }],
        };

        current = retain_stale_workspaces(current, Some(&previous));
        assert_eq!(current.workspaces, previous.workspaces);
        assert!(current.failures[0].has_stale_model);

        let removed = retain_stale_workspaces(
            CargoWorkspaceSnapshot {
                revision: 3,
                input_fingerprint: 3,
                completeness: CargoSnapshotCompleteness::Complete,
                workspaces: Vec::new(),
                failures: Vec::new(),
            },
            Some(&previous),
        );
        assert!(removed.workspaces.is_empty());
    }

    #[test]
    fn unopened_panel_host_is_dormant_and_does_not_request_metadata() {
        let host = LanguageToolTreeHost::default();
        assert_eq!(host.status(), &LanguageToolTreeStatus::Dormant);
    }

    fn collect_ids(nodes: &[LanguageToolNode]) -> Vec<LanguageToolNodeId> {
        let mut ids = Vec::new();
        for node in nodes {
            ids.push(node.id.clone());
            ids.extend(collect_ids(&node.children));
        }
        ids
    }

    fn find_node<'a>(nodes: &'a [LanguageToolNode], label: &str) -> Option<&'a LanguageToolNode> {
        for node in nodes {
            if node.label == label {
                return Some(node);
            }
            if let Some(found) = find_node(&node.children, label) {
                return Some(found);
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_profile_action_real_receiver_preserves_operands_and_custom_shell_tasks(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::Focusable as _;
        use project::trusted_worktrees::{DbTrustedPaths, PathTrust, TrustedWorktrees};
        use std::os::unix::fs::PermissionsExt as _;

        async fn refresh_external_fixture_metadata(
            project: &Entity<Project>,
            manifest: &ProjectPath,
            root: &std::path::Path,
            phase: &str,
            cx: &mut TestAppContext,
        ) {
            let (discovery, filesystem, worktree, store) = project.read_with(cx, |project, cx| {
                (
                    project.rust_test_provider_store().clone(),
                    project.fs().clone(),
                    project
                        .worktrees(cx)
                        .find(|tree| tree.read(cx).id() == manifest.worktree_id)
                        .expect("original external fixture Worktree"),
                    project.cargo_workspace_store().clone(),
                )
            });
            let lock_path: Arc<RelPath> =
                Arc::from(RelPath::from_unix_str("Cargo.lock").expect("literal Cargo input"));
            let deadline = cx.background_executor.timer(Duration::from_secs(10));
            let settling = async {
                loop {
                    while !discovery.read_with(cx, |store, _| store.discovery_idle_for_test())
                        || worktree
                            .read_with(cx, |tree, _| tree.scan_id() != tree.completed_scan_id())
                    {
                        cx.background_executor
                            .timer(Duration::from_millis(10))
                            .await;
                    }
                    let metadata = filesystem
                        .metadata(&root.join("Cargo.lock"))
                        .await
                        .expect("actual external fixture Cargo.lock metadata")
                        .expect("actual discovery retains Cargo.lock");
                    let entry = worktree
                        .update(cx, |worktree, cx| {
                            worktree
                                .as_local()
                                .expect("original local Worktree")
                                .refresh_entry(lock_path.clone(), None, cx)
                        })
                        .await
                        .expect("actual external fixture lock publication")
                        .expect("actual Cargo.lock indexed");
                    assert_eq!(
                        (entry.inode, entry.mtime, entry.size),
                        (metadata.inode, Some(metadata.mtime), metadata.len),
                        "Actual discovery lock must be indexed before public metadata admission"
                    );
                    cx.run_until_parked();
                    while !discovery.read_with(cx, |store, _| store.discovery_idle_for_test())
                        || worktree
                            .read_with(cx, |tree, _| tree.scan_id() != tree.completed_scan_id())
                    {
                        cx.background_executor
                            .timer(Duration::from_millis(10))
                            .await;
                    }
                    let metadata = filesystem
                        .metadata(&root.join("Cargo.lock"))
                        .await
                        .expect("retired external discovery lock metadata")
                        .expect("original Cargo.lock remains present");
                    worktree.read_with(cx, |tree, _| {
                        let indexed = tree
                            .entry_for_path(&lock_path)
                            .expect("published Cargo.lock remains indexed");
                        assert_eq!(
                            (indexed.inode, indexed.mtime, indexed.size),
                            (metadata.inode, Some(metadata.mtime), metadata.len),
                            "Retired discovery and original Worktree agree on final lock identity"
                        );
                        assert_eq!(tree.scan_id(), tree.completed_scan_id());
                    });
                    let reviewed = store
                        .read_with(cx, |store, cx| {
                            store.prepare_reviewed_metadata_import(manifest.clone(), cx)
                        })
                        .expect("public metadata admission after original discovery retirement");
                    let metadata =
                        include_str!("../../project/test_data/cargo_workspace/workspace-v1.json")
                            .replace("/workspace", root.to_str().expect("root UTF-8"));
                    store
                        .update(cx, |store, cx| {
                            store.admit_reviewed_metadata_import(
                                reviewed,
                                metadata.into_bytes(),
                                cx,
                            )
                        })
                        .await
                        .expect("current external selection metadata");
                    cx.run_until_parked();
                    while !discovery.read_with(cx, |store, _| store.discovery_idle_for_test())
                        || worktree
                            .read_with(cx, |tree, _| tree.scan_id() != tree.completed_scan_id())
                    {
                        cx.background_executor
                            .timer(Duration::from_millis(10))
                            .await;
                    }
                    if store.read_with(cx, |store, cx| {
                        store
                            .cached_snapshot(cx)
                            .is_some_and(|(_, current)| current)
                    }) {
                        break;
                    }
                }
            };
            futures::pin_mut!(settling);
            match futures::future::select(settling, deadline).await {
                futures::future::Either::Left(((), _)) => {}
                futures::future::Either::Right(_) => panic!(
                    "Original discovery, Cargo.lock, Worktree scan and public metadata admission did not settle ({phase})"
                ),
            }
        }

        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
            tasks_ui::init(cx);
            terminal_view::init(cx);
            project::trusted_worktrees::init(DbTrustedPaths::default(), cx);
        });
        let directory = tempfile::tempdir().expect("controlled external collector");
        let directory_path = std::fs::canonicalize(directory.path()).expect("canonical fixture");
        let root = directory_path.join("project with spaces $literal");
        let installed = directory_path.join("installed");
        for path in ["member-one/src", "member-two/src", "target"] {
            std::fs::create_dir_all(root.join(path)).expect("project directories");
        }
        std::fs::create_dir(&installed).expect("controlled installed external collector");
        for (path, contents) in [
            (
                "Cargo.toml",
                "[workspace]\nmembers=['member-one','member-two']\nresolver='2'\n",
            ),
            (".gitignore", "target/\n"),
            (
                "Cargo.lock",
                "# This file is automatically @generated by Cargo.\n# It is not intended for manual editing.\nversion = 4\n\n[[package]]\nname = \"member-one\"\nversion = \"0.1.0\"\n\n[[package]]\nname = \"member-two\"\nversion = \"0.2.0\"\n",
            ),
            (
                "member-one/Cargo.toml",
                "[package]\nname='member-one'\nversion='0.1.0'\nedition='2024'\n",
            ),
            ("member-one/src/main.rs", "fn main() {}\n"),
            ("member-one/src/lib.rs", "pub fn retained() {}\n"),
            (
                "member-two/Cargo.toml",
                "[package]\nname='member-two'\nversion='0.2.0'\nedition='2021'\n",
            ),
            ("member-two/src/lib.rs", "pub fn other() {}\n"),
        ] {
            std::fs::write(root.join(path), contents).expect("fixed project inputs");
        }
        let collector = installed.join("receiver");
        std::fs::write(&collector,
            "#!/bin/sh\n[ \"$PROFILE_ENV_VALUE\" = 'value with spaces $literal' ] || exit 71\nprintf '%s\\0' \"$@\" >> \"$PROFILE_RECEIVED_ARGV\" || exit 72\nprintf '%s\\0' \"$(pwd -P)\" >> \"$PROFILE_RECEIVED_CWD\" || exit 73\nprintf '<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>' > \"$1\" || exit 74\nprintf 'completed\\n' >> \"$PROFILE_RUN_LOG\" || exit 75\n"
        ).expect("controlled actual external collector");
        std::fs::set_permissions(&collector, std::fs::Permissions::from_mode(0o700))
            .expect("collector executable");
        let argv_receipt = installed.join("received.argv");
        let cwd_receipt = installed.join("received.cwd");
        let run_log = installed.join("runs.log");
        let injection = installed.join("unexpected-shell-evaluation");
        let artifact_path = "target/profile with spaces $literal.svg";
        let operands = vec![
            artifact_path.to_string(),
            root.join("member-one/src/main.rs")
                .to_str()
                .expect("UTF-8 workload source")
                .to_string(),
            "quoted 'single' and \"double\" operands".to_string(),
            "dollar $PROFILE_ARG_TOKEN".to_string(),
            "semi; /bin/true".to_string(),
            format!(
                "$(/usr/bin/touch '{}')",
                injection.to_str().expect("private sentinel")
            ),
            String::new(),
        ];
        let content = settings::CargoPresetSettingsContent {
            subcommand: Some("run".into()),
            working_directory: Some("workspace".into()),
            save: Some("none".into()),
            use_new_terminal: Some(true),
            allow_concurrent_runs: Some(true),
            profiler: Some(settings::CargoProfilerSettingsContent {
                command: Some(collector.to_str().expect("collector UTF-8").into()),
                args: Some(operands.clone()),
                artifact_path: Some(artifact_path.into()),
                artifact_kind: Some("svg".into()),
                artifact_max_bytes: Some(4096),
            }),
            environment: Some(collections::HashMap::from_iter([
                (
                    "PROFILE_ENV_VALUE".into(),
                    "value with spaces $literal".into(),
                ),
                (
                    "PROFILE_ARG_TOKEN".into(),
                    "must remain literal in external operands".into(),
                ),
                (
                    "PROFILE_RECEIVED_ARGV".into(),
                    argv_receipt.to_str().expect("argv receipt").into(),
                ),
                (
                    "PROFILE_RECEIVED_CWD".into(),
                    cwd_receipt.to_str().expect("cwd receipt").into(),
                ),
                (
                    "PROFILE_RUN_LOG".into(),
                    run_log.to_str().expect("run log").into(),
                ),
            ])),
            ..Default::default()
        };
        let zero_collector = installed.join("zero argument receiver");
        let zero_argc_receipt = installed.join("zero.argc");
        let zero_cwd_receipt = installed.join("zero.cwd");
        let zero_artifact_path = "target/zero argument profile.svg";
        let zero_artifact = root.join(zero_artifact_path);
        std::fs::write(
            &zero_collector,
            "#!/bin/sh\n[ \"$PROFILE_ZERO_VALUE\" = 'zero argv value $literal' ] || exit 81\nprintf '%s\\n' \"$#\" >> \"$PROFILE_ZERO_ARGC\" || exit 82\nprintf '%s\\0' \"$(pwd -P)\" >> \"$PROFILE_ZERO_CWD\" || exit 83\nprintf '<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>' > \"$PROFILE_ZERO_OUTPUT\" || exit 84\n",
        )
        .expect("actual installed zero-argument receiver");
        std::fs::set_permissions(&zero_collector, std::fs::Permissions::from_mode(0o700))
            .expect("zero-argument receiver executable");
        let zero_content = settings::CargoPresetSettingsContent {
            subcommand: Some("run".into()),
            working_directory: Some("workspace".into()),
            save: Some("none".into()),
            use_new_terminal: Some(true),
            allow_concurrent_runs: Some(true),
            profiler: Some(settings::CargoProfilerSettingsContent {
                command: Some(zero_collector.to_str().expect("zero receiver UTF-8").into()),
                args: Some(Vec::new()),
                artifact_path: Some(zero_artifact_path.into()),
                artifact_kind: Some("svg".into()),
                artifact_max_bytes: Some(4096),
            }),
            environment: Some(collections::HashMap::from_iter([
                (
                    "PROFILE_ZERO_VALUE".into(),
                    "zero argv value $literal".into(),
                ),
                (
                    "PROFILE_ZERO_ARGC".into(),
                    zero_argc_receipt.to_str().expect("argc receipt").into(),
                ),
                (
                    "PROFILE_ZERO_CWD".into(),
                    zero_cwd_receipt.to_str().expect("cwd receipt").into(),
                ),
                (
                    "PROFILE_ZERO_OUTPUT".into(),
                    zero_artifact.to_str().expect("zero artifact").into(),
                ),
            ])),
            ..Default::default()
        };
        let user_settings =
            serde_json::json!({"cargo":{"schema_version":crate::CARGO_PRESET_SCHEMA_VERSION,
            "presets":{"external-receiver":content,"external-zero-receiver":zero_content}}})
            .to_string();
        cx.update(|cx| {
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(&user_settings, cx)
                    .expect("explicit user external preset");
            });
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
        });
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let project = Project::test_with_worktree_trust(filesystem, [root.as_path()], cx).await;
        let (worktree_store, worktree_id, store) = project.read_with(cx, |project, cx| {
            (
                project.worktree_store(),
                project
                    .visible_worktrees(cx)
                    .next()
                    .expect("worktree")
                    .read(cx)
                    .id(),
                project.cargo_workspace_store().clone(),
            )
        });
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        cx.run_until_parked();
        let manifest = ProjectPath {
            worktree_id,
            path: Arc::from(RelPath::from_unix_str("Cargo.toml").expect("manifest")),
        };
        refresh_external_fixture_metadata(&project, &manifest, &root, "initial", cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let terminal_panel = terminal_view::terminal_panel::TerminalPanel::load(
            workspace.downgrade(),
            cx.update(|window, cx| window.to_async(cx)),
        )
        .await
        .expect("actual production terminal provider");
        let panel = CargoPanel::load(
            workspace.downgrade(),
            cx.update(|window, cx| window.to_async(cx)),
        )
        .await
        .expect("selected Cargo panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(terminal_panel, window, cx);
            workspace.add_panel(panel.clone(), window, cx);
        });
        let snapshot = store.read_with(cx, |store, cx| {
            let (snapshot, current) = store.cached_snapshot(cx).expect("metadata snapshot");
            assert!(current);
            snapshot.clone()
        });
        panel.update(cx, |panel, cx| {
            panel.accept_snapshot(panel.host.refresh_generation(), snapshot, cx)
        });
        cx.run_until_parked();
        let selected = panel.read_with(cx, |panel, _| {
            panel
                .provider
                .action_selections
                .iter()
                .find(|(_, selection)| {
                    matches!(
                        selection.node_kind,
                        CargoActionNodeKind::Target(CargoActionTargetKind::Binary)
                    )
                })
                .map(|(id, _)| id.clone())
                .expect("actual selected Binary")
        });
        panel.update(cx, |panel, cx| {
            panel.set_active_preset(
                Some("external-receiver".into()),
                CargoSafeSelectionState::default(),
                cx,
            );
            panel.update_selection(
                |host| {
                    host.expand_all();
                    host.select(selected);
                },
                cx,
            );
        });
        refresh_external_fixture_metadata(&project, &manifest, &root, "nonempty preset", cx).await;
        let snapshot = store.read_with(cx, |store, cx| {
            let (snapshot, current) = store.cached_snapshot(cx).expect("current preset metadata");
            assert!(current);
            snapshot.clone()
        });
        panel.update(cx, |panel, cx| {
            panel.accept_snapshot(panel.host.refresh_generation(), snapshot, cx)
        });
        cx.run_until_parked();
        assert!(
            panel.read_with(cx, |panel, _| panel.execution_host_available),
            "active preset configuration must finish real projection before selected invocation"
        );
        let mut stable_id = None;
        let expected_argv: Vec<u8> = operands
            .iter()
            .flat_map(|operand| operand.as_bytes().iter().copied().chain(std::iter::once(0)))
            .collect();
        let expected_cwd: Vec<u8> = root
            .to_str()
            .expect("actual cwd UTF-8")
            .as_bytes()
            .iter()
            .copied()
            .chain(std::iter::once(0))
            .collect();
        for route in 0..3 {
            workspace.update_in(cx, |workspace, window, cx| {
                if route == 0 {
                    workspace
                        .focus_panel::<CargoPanel>(window, cx)
                        .expect("Cargo focus");
                } else {
                    window.focus(&workspace.focus_handle(cx), cx);
                }
            });
            if route == 0 {
                cx.dispatch_action(ProfileSelected);
            } else {
                cx.dispatch_action(tasks_ui::Rerun {
                    reevaluate_context: route == 2,
                    ..Default::default()
                });
            }
            for _ in 0..200 {
                match std::fs::read_to_string(&run_log) {
                    Ok(runs) if runs.lines().count() == route + 1 => break,
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => panic!("external run receipt could not be read: {error}"),
                }
                cx.background_executor
                    .timer(Duration::from_millis(50))
                    .await;
            }
            assert_eq!(
                std::fs::read(&argv_receipt).expect("actual external child argv"),
                expected_argv.repeat(route + 1),
                "initial/cached/template argv reception"
            );
            assert_eq!(
                std::fs::read(&cwd_receipt).expect("actual external child cwd"),
                expected_cwd.repeat(route + 1)
            );
            assert!(
                !injection.exists(),
                "external data operand must not execute shell substitution"
            );
            let (_, history) = project.read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("Task history")
                    .read(cx)
                    .last_scheduled_task(None)
                    .expect("actual external profile history")
            });
            if let Some(stable_id) = &stable_id {
                assert_eq!(&history.id, stable_id);
            } else {
                stable_id = Some(history.id.clone());
            }
            let terminal = project.read_with(cx, |project, cx| {
                project
                    .local_terminal_handles()
                    .iter()
                    .rev()
                    .filter_map(|handle| handle.upgrade())
                    .find(|terminal| {
                        terminal
                            .read(cx)
                            .task()
                            .is_some_and(|state| state.spawned_task.id == history.id)
                    })
                    .expect("actual profile terminal")
            });
            let completed = futures::future::select(
                Box::pin(
                    terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx)),
                ),
                Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
            )
            .await;
            match completed {
                futures::future::Either::Left((status, _)) => {
                    assert!(status.expect("actual external kernel status").success())
                }
                futures::future::Either::Right(_) => panic!("actual external profile did not exit"),
            }
            for _ in 0..200 {
                if workspace.read_with(cx, |workspace, cx| {
                    workspace
                        .active_item(cx)
                        .and_then(|item| item.project_path(cx))
                        .is_some_and(|path| path.path.as_unix_str() == artifact_path)
                }) {
                    break;
                }
                cx.background_executor
                    .timer(Duration::from_millis(50))
                    .await;
            }
            assert!(
                workspace.read_with(cx, |workspace, cx| workspace
                    .active_item(cx)
                    .and_then(|item| item.project_path(cx))
                    .is_some_and(|path| path.path.as_unix_str() == artifact_path)),
                "successful external selected action opens its bounded declared artifact"
            );
            workspace
                .update_in(cx, |workspace, window, cx| {
                    workspace.active_pane().update(cx, |pane, cx| {
                        pane.close_active_item(&workspace::CloseActiveItem::default(), window, cx)
                    })
                })
                .await
                .expect("close the actual clean artifact before registered Rerun");
            assert!(
                workspace.read_with(cx, |workspace, cx| workspace
                    .active_item(cx)
                    .and_then(|item| item.project_path(cx))
                    .is_none_or(|path| path.path.as_unix_str() != artifact_path)),
                "each registered Rerun must reopen a currently absent artifact"
            );
        }
        assert!(!zero_artifact.exists());
        panel.update(cx, |panel, cx| {
            panel.set_active_preset(
                Some("external-zero-receiver".into()),
                CargoSafeSelectionState::default(),
                cx,
            );
        });
        refresh_external_fixture_metadata(&project, &manifest, &root, "zero-argument preset", cx)
            .await;
        let snapshot = store.read_with(cx, |store, cx| {
            let (snapshot, current) = store.cached_snapshot(cx).expect("zero preset metadata");
            assert!(current);
            snapshot.clone()
        });
        panel.update(cx, |panel, cx| {
            panel.accept_snapshot(panel.host.refresh_generation(), snapshot, cx)
        });
        cx.run_until_parked();
        let selected_zero = panel.read_with(cx, |panel, _| {
            panel
                .provider
                .action_selections
                .iter()
                .find(|(_, selection)| {
                    matches!(
                        selection.node_kind,
                        CargoActionNodeKind::Target(CargoActionTargetKind::Binary)
                    )
                })
                .map(|(id, _)| id.clone())
                .expect("actual zero-argv Binary selection")
        });
        panel.update(cx, |panel, cx| {
            panel.update_selection(
                |host| {
                    host.expand_all();
                    host.select(selected_zero);
                },
                cx,
            );
        });
        cx.run_until_parked();
        assert!(panel.read_with(cx, |panel, _| panel.execution_host_available));
        let mut zero_stable_id = None;
        for route in 0..3 {
            cx.update(|window, cx| panel.read(cx).focus_handle(cx).focus(window, cx));
            if route == 0 {
                cx.dispatch_action(ProfileSelected);
            } else {
                cx.dispatch_action(tasks_ui::Rerun {
                    reevaluate_context: route == 2,
                    ..Default::default()
                });
            }
            for _ in 0..200 {
                match std::fs::read_to_string(&zero_argc_receipt) {
                    Ok(received) if received.lines().count() == route + 1 => break,
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => panic!("zero-argument receipt could not be read: {error}"),
                }
                cx.background_executor
                    .timer(Duration::from_millis(50))
                    .await;
            }
            let (_, history) = project.read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("Task history")
                    .read(cx)
                    .last_scheduled_task(None)
                    .expect("zero-argv history")
            });
            assert!(history.original_task().literal_arguments);
            assert!(history.resolved.args.is_empty());
            assert_eq!(history.resolved.literal_prefix_args, 0);
            assert_eq!(history.resolved.command.as_deref(), zero_collector.to_str());
            if let Some(id) = &zero_stable_id {
                assert_eq!(&history.id, id);
            } else {
                zero_stable_id = Some(history.id.clone());
            }
            let terminal = project.read_with(cx, |project, cx| {
                project
                    .local_terminal_handles()
                    .iter()
                    .rev()
                    .filter_map(|handle| handle.upgrade())
                    .find(|terminal| {
                        terminal
                            .read(cx)
                            .task()
                            .is_some_and(|state| state.spawned_task.id == history.id)
                    })
                    .expect("actual zero-argument terminal")
            });
            match futures::future::select(
                Box::pin(
                    terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx)),
                ),
                Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
            )
            .await
            {
                futures::future::Either::Left((status, _)) => {
                    assert!(status.expect("actual zero-argument kernel exit").success())
                }
                futures::future::Either::Right(_) => {
                    panic!("zero-argument external Task did not exit")
                }
            }
            assert_eq!(
                std::fs::read(&zero_argc_receipt).expect("real zero-argv reception"),
                b"0\n".repeat(route + 1)
            );
            assert_eq!(
                std::fs::read(&zero_cwd_receipt).expect("actual zero-argv cwd"),
                expected_cwd.repeat(route + 1)
            );
            for _ in 0..200 {
                if workspace.read_with(cx, |workspace, cx| {
                    workspace
                        .active_item(cx)
                        .and_then(|item| item.project_path(cx))
                        .is_some_and(|path| path.path.as_unix_str() == zero_artifact_path)
                }) {
                    break;
                }
                cx.background_executor
                    .timer(Duration::from_millis(50))
                    .await;
            }
            assert!(
                workspace.read_with(cx, |workspace, cx| workspace
                    .active_item(cx)
                    .and_then(|item| item.project_path(cx))
                    .is_some_and(|path| path.path.as_unix_str() == zero_artifact_path)),
                "zero-argv selected/cached/template run opens its newly produced bounded artifact"
            );
            workspace
                .update_in(cx, |workspace, window, cx| {
                    workspace.active_pane().update(cx, |pane, cx| {
                        pane.close_active_item(&workspace::CloseActiveItem::default(), window, cx)
                    })
                })
                .await
                .expect("close clean zero-argv artifact before Rerun");
            assert!(workspace.read_with(cx, |workspace, cx| {
                workspace
                    .active_item(cx)
                    .and_then(|item| item.project_path(cx))
                    .is_none_or(|path| path.path.as_unix_str() != zero_artifact_path)
            }));
        }
        assert_ne!(zero_stable_id, stable_id);
        assert_eq!(
            std::fs::read(&argv_receipt).expect("nonempty receiver retained"),
            expected_argv.repeat(3)
        );
        assert!(!injection.exists());
        let custom_output = installed.join("intentional-shell.txt");
        let custom_value = "intentional shell expansion survives";
        let custom = task::TaskTemplate {
            label: "authored custom shell fragments".into(),
            command: "/usr/bin/printf".into(),
            args: vec![
                "'%s'".into(),
                "\"$INTENTIONAL_PROFILE_VALUE\"".into(),
                ">".into(),
                format!("'{}'", custom_output.to_str().expect("custom output")),
            ],
            env: collections::HashMap::from_iter([(
                "INTENTIONAL_PROFILE_VALUE".into(),
                custom_value.into(),
            )]),
            cwd: Some(root.to_str().expect("cwd").into()),
            save: task::SaveStrategy::None,
            ..task::TaskTemplate::default()
        };
        assert!(!custom.literal_arguments);
        let custom = custom
            .resolve_task(
                "external-custom-shell-control",
                &task::TaskContext::default(),
            )
            .expect("existing custom shell Task resolution");
        let (sender, receiver) = futures::channel::oneshot::channel();
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                project::TaskSourceKind::UserInput,
                custom,
                false,
                move |result, _| sender.send(result).expect("actual custom Task callback"),
                window,
                cx,
            )
        });
        let completed = futures::future::select(
            Box::pin(receiver),
            Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
        )
        .await;
        match completed {
            futures::future::Either::Left((result, _)) => assert_eq!(
                result.expect("custom Task callback"),
                workspace::tasks::ScheduledTaskResult::Success
            ),
            futures::future::Either::Right(_) => panic!("actual authored custom Task did not exit"),
        }
        assert_eq!(
            std::fs::read_to_string(custom_output).expect("actual intentional shell result"),
            custom_value
        );
        cx.run_until_parked();
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }
    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_real_two_task_workflow_and_registered_rerun(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("rerun", None, false, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_real_two_task_rejects_expired_work_and_recovers(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("failed-build", None, false, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_real_two_task_cancelled_build_recovery(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("cancelled-build", None, false, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_real_two_task_cancelled_capture_recovery(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("cancelled-capture", None, false, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_real_two_task_superseded_build_recovery(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("superseded-build", None, false, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_real_two_task_declined_capture_recovery(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("declined-capture", None, false, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_real_two_task_dirty_consent_recovery(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("dirty-consent", None, false, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_real_two_task_held_capture_supersession_and_registered_rerun(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("superseded-capture", None, false, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_named_release_binary_and_registered_rerun(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("rerun", Some("release"), false, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_named_custom_example_and_registered_rerun(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow("rerun", Some("native-review"), true, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_selected_named_profile_manifest_change_refuses_and_recovers(
        cx: &mut gpui::TestAppContext,
    ) {
        check_native_selected_two_task_workflow(
            "profile-manifest",
            Some("native-review"),
            false,
            cx,
        )
        .await;
    }

    #[cfg(target_os = "macos")]
    async fn check_native_selected_two_task_workflow(
        scenario: &str,
        selected_profile: Option<&str>,
        example: bool,
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::Focusable as _;
        use project::trusted_worktrees::{DbTrustedPaths, PathTrust, TrustedWorktrees};
        use std::{cell::RefCell, os::unix::fs::PermissionsExt as _, rc::Rc};

        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
            tasks_ui::init(cx);
            terminal_view::init(cx);
            language_tools::profile_view::init(cx);
            project::trusted_worktrees::init(DbTrustedPaths::default(), cx);
        });
        let directory = tempfile::tempdir().expect("controlled native workflow");
        let directory_path = std::fs::canonicalize(directory.path()).expect("canonical fixture");
        let root = directory_path.join("workspace with spaces");
        let installed = directory_path.join("installed");
        for path in [
            ".git",
            "member-one/src",
            "member-one/examples",
            "member-two/src",
            "target",
        ] {
            std::fs::create_dir_all(root.join(path)).expect("worktree directories");
        }
        std::fs::create_dir(&installed).expect("installed controls outside worktree source");
        let workload_source = installed.join("selected-workload.rs");
        let workload_executable = installed.join("selected-workload");
        std::fs::write(
            &workload_source,
            r#"fn main() {
    assert_eq!(std::env::args().skip(1).collect::<Vec<_>>(),
        ["space value", "$literal", "quote'\"", "; no shell"]);
}
"#,
        )
        .expect("disposable selected Rust workload source");
        let executor = cx.background_executor.clone();
        cx.background_spawn({
            let workload_source = workload_source.clone();
            let workload_executable = workload_executable.clone();
            async move {
                project::cargo_workspace_store::compile_native_profile_test_workload(
                    &workload_source,
                    &workload_executable,
                    &executor,
                )
                .await
            }
        })
        .await
        .expect("installed passive compiler creates controlled Rust workload");
        let workload_metadata = std::fs::symlink_metadata(&workload_executable)
            .expect("actual compiled workload metadata");
        assert!(
            workload_metadata.is_file()
                && !workload_metadata.file_type().is_symlink()
                && (1..=256 * 1024 * 1024).contains(&workload_metadata.len())
                && workload_metadata.permissions().mode() & 0o111 != 0,
            "compiled workload is a bounded ordinary executable"
        );
        for (path, contents) in [
            (
                "Cargo.toml",
                "[workspace]\nmembers=['member-one','member-two']\nresolver='2'\n",
            ),
            (".gitignore", "target/\n"),
            (
                "Cargo.lock",
                "# This file is automatically @generated by Cargo.\n# It is not intended for manual editing.\nversion = 4\n\n[[package]]\nname = \"member-one\"\nversion = \"0.1.0\"\n\n[[package]]\nname = \"member-two\"\nversion = \"0.2.0\"\n",
            ),
            (
                "member-one/Cargo.toml",
                "[package]\nname='member-one'\nversion='0.1.0'\nedition='2024'\n",
            ),
            ("member-one/src/lib.rs", "pub fn retained() -> u32 { 7 }\n"),
            ("member-one/src/main.rs", "fn main() {}\n"),
            (
                "member-two/Cargo.toml",
                "[package]\nname='member-two'\nversion='0.2.0'\nedition='2021'\n",
            ),
            ("member-two/src/lib.rs", "pub fn other() -> u32 { 11 }\n"),
        ] {
            std::fs::write(root.join(path), contents).expect("fixed admitted source/config");
        }
        if selected_profile == Some("native-review") {
            std::fs::write(root.join("Cargo.toml"), "[workspace]\nmembers=['member-one','member-two']\nresolver='2'\n[profile.native-review]\ninherits='dev'\n")
                .expect("explicit declared custom profile fixture");
        }
        std::fs::copy(&workload_source, root.join("member-one/src/main.rs"))
            .expect("actual Cargo compiles the same admitted Rust workload source");
        if example {
            std::fs::copy(
                &workload_source,
                root.join("member-one/examples/profile-example.rs"),
            )
            .expect("actual Cargo compiles the explicitly selected example");
        }

        std::fs::write(
            root.join("member-one/build.rs"),
            r#"use std::{fs::{self, OpenOptions}, io::Write, path::PathBuf, time::Duration};
fn main() {
    let controls = PathBuf::from(std::env::var_os("NATIVE_WORKFLOW_CONTROLS").expect("owned controls"));
    let output = PathBuf::from(std::env::var_os("OUT_DIR").expect("actual Cargo OUT_DIR"));
    let invocation = output.ancestors().find(|path| path.file_name().and_then(|name| name.to_str())
        .is_some_and(|name| name.strip_prefix("zed-native-profile-build-")
            .is_some_and(|identifier| identifier.len() == 32 && identifier.bytes().all(|byte| byte.is_ascii_hexdigit()))))
        .expect("actual producing UUID directory");
    assert!(invocation.is_absolute());
    let log = controls.join("build.log");
    writeln!(OpenOptions::new().create(true).append(true).open(&log).expect("owned Build receipt"),
        "{}", invocation.to_str().expect("UTF-8 invocation")).expect("Build receipt write");
    let first = fs::read_to_string(&log).expect("own receipts").lines().count() == 1;
    if controls.join("hold-first-build").exists() && first {
        while !controls.join("release-first-build").exists() { std::thread::sleep(Duration::from_millis(20)); }
    }
    if controls.join("fail-build").exists() { std::process::exit(7); }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NATIVE_WORKFLOW_CONTROLS");
}
"#,
        )
        .expect("controlled real Cargo Build script, never fabricated report JSON");
        std::fs::write(
            installed.join("profile-template.json"),
            include_bytes!(
                "../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.json"
            ),
        )
        .expect("profile fixture");
        std::fs::write(installed.join("profile-template.syms.json"), include_bytes!("../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.syms.json")).expect("symbols fixture");
        let collector = installed.join("collector");
        std::fs::write(&collector, "#!/bin/sh\nbase=$(/usr/bin/dirname \"$0\")\ncase \"$1\" in\n--version) printf probe >> \"$base/probe.log\"; printf 'samply 0.13.1\\n';;\nrecord) [ -z \"${_NT_SYMBOL_PATH+x}\" ] && [ -z \"${ZED_PROFILE_TASK_TOKEN+x}\" ] && [ -z \"${ZED_PROFILE_BUILD_INVOCATION_ENV+x}\" ] && [ -z \"${ZED_PROFILE_NATIVE_INVOCATION_ENV+x}\" ] && [ -z \"${ZED_PROFILE_CARGO_RUNTIME_ENDPOINT+x}\" ] && [ -z \"${ZED_PROFILE_CARGO_RUNTIME_TOKEN+x}\" ] || exit 50\nout=\"${12}\"\nprintf '%s\\n' \"$out\" >> \"$base/capture.log\"\nprintf '%s\\0' \"$@\" > \"$base/$(/usr/bin/basename \"$out\").argv\"\nif [ -f \"$base/hold-first-capture\" ] && [ \"$(/usr/bin/wc -l < \"$base/capture.log\" | /usr/bin/tr -d ' ')\" = 1 ]; then printf '%s\\n' \"$$\" > \"$base/held-capture.pid\"; while [ ! -f \"$base/release-first-capture\" ]; do /bin/sleep 0.02; done; fi\nshift 13\n\"$@\" >/dev/null || exit 51\n/bin/cp \"$base/profile-template.json\" \"$out\" || exit 52\n/bin/cp \"$base/profile-template.syms.json\" \"${out%.json}.syms.json\" || exit 53;;\n*) exit 54;;\nesac\n").expect("controlled admitted collector");
        std::fs::set_permissions(&collector, std::fs::Permissions::from_mode(0o700))
            .expect("collector executable");
        let mut preset = CargoPreset::ephemeral_default(crate::CargoSubcommand::Run);
        preset.profile = selected_profile.map(str::to_owned);
        preset.presentation.save = task::SaveStrategy::None;
        preset.presentation.allow_concurrent_runs = true;
        preset.presentation.use_new_terminal = true;
        preset.native_profiler = Some(crate::CargoNativeProfiler {
            collector_path: collector.to_str().expect("collector").into(),
            profile_path: "target/capture.json".into(),
        });
        preset.trailing_args = vec![
            "space value".into(),
            "$literal".into(),
            "quote'\"".into(),
            "; no shell".into(),
        ];
        preset.environment.insert(
            "_NT_SYMBOL_PATH".into(),
            "https://must-not-reach-controlled-capture.invalid/symbols".into(),
        );
        preset.environment.insert(
            "PATH".into(),
            std::env::var("PATH").expect("actual installed Cargo/compiler PATH"),
        );
        preset.environment.insert(
            "NATIVE_WORKFLOW_CONTROLS".into(),
            installed
                .to_str()
                .expect("bounded owned control directory")
                .into(),
        );
        preset
            .environment
            .insert("CARGO_NET_OFFLINE".into(), "true".into());
        preset
            .environment
            .insert("RUSTUP_AUTO_INSTALL".into(), "0".into());
        preset.id = "native-workflow".into();
        let content = settings::CargoPresetSettingsContent {
            subcommand: Some("run".into()),
            profile: preset.profile.clone(),
            scope: Some("workspace".into()),
            save: Some("none".into()),
            allow_concurrent_runs: Some(true),
            use_new_terminal: Some(true),
            native_profiler: preset
                .native_profiler
                .as_ref()
                .map(crate::CargoNativeProfiler::settings_content),
            trailing_args: Some(preset.trailing_args.clone()),
            environment: Some(preset.environment.clone()),
            ..Default::default()
        };
        let user_settings = serde_json::json!({"cargo":{"schema_version":crate::CARGO_PRESET_SCHEMA_VERSION,"presets":{"native-workflow":content}}}).to_string();
        cx.update(|cx| {
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings(&user_settings, cx)
                    .expect("explicit reachable native preset")
            });
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
        });
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let project = Project::test_with_worktree_trust(filesystem, [root.as_path()], cx).await;
        let (worktree_store, worktree_id, store, profile_store) =
            project.read_with(cx, |project, cx| {
                (
                    project.worktree_store(),
                    project
                        .visible_worktrees(cx)
                        .next()
                        .expect("worktree")
                        .read(cx)
                        .id(),
                    project.cargo_workspace_store().clone(),
                    project
                        .profile_results_store_for_test()
                        .expect("profile results owner")
                        .clone(),
                )
            });
        let owner_trace_phase = Rc::new(std::cell::Cell::new("setup"));
        let owner_trace_count = Rc::new(std::cell::Cell::new(0_u32));
        let _owner_trace_subscriptions = cx.update(|cx| {
            let phase = owner_trace_phase.clone();
            let count = owner_trace_count.clone();
            let cargo = cx.subscribe(&store, move |_, _, _| {
                if count.get() < 32 {
                    count.set(count.get() + 1);
                    eprintln!(
                        "native fixture owner event phase={} kind=cargo-invalidated",
                        phase.get()
                    );
                }
            });
            let phase = owner_trace_phase.clone();
            let count = owner_trace_count.clone();
            let worktree = cx.subscribe(&worktree_store, move |_, event, _| {
                let kind = match event {
                    project::worktree_store::WorktreeStoreEvent::WorktreeUpdatedEntries(_, entries) => {
                        if entries
                            .iter()
                            .any(|(path, _, _)| path.file_name() == Some("Cargo.lock"))
                        {
                            "cargo-lock-index"
                        } else if entries
                            .iter()
                            .any(|(path, _, _)| path.file_name() == Some("Cargo.toml"))
                        {
                            "cargo-manifest-index"
                        } else {
                            "other-worktree-index"
                        }
                    }
                    project::worktree_store::WorktreeStoreEvent::WorktreeUpdatedGitRepositories(_, _)
                    | project::worktree_store::WorktreeStoreEvent::WorktreeUpdatedRootRepoCommonDir(_)
                    | project::worktree_store::WorktreeStoreEvent::WorktreeUpdateSent(_) => "worktree-non-input",
                    _ => "worktree-lifecycle",
                };
                if count.get() < 32 {
                    count.set(count.get() + 1);
                    eprintln!(
                        "native fixture owner event phase={} kind={kind}",
                        phase.get()
                    );
                }
            });
            vec![cargo, worktree]
        });
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        cx.run_until_parked();
        project.read_with(cx, |project, cx| {
            let path = project
                .find_project_path(root.join("target"), cx)
                .expect("output parent inside actual worktree");
            let entry = project
                .entry_for_path(&path, cx)
                .expect("actual indexed output parent");
            assert!(entry.is_dir() && entry.is_ignored && !entry.is_private && !entry.is_external);
        });
        let manifest = ProjectPath {
            worktree_id,
            path: Arc::from(RelPath::from_unix_str("Cargo.toml").expect("manifest")),
        };
        let refreshed = store
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("actual installed offline Cargo metadata through the existing host owner");
        assert!(
            refreshed.failures.is_empty(),
            "real metadata must not hide a tool failure"
        );
        assert!(
            refreshed
                .workspaces
                .iter()
                .any(|workspace| workspace.root_manifest.as_ref() == Some(&manifest)),
            "real metadata contains the exact selected workspace"
        );
        profile_store.update(cx, |store, _| {
            store.set_local_task_helper(
                std::env::current_exe().expect("source matched production-ABI helper"),
            )
        });
        let runtime_image = project
            .update(cx, |project, cx| {
                project.prepare_local_native_cargo_runtime_helper_image(cx)
            })
            .expect("genuine local bundled runtime-image issuer")
            .await
            .expect("actual SDK/signature/architecture/dependency-qualified dedicated sibling");
        assert_ne!(
            runtime_image.helper_path(),
            std::env::current_exe().expect("actual App process")
        );
        drop(runtime_image);
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let terminal_panel = terminal_view::terminal_panel::TerminalPanel::load(
            workspace.downgrade(),
            cx.update(|window, cx| window.to_async(cx)),
        )
        .await
        .expect("actual TerminalPanel and provider");
        let panel = CargoPanel::load(
            workspace.downgrade(),
            cx.update(|window, cx| window.to_async(cx)),
        )
        .await
        .expect("selected Cargo panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(terminal_panel, window, cx);
            workspace.add_panel(panel.clone(), window, cx);
        });
        let snapshot = store.read_with(cx, |store, cx| {
            let (snapshot, current) = store.cached_snapshot(cx).expect("current metadata");
            assert!(current);
            snapshot.clone()
        });
        panel.update(cx, |panel, cx| {
            panel.accept_snapshot(panel.host.refresh_generation(), snapshot, cx)
        });
        cx.run_until_parked();
        let selected = panel.read_with(cx, |panel, _| {
            panel
                .provider
                .action_selections
                .iter()
                .find(|(_, selection)| {
                    matches!(
                        selection.node_kind,
                        CargoActionNodeKind::Target(CargoActionTargetKind::Example) if example
                    ) || matches!(
                        selection.node_kind,
                        CargoActionNodeKind::Target(CargoActionTargetKind::Binary) if !example
                    )
                })
                .map(|(id, _)| id.clone())
                .expect("actual selected Binary or Example")
        });
        panel.update(cx, |panel, cx| {
            panel.set_active_preset(
                Some("native-workflow".into()),
                CargoSafeSelectionState::default(),
                cx,
            );
            panel.update_selection(
                |host| {
                    host.expand_all();
                    host.select(selected);
                },
                cx,
            );
        });
        let snapshot = store.read_with(cx, |store, cx| {
            let (snapshot, current) = store.cached_snapshot(cx).expect("current preset metadata");
            assert!(current);
            snapshot.clone()
        });
        panel.update(cx, |panel, cx| {
            panel.accept_snapshot(panel.host.refresh_generation(), snapshot, cx)
        });
        cx.run_until_parked();
        let (discovery, filesystem, worktree) = project.read_with(cx, |project, cx| {
            (
                project.rust_test_provider_store().clone(),
                project.fs().clone(),
                project
                    .worktrees(cx)
                    .find(|tree| tree.read(cx).id() == worktree_id)
                    .expect("original selected Worktree"),
            )
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace
                .focus_panel::<CargoPanel>(window, cx)
                .expect("initial actual Cargo panel activation");
        });
        {
            let deadline = cx.background_executor.timer(Duration::from_secs(10));
            let settling = async {
                while !discovery.read_with(cx, |store, _| store.discovery_idle_for_test()) {
                    cx.background_executor
                        .timer(Duration::from_millis(10))
                        .await;
                }
                let lock_path: Arc<RelPath> =
                    Arc::from(RelPath::from_unix_str("Cargo.lock").expect("literal Cargo input"));
                let metadata = filesystem
                    .metadata(&root.join("Cargo.lock"))
                    .await
                    .expect("actual discovery lock metadata")
                    .expect("original Cargo.lock remains present");
                let entry = worktree
                    .update(cx, |worktree, cx| {
                        worktree
                            .as_local()
                            .expect("original local Worktree")
                            .refresh_entry(lock_path.clone(), None, cx)
                    })
                    .await
                    .expect("actual final Cargo.lock publication")
                    .expect("actual Cargo.lock indexed");
                assert_eq!(
                    (entry.inode, entry.mtime, entry.size),
                    (metadata.inode, Some(metadata.mtime), metadata.len),
                    "Original Worktree indexes the actual discovery lock before selected admission"
                );
                cx.run_until_parked();
                loop {
                    let discovery_idle =
                        discovery.read_with(cx, |store, _| store.discovery_idle_for_test());
                    let scan_idle = worktree
                        .read_with(cx, |tree, _| tree.scan_id() == tree.completed_scan_id());
                    let metadata_current = store.read_with(cx, |store, cx| {
                        store
                            .cached_snapshot(cx)
                            .is_some_and(|(_, current)| current)
                    });
                    let panel_ready = panel.read_with(cx, |panel, _| {
                        !panel.refresh_pending
                            && panel.projection_task.is_ready()
                            && panel.execution_host_available
                    });
                    if discovery_idle && scan_idle && metadata_current && panel_ready {
                        break;
                    }
                    cx.background_executor
                        .timer(Duration::from_millis(10))
                        .await;
                }
                let metadata = filesystem
                    .metadata(&root.join("Cargo.lock"))
                    .await
                    .expect("retired discovery lock metadata")
                    .expect("original Cargo.lock remains present");
                worktree.read_with(cx, |tree, _| {
                    let indexed = tree
                        .entry_for_path(&lock_path)
                        .expect("published lock remains indexed");
                    assert_eq!(
                        (indexed.inode, indexed.mtime, indexed.size),
                        (metadata.inode, Some(metadata.mtime), metadata.len),
                        "Retired discovery and original Worktree agree on final Cargo.lock identity"
                    );
                    assert_eq!(tree.scan_id(), tree.completed_scan_id());
                });
            };
            futures::pin_mut!(settling);
            match futures::future::select(settling, deadline).await {
                futures::future::Either::Left(((), _)) => {}
                futures::future::Either::Right(_) => panic!(
                    "Original discovery, Worktree scan and active Cargo panel did not settle before native admission"
                ),
            }
        }
        owner_trace_phase.set("selected-action");
        owner_trace_count.set(0);
        assert!(
            panel.read_with(cx, |panel, _| panel.execution_host_available),
            "active preset configuration must finish real projection before selected invocation"
        );
        let offers = Rc::new(RefCell::new(Vec::new()));
        let _subscription = cx.update(|_, cx| {
            let offers = offers.clone();
            cx.subscribe(&workspace, move |_, event, _| {
                if let workspace::Event::TaskArtifactGroupReady(group) = event {
                    offers.borrow_mut().push(group.clone());
                }
            })
        });
        let mut build_id = None;
        let mut discarded_builds = 0;
        let mut discarded_captures = 0;
        let mut cancelled_capture_path = None;
        let mut superseded_terminal = None;
        let mut obsolete_capture_path = None;
        let mut already_prepared_capture = false;
        let mut already_approved_capture = false;
        if scenario != "rerun" {
            if scenario == "failed-build" {
                std::fs::write(installed.join("fail-build"), b"fail")
                    .expect("actual failed Build control");
            }
            if matches!(scenario, "cancelled-build" | "superseded-build") {
                std::fs::write(installed.join("hold-first-build"), b"hold")
                    .expect("actual held Build control");
            }
            if scenario == "cancelled-capture" {
                std::fs::write(installed.join("hold-first-capture"), b"hold")
                    .expect("actual active collector control");
            }
            workspace.update_in(cx, |workspace, window, cx| {
                workspace
                    .focus_panel::<CargoPanel>(window, cx)
                    .expect("mounted Cargo panel focus")
            });
            cx.dispatch_action(ProfileSelected);
            for _ in 0..200 {
                if installed.join("build.log").exists() {
                    break;
                }
                cx.background_executor
                    .timer(Duration::from_millis(50))
                    .await;
            }
            assert!(
                installed.join("build.log").exists(),
                "actual selected Build child starts: {scenario}"
            );
            let (_, original) = project.read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("history")
                    .read(cx)
                    .last_scheduled_task(None)
                    .expect("actual selected Build")
            });
            build_id = Some(original.id.clone());
            let terminal = project.read_with(cx, |project, cx| {
                project
                    .local_terminal_handles()
                    .iter()
                    .filter_map(|terminal| terminal.upgrade())
                    .find(|terminal| {
                        terminal
                            .read(cx)
                            .task()
                            .is_some_and(|state| state.spawned_task.id == original.id)
                    })
                    .expect("actual held Build terminal")
            });
            if scenario == "cancelled-build" {
                let handle = terminal.read_with(cx, |terminal, _| {
                    terminal
                        .task()
                        .expect("Build state")
                        .structured_task
                        .clone()
                        .expect("real cancellable Build owner")
                });
                cx.update(|_, cx| assert!(handle.cancel(cx)));
                assert!(matches!(
                    handle.state(),
                    task::StructuredTaskState::Cancelled { .. }
                ));
            }
            if scenario == "superseded-build" {
                superseded_terminal = Some(terminal);
            } else if matches!(scenario, "failed-build" | "cancelled-build") {
                let outcome = futures::future::select(
                    Box::pin(
                        terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx)),
                    ),
                    Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
                )
                .await;
                match outcome {
                    futures::future::Either::Left((status, _)) => assert!(
                        !status.is_some_and(|status| status.success()),
                        "failed or cancelled real child cannot become successful workload authority"
                    ),
                    futures::future::Either::Right(_) => {
                        panic!("actual negative Build did not exit")
                    }
                }
                cx.run_until_parked();
                assert!(!cx.has_pending_prompt());
                assert!(!installed.join("capture.log").exists());
                assert!(offers.borrow().is_empty());
                if scenario == "failed-build" {
                    std::fs::remove_file(installed.join("fail-build"))
                        .expect("fresh Build recovery");
                }
            } else {
                for _ in 0..200 {
                    if cx.has_pending_prompt() {
                        break;
                    }
                    cx.background_executor
                        .timer(Duration::from_millis(50))
                        .await;
                }
                assert!(
                    cx.has_pending_prompt(),
                    "genuine workload/probe reaches held capture consent"
                );
                assert!(!installed.join("capture.log").exists());
                if scenario == "superseded-capture" {
                    let (_, old_detail) = cx.pending_prompt().expect("old real Capture review");
                    obsolete_capture_path = Some(std::path::PathBuf::from(
                        old_detail
                            .lines()
                            .next()
                            .and_then(|line| line.strip_prefix("Profile: "))
                            .expect("old exact disclosed profile path"),
                    ));
                    let (source, cached) = project.read_with(cx, |project, cx| {
                        project
                            .task_store()
                            .read(cx)
                            .task_inventory()
                            .expect("history")
                            .read(cx)
                            .last_scheduled_task(None)
                            .expect("same declarative Build for fresh invocation")
                    });
                    assert_eq!(cached.id, original.id);
                    let (sender, receiver) = futures::channel::oneshot::channel();
                    let newer_build = workspace.update_in(cx, |workspace, window, cx| {
                        workspace.schedule_resolved_task_with_structured_handle_and_completion(
                            source,
                            cached,
                            false,
                            move |result, _| {
                                sender
                                    .send(result)
                                    .expect("newer real Build completion observer");
                            },
                            window,
                            cx,
                        )
                    });
                    let completed = futures::future::select(
                        Box::pin(receiver),
                        Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
                    )
                    .await;
                    match completed {
                        futures::future::Either::Left((result, _)) => assert_eq!(
                            result.expect("newer Build callback"),
                            workspace::tasks::ScheduledTaskResult::Success,
                            "actual Build observer finishes the genuine Build-to-Capture transition"
                        ),
                        futures::future::Either::Right(_) => {
                            panic!("newer actual Build did not submit its Capture")
                        }
                    }
                    assert!(matches!(
                        newer_build.state(),
                        task::StructuredTaskState::Completed { success: true, .. }
                    ));
                    assert_eq!(
                        cx.pending_prompt().expect("old FIFO prompt remains held").1,
                        old_detail
                    );
                    assert!(!installed.join("capture.log").exists());
                    assert!(offers.borrow().is_empty());
                    cx.run_until_parked();
                    cx.simulate_prompt_answer("Capture");
                    let (_, new_detail) = cx
                        .pending_prompt()
                        .expect("newer actual Capture prompt is queued behind the old approval");
                    assert_ne!(new_detail, old_detail, "new Capture has fresh UUID paths");
                    assert!(
                        !installed.join("capture.log").exists(),
                        "late old approval cannot launch its collector"
                    );
                    assert!(offers.borrow().is_empty());
                    cx.simulate_prompt_answer("Capture");
                    already_prepared_capture = true;
                    already_approved_capture = true;
                } else if scenario == "profile-manifest" {
                    let profile = selected_profile
                        .expect("the held scenario selects a declared custom profile");
                    let (_, original_detail) = cx
                        .pending_prompt()
                        .expect("original completed Build awaits Capture consent");
                    obsolete_capture_path = Some(std::path::PathBuf::from(
                        original_detail
                            .lines()
                            .next()
                            .and_then(|line| line.strip_prefix("Profile: "))
                            .expect("old consent discloses its exact profile path"),
                    ));
                    let manifest_file = root.join("Cargo.toml");
                    let original_manifest =
                        std::fs::read(&manifest_file).expect("original declared profile manifest");
                    let removed_manifest =
                        b"[workspace]\nmembers=['member-one','member-two']\nresolver='2'\n"
                            .to_vec();
                    assert_ne!(original_manifest, removed_manifest);
                    let invalidated = Rc::new(std::cell::Cell::new(false));
                    let _profile_subscription = cx.update(|_, cx| {
                        let invalidated = invalidated.clone();
                        cx.subscribe(&store, move |_, event, _| {
                            if matches!(event, project::cargo_workspace_store::CargoWorkspaceStoreEvent::Invalidated) {
                                invalidated.set(true);
                            }
                        })
                    });
                    for (contents, profile_present) in
                        [(removed_manifest, false), (original_manifest, true)]
                    {
                        invalidated.set(false);
                        std::fs::write(&manifest_file, &contents)
                            .expect("real profile manifest mutation");
                        let metadata = filesystem
                            .metadata(&manifest_file)
                            .await
                            .expect("changed manifest metadata")
                            .expect("changed manifest remains ordinary source");
                        {
                            let deadline = cx.background_executor.timer(Duration::from_secs(10));
                            let settling = async {
                                loop {
                                    let indexed = worktree.read_with(cx, |tree, _| {
                                        tree.entry_for_path(&manifest.path).is_some_and(|entry| {
                                            (entry.inode, entry.mtime, entry.size)
                                                == (
                                                    metadata.inode,
                                                    Some(metadata.mtime),
                                                    metadata.len,
                                                )
                                        }) && tree.scan_id() == tree.completed_scan_id()
                                    });
                                    if indexed && invalidated.get() {
                                        break;
                                    }
                                    cx.background_executor
                                        .timer(Duration::from_millis(10))
                                        .await;
                                }
                                panel.update(cx, |panel, cx| panel.refresh(cx));
                                loop {
                                    let metadata_current = store.read_with(cx, |store, cx| {
                                        store.cached_snapshot(cx).is_some_and(
                                            |(snapshot, current)| {
                                                current
                                                    && snapshot.failures.is_empty()
                                                    && snapshot
                                                        .workspaces
                                                        .iter()
                                                        .find(|workspace| {
                                                            workspace.root_manifest.as_ref()
                                                                == Some(&manifest)
                                                        })
                                                        .is_some_and(|workspace| {
                                                            workspace
                                                                .configuration
                                                                .profiles
                                                                .iter()
                                                                .filter(|candidate| {
                                                                    candidate.name == profile
                                                                })
                                                                .count()
                                                                == usize::from(profile_present)
                                                        })
                                            },
                                        )
                                    });
                                    let discovery_idle = discovery
                                        .read_with(cx, |store, _| store.discovery_idle_for_test());
                                    let scan_idle = worktree.read_with(cx, |tree, _| {
                                        tree.scan_id() == tree.completed_scan_id()
                                    });
                                    let panel_ready = panel.read_with(cx, |panel, _| {
                                        !panel.refresh_pending && panel.projection_task.is_ready()
                                    });
                                    if metadata_current
                                        && discovery_idle
                                        && scan_idle
                                        && panel_ready
                                    {
                                        break;
                                    }
                                    cx.background_executor
                                        .timer(Duration::from_millis(10))
                                        .await;
                                }
                            };
                            futures::pin_mut!(settling);
                            match futures::future::select(settling, deadline).await {
                                futures::future::Either::Left(((), _)) => {}
                                futures::future::Either::Right(_) => panic!(
                                    "Actual profile manifest watcher, invalidation and production refresh failed to settle; profile_present={profile_present}"
                                ),
                            }
                        }
                        assert!(
                            invalidated.get(),
                            "real Cargo input mutation reaches the existing store invalidation owner"
                        );
                        assert_eq!(
                            std::fs::read(&manifest_file).expect("current real manifest bytes"),
                            contents
                        );
                        if !profile_present {
                            if let Some((_, detail)) = cx.pending_prompt() {
                                assert_eq!(
                                    detail, original_detail,
                                    "only the old consent may remain pending"
                                );
                                cx.simulate_prompt_answer("Capture");
                            }
                            cx.run_until_parked();
                            assert!(
                                !cx.has_pending_prompt(),
                                "the expired selected profile cannot retain a live Capture consent"
                            );
                            assert!(
                                !installed.join("capture.log").exists(),
                                "a removed named profile never launches its old collector"
                            );
                            assert!(
                                offers.borrow().is_empty(),
                                "manifest invalidation cannot publish an old profile pair"
                            );
                            let old_profile =
                                obsolete_capture_path.as_ref().expect("old disclosed pair");
                            assert!(!old_profile.exists());
                            assert!(!old_profile.with_extension("syms.json").exists());
                            assert_eq!(
                                std::fs::read_to_string(installed.join("build.log"))
                                    .expect("original completed Build receipt")
                                    .lines()
                                    .count(),
                                1
                            );
                        }
                    }
                    assert!(
                        panel.read_with(cx, |panel, _| panel.execution_host_available),
                        "the restored profile is admitted only after actual metadata and panel recovery"
                    );
                } else if scenario == "declined-capture" {
                    cx.simulate_prompt_answer("Cancel");
                } else if scenario == "cancelled-capture" {
                    cx.simulate_prompt_answer("Capture");
                    for _ in 0..200 {
                        if installed.join("held-capture.pid").exists() {
                            break;
                        }
                        cx.background_executor
                            .timer(Duration::from_millis(50))
                            .await;
                    }
                    let collector_process_id = std::fs::read_to_string(
                        installed.join("held-capture.pid"),
                    )
                    .expect(
                        "collector really executes after the native registration acknowledgement",
                    )
                    .trim()
                    .parse::<i32>()
                    .expect("actual held collector process identity");
                    assert!(collector_process_id > 0);
                    let captures = std::fs::read_to_string(installed.join("capture.log"))
                        .expect("actual registered collector invocation");
                    assert_eq!(captures.lines().count(), 1);
                    let profile = captures.lines().next().expect("held profile path");
                    let profile_path = root.join(profile);
                    assert!(!profile_path.exists());
                    assert!(!profile_path.with_extension("syms.json").exists());
                    assert!(offers.borrow().is_empty());
                    let capture_id = task::TaskId(format!("{}:native-capture", original.id.0));
                    let capture_terminal = project.read_with(cx, |project, cx| {
                        project
                            .local_terminal_handles()
                            .iter()
                            .filter_map(|terminal| terminal.upgrade())
                            .find(|terminal| {
                                terminal.read(cx).task().is_some_and(|state| {
                                    state.spawned_task.id == capture_id
                                        && state
                                            .spawned_task
                                            .args
                                            .iter()
                                            .any(|argument| argument.contains(profile))
                                })
                            })
                            .expect("actual held native Capture terminal")
                    });
                    let (handle, raw_completion) =
                        capture_terminal.read_with(cx, |terminal, cx| {
                            let state = terminal.task().expect("held Capture state");
                            let handle = state
                                .structured_task
                                .clone()
                                .expect("actual Capture cancellation owner");
                            assert!(matches!(
                                handle.state(),
                                task::StructuredTaskState::Running { .. }
                            ));
                            assert!(terminal.blocks_application_restart());
                            (handle, terminal.wait_for_completed_task(cx))
                        });
                    let alive = cx
                        .background_spawn(async move {
                            util::command::new_command("/bin/kill")
                                .args(["-0", &collector_process_id.to_string()])
                                .status()
                                .await
                        })
                        .await
                        .expect("actual collector liveness check");
                    assert!(
                        alive.success(),
                        "collector is alive after ACK, before cancellation"
                    );
                    cx.update(|_, cx| assert!(handle.cancel(cx)));
                    assert!(matches!(
                        handle.state(),
                        task::StructuredTaskState::Cancelled { .. }
                    ));
                    let completion = futures::future::select(
                        Box::pin(raw_completion),
                        Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
                    )
                    .await;
                    let status = match completion {
                        futures::future::Either::Left((status, _)) => {
                            status.expect("real cancelled Capture kernel status")
                        }
                        futures::future::Either::Right(_) => {
                            panic!("actual active Capture did not terminate")
                        }
                    };
                    assert!(
                        !status.success(),
                        "cancelling a running collector cannot manufacture Success"
                    );
                    let alive = cx
                        .background_spawn(async move {
                            util::command::new_command("/bin/kill")
                                .args(["-0", &collector_process_id.to_string()])
                                .status()
                                .await
                        })
                        .await
                        .expect("actual cancelled collector liveness check");
                    assert!(
                        !alive.success(),
                        "cancelled collector has actually disappeared"
                    );
                    cx.run_until_parked();
                    assert!(matches!(
                        handle.state(),
                        task::StructuredTaskState::Cancelled {
                            termination_confirmed: true,
                            ..
                        }
                    ));
                    assert!(
                        !capture_terminal
                            .read_with(cx, |terminal, _| terminal.blocks_application_restart())
                    );
                    assert!(!cx.has_pending_prompt());
                    assert!(!profile_path.exists());
                    assert!(!profile_path.with_extension("syms.json").exists());
                    assert!(offers.borrow().is_empty());
                    eprintln!(
                        "native-active-capture-cancellation: collector_alive_before_cancel=true raw_status={status} termination_confirmed=true pair_missing=true offers=0"
                    );
                    cancelled_capture_path = Some(profile_path);
                    discarded_captures = 1;
                    std::fs::remove_file(installed.join("hold-first-capture"))
                        .expect("fresh actual collector recovery");
                } else {
                    let source = project
                        .update(cx, |project, cx| {
                            let path = project
                                .find_project_path(root.join("member-one/src/main.rs"), cx)
                                .expect("selected source");
                            project.open_buffer(path, cx)
                        })
                        .await
                        .expect("actual source buffer");
                    let editor = workspace.update_in(cx, |workspace, window, cx| {
                        let editor = cx.new(|cx| {
                            editor::Editor::for_buffer(
                                source.clone(),
                                Some(project.clone()),
                                window,
                                cx,
                            )
                        });
                        workspace.add_item_to_active_pane(
                            Box::new(editor.clone()),
                            None,
                            true,
                            window,
                            cx,
                        );
                        editor
                    });
                    editor.update(cx, |editor, cx| {
                        editor.edit(
                            [(
                                editor::MultiBufferOffset(0)..editor::MultiBufferOffset(0),
                                "// changed during actual consent\n",
                            )],
                            cx,
                        )
                    });
                    cx.simulate_prompt_answer("Capture");
                    cx.run_until_parked();
                    assert!(
                        !installed.join("capture.log").exists(),
                        "dirty selected source denies the collector before child execution"
                    );
                    assert!(offers.borrow().is_empty());
                    project
                        .update(cx, |project, cx| project.save_buffer(source, cx))
                        .await
                        .expect("actual fresh-save recovery");
                    {
                        let deadline = cx.background_executor.timer(Duration::from_secs(10));
                        let settling = async {
                            let source_path: Arc<RelPath> = Arc::from(
                                RelPath::from_unix_str("member-one/src/main.rs")
                                    .expect("original selected source path"),
                            );
                            let source_file = root.join("member-one/src/main.rs");
                            let metadata = filesystem
                                .metadata(&source_file)
                                .await
                                .expect("actual saved-source metadata")
                                .expect("original saved source remains present");
                            let entry = worktree
                                .update(cx, |worktree, cx| {
                                    worktree
                                        .as_local()
                                        .expect("original local Worktree")
                                        .refresh_entry(source_path.clone(), None, cx)
                                })
                                .await
                                .expect("actual saved-source publication")
                                .expect("actual saved source indexed");
                            assert_eq!(
                                (entry.inode, entry.mtime, entry.size),
                                (metadata.inode, Some(metadata.mtime), metadata.len),
                                "Original Worktree indexes the saved source before recovery admission"
                            );
                            cx.run_until_parked();
                            loop {
                                let discovery_idle = discovery
                                    .read_with(cx, |store, _| store.discovery_idle_for_test());
                                let scan_idle = worktree.read_with(cx, |tree, _| {
                                    tree.scan_id() == tree.completed_scan_id()
                                });
                                let metadata_current = store.read_with(cx, |store, cx| {
                                    store
                                        .cached_snapshot(cx)
                                        .is_some_and(|(_, current)| current)
                                });
                                let panel_ready = panel.read_with(cx, |panel, _| {
                                    !panel.refresh_pending
                                        && panel.projection_task.is_ready()
                                        && panel.execution_host_available
                                });
                                if discovery_idle && scan_idle && metadata_current && panel_ready {
                                    break;
                                }
                                cx.background_executor
                                    .timer(Duration::from_millis(10))
                                    .await;
                            }
                            let metadata = filesystem
                                .metadata(&source_file)
                                .await
                                .expect("retired saved-source metadata")
                                .expect("original saved source remains present");
                            worktree.read_with(cx, |tree, _| {
                                let indexed = tree
                                    .entry_for_path(&source_path)
                                    .expect("saved source remains indexed");
                                assert_eq!(
                                    (indexed.inode, indexed.mtime, indexed.size),
                                    (metadata.inode, Some(metadata.mtime), metadata.len),
                                    "Retired discovery and original Worktree agree on saved-source identity"
                                );
                                assert_eq!(tree.scan_id(), tree.completed_scan_id());
                            });
                        };
                        futures::pin_mut!(settling);
                        match futures::future::select(settling, deadline).await {
                            futures::future::Either::Left(((), _)) => {}
                            futures::future::Either::Right(_) => panic!(
                                "Saved-source discovery, Worktree scan and active Cargo panel did not settle before recovery admission"
                            ),
                        }
                    }
                }
                if !already_approved_capture {
                    cx.run_until_parked();
                    if discarded_captures == 0 {
                        assert!(!installed.join("capture.log").exists());
                    } else {
                        assert_eq!(
                            std::fs::read_to_string(installed.join("capture.log"))
                                .expect("actual cancelled capture")
                                .lines()
                                .count(),
                            discarded_captures
                        );
                    }
                    assert!(offers.borrow().is_empty());
                }
            }
            discarded_builds = 1;
        }
        let mut pair_paths = HashSet::new();
        let mut successful_build_directories = HashSet::new();
        let successful_routes = match scenario {
            "rerun" => 3,
            "superseded-capture" => 2,
            _ => 1,
        };
        for route in 0..successful_routes {
            if route == 0 && scenario == "rerun" {
                workspace.update_in(cx, |workspace, window, cx| {
                    workspace
                        .focus_panel::<CargoPanel>(window, cx)
                        .expect("mounted Cargo panel focus")
                });
                cx.dispatch_action(ProfileSelected);
            } else if !(route == 0 && already_prepared_capture) {
                workspace.update_in(cx, |workspace, window, cx| {
                    window.focus(&workspace.focus_handle(cx), cx)
                });
                cx.update(|window, cx| {
                    assert!(
                        window.is_action_available(&tasks_ui::Rerun::default(), cx),
                        "registered Rerun is available in the rendered focused workspace"
                    );
                });
                cx.dispatch_action(tasks_ui::Rerun {
                    reevaluate_context: route == 2,
                    ..Default::default()
                });
            }
            if !(route == 0 && already_approved_capture) {
                for _ in 0..200 {
                    if cx.has_pending_prompt() {
                        break;
                    }
                    cx.background_executor
                        .timer(Duration::from_millis(50))
                        .await;
                }
                if !cx.has_pending_prompt() {
                    let markers: Vec<_> = ["build.log", "probe.log", "capture.log"]
                        .into_iter()
                        .map(|name| (name, installed.join(name).exists()))
                        .collect();
                    let terminals = project.read_with(cx, |project, cx| {
                        project
                            .local_terminal_handles()
                            .iter()
                            .filter_map(|handle| {
                                let terminal = handle.upgrade()?;
                                let terminal = terminal.read(cx);
                                Some((
                                    terminal.task().map(|state| format!("{:?}", state.status)),
                                    terminal
                                        .get_content()
                                        .chars()
                                        .take(4096)
                                        .collect::<String>(),
                                ))
                            })
                            .collect::<Vec<_>>()
                    });
                    eprintln!(
                        "native controlled route={route} markers={markers:?} terminals={terminals:?}"
                    );
                }
                assert!(
                    cx.has_pending_prompt(),
                    "actual successful Build and probe reach held consent: route {route}"
                );
            }
            let builds = std::fs::read_to_string(installed.join("build.log"))
                .expect("actual ACK4 Build execution");
            assert_eq!(builds.lines().count(), route + 1 + discarded_builds);
            let all_build_directories: HashSet<_> = builds.lines().collect();
            assert_eq!(
                all_build_directories.len(),
                builds.lines().count(),
                "every actual Build uses a distinct UUID output directory"
            );
            let current_build_directory = std::path::PathBuf::from(
                builds
                    .lines()
                    .last()
                    .expect("latest actual Build output directory"),
            );
            assert!(
                successful_build_directories.insert(current_build_directory.clone()),
                "each successful Capture consumes its own latest Build invocation"
            );
            assert_eq!(
                current_build_directory.parent(),
                Some(root.join("target").as_path())
            );
            let build_component = current_build_directory
                .file_name()
                .and_then(|component| component.to_str())
                .and_then(|component| component.strip_prefix("zed-native-profile-build-"))
                .expect("per-invocation Build directory");
            assert_eq!(build_component.len(), 32);
            assert!(build_component.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert!(
                installed.join("probe.log").exists(),
                "real collector probe completed"
            );
            let captures = match std::fs::read_to_string(installed.join("capture.log")) {
                Ok(contents) => contents,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
                Err(error) => panic!("capture log cannot be read: {error}"),
            };
            if !(route == 0 && already_approved_capture) {
                assert_eq!(
                    captures.lines().count(),
                    route + discarded_captures,
                    "no new capture child before this consent"
                );
            }
            assert_eq!(offers.borrow().len(), route);
            let (_, history) = project.read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("history")
                    .read(cx)
                    .last_scheduled_task(None)
                    .expect("declarative Build history")
            });
            if let Some(build_id) = &build_id {
                assert_eq!(&history.id, build_id);
            } else {
                build_id = Some(history.id.clone());
            }
            assert!(
                history
                    .original_task()
                    .stdout_report
                    .as_ref()
                    .expect("Build descriptor")
                    .native_profile
                    .is_some()
            );
            assert_eq!(
                history
                    .stdout_report()
                    .expect("genuine runtime descriptor")
                    .format,
                task::TaskStdoutReportFormat::CargoRuntimePreparationJson
            );
            assert!(
                history.resolved_artifact_group().is_none(),
                "Build is not relabeled Capture"
            );
            for private_key in [
                util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
                util::macos_task_completion::TASK_BUILD_INVOCATION_ENV,
                util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV,
                util::macos_task_completion::TASK_CARGO_RUNTIME_ENDPOINT_ENV,
                util::macos_task_completion::TASK_CARGO_RUNTIME_TOKEN_ENV,
            ] {
                assert!(!history.resolved.env.contains_key(private_key));
                assert!(!history.original_task().env.contains_key(private_key));
            }
            project.read_with(cx, |project, cx| {
                let handle = project
                    .local_terminal_handles()
                    .iter()
                    .filter_map(|terminal| terminal.upgrade())
                    .find_map(|terminal| {
                        let terminal = terminal.read(cx);
                        let task = terminal.task()?;
                        (task.spawned_task.id == history.id
                            && task.spawned_task.args.iter().any(|argument| {
                                argument.contains(
                                    current_build_directory
                                        .to_str()
                                        .expect("current Build path"),
                                )
                            }))
                        .then(|| {
                            task.structured_task
                                .clone()
                                .expect("actual Build cancellation owner")
                        })
                    })
                    .expect("actual mounted Build terminal");
                assert!(
                    matches!(
                        handle.state(),
                        task::StructuredTaskState::Completed { success: true, .. }
                    ),
                    "Build remains actually Completed during held Capture consent"
                );
            });
            if !(route == 0 && already_approved_capture) {
                cx.simulate_prompt_answer("Capture");
            }
            for _ in 0..200 {
                if offers.borrow().len() == route + 1 {
                    break;
                }
                cx.background_executor
                    .timer(Duration::from_millis(50))
                    .await;
            }
            assert_eq!(
                offers.borrow().len(),
                route + 1,
                "actual ACK3 pair completion is offered"
            );
            let group = offers.borrow().last().expect("actual pair offer").clone();
            assert!(group.host_verified());
            assert!(cx.read(|cx| group.is_current(&project, cx)));
            let profile = &group
                .group()
                .files
                .get("profile")
                .expect("profile role")
                .path;
            if let Some(cancelled_capture_path) = &cancelled_capture_path {
                assert_ne!(&root.join(profile), cancelled_capture_path);
                assert!(!cancelled_capture_path.exists());
                assert!(!cancelled_capture_path.with_extension("syms.json").exists());
            }
            if let Some(obsolete_capture_path) = &obsolete_capture_path {
                assert_ne!(&root.join(profile), obsolete_capture_path);
                assert!(!obsolete_capture_path.exists());
                assert!(!obsolete_capture_path.with_extension("syms.json").exists());
            }
            let symbols = &group
                .group()
                .files
                .get("symbols")
                .expect("symbols role")
                .path;
            assert_eq!(
                std::path::Path::new(profile).with_extension("syms.json"),
                std::path::Path::new(symbols)
            );
            assert!(
                pair_paths.insert(profile.clone()),
                "fresh pair on actual registered Rerun"
            );
            assert_eq!(
                profile.matches('-').count(),
                1,
                "one UUID suffix without accumulation"
            );
            assert_eq!(
                std::fs::read(root.join(profile)).expect("produced profile"),
                include_bytes!(
                    "../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.json"
                )
            );
            let profile_file_name = std::path::Path::new(profile)
                .file_name()
                .expect("profile filename")
                .to_str()
                .expect("UTF-8");
            let argv_file = installed.join(format!("{profile_file_name}.argv"));
            let received = std::fs::read(argv_file).expect("actual collector argv");
            let fields: Vec<_> = received
                .split(|byte| *byte == 0)
                .filter(|field| !field.is_empty())
                .collect();
            let expected_profile_path = root.join(profile);
            assert_eq!(
                fields.get(11).copied(),
                Some(
                    expected_profile_path
                        .to_str()
                        .expect("actual anchored profile path UTF-8")
                        .as_bytes()
                )
            );
            if let Some(profile) = selected_profile {
                let matches = history
                    .resolved
                    .args
                    .windows(2)
                    .filter(|pair| pair[0] == "--profile" && pair[1] == profile)
                    .count();
                assert_eq!(
                    matches, 1,
                    "cached and reevaluated Tasks preserve the exact selected profile"
                );
                assert!(
                    history
                        .original_task()
                        .args
                        .windows(2)
                        .any(|pair| pair == ["--profile", profile])
                );
            }
            let output_profile = match selected_profile {
                None | Some("dev") => "debug",
                Some(profile) => profile,
            };
            let expected_executable =
                current_build_directory
                    .join(output_profile)
                    .join(if example {
                        "examples/profile-example"
                    } else {
                        "member-one"
                    });
            let actual_workload_metadata = std::fs::symlink_metadata(&expected_executable)
                .expect("genuine Cargo-produced selected executable metadata");
            assert!(
                actual_workload_metadata.is_file()
                    && !actual_workload_metadata.file_type().is_symlink()
                    && (1..=256 * 1024 * 1024).contains(&actual_workload_metadata.len())
                    && actual_workload_metadata.permissions().mode() & 0o111 != 0,
                "actual selected Cargo output remains a bounded ordinary executable"
            );
            assert_eq!(
                fields.get(13).copied(),
                Some(
                    expected_executable
                        .to_str()
                        .expect("actual Build executable UTF-8")
                        .as_bytes()
                ),
                "collector receives the executable from this latest actual Build UUID directory"
            );
            assert_eq!(
                fields
                    .get(
                        fields
                            .len()
                            .checked_sub(preset.trailing_args.len())
                            .expect("bounded literal tail")..
                    )
                    .expect("literal workload tail"),
                preset
                    .trailing_args
                    .iter()
                    .map(|argument| argument.as_bytes())
                    .collect::<Vec<_>>()
            );
            let capture_id = task::TaskId(format!("{}:native-capture", history.id.0));
            let capture_terminal = project.read_with(cx, |project, cx| {
                project
                    .local_terminal_handles()
                    .iter()
                    .filter_map(|terminal| terminal.upgrade())
                    .find(|terminal| {
                        terminal.read(cx).task().is_some_and(|state| {
                            state.spawned_task.id == capture_id
                                && state
                                    .spawned_task
                                    .args
                                    .iter()
                                    .any(|argument| argument.contains(profile))
                        })
                    })
                    .expect("separate actual current-pair Capture terminal")
            });
            let actual_capture = capture_terminal
                .read_with(cx, |terminal, cx| {
                    let state = terminal.task().expect("Capture Task state");
                    assert_ne!(state.spawned_task.id, history.id);
                    assert!(
                        state
                            .spawned_task
                            .args
                            .iter()
                            .any(|argument| argument == "--checked-native-profile-capture")
                    );
                    let handle = state
                        .structured_task
                        .as_ref()
                        .expect("actual Capture cancellation owner");
                    assert_eq!(handle.task_id(), capture_id);
                    terminal.wait_for_completed_task(cx)
                })
                .await;
            assert!(
                actual_capture
                    .expect("actual Capture kernel status")
                    .success()
            );
            if obsolete_capture_path.is_some() {
                cx.run_until_parked();
                assert!(
                    cx.read(|cx| group.is_current(&project, cx)),
                    "old approval, completion and cleanup cannot retire the newer pair"
                );
                assert!(!cx.has_pending_prompt());
                assert_eq!(offers.borrow().len(), route + 1);
            }
            let (_, latest) = project.read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("history")
                    .read(cx)
                    .last_scheduled_task(None)
                    .expect("preserved Build")
            });
            assert_eq!(
                latest.id, history.id,
                "Capture omits history; Rerun renews Build"
            );
        }
        assert_eq!(pair_paths.len(), successful_routes);
        if discarded_captures > 0 {
            assert_eq!(
                std::fs::read_to_string(installed.join("capture.log"))
                    .expect("actual recovery collector receipt")
                    .lines()
                    .count(),
                successful_routes + discarded_captures
            );
            assert_eq!(offers.borrow().len(), successful_routes);
            assert!(
                offers
                    .borrow()
                    .last()
                    .expect("fresh recovery pair")
                    .host_verified()
            );
            eprintln!(
                "native-active-capture-cancellation: fresh_registered_rerun_recovery=true cancelled_pair_reused=false offers={successful_routes}"
            );
        }
        if let Some(terminal) = superseded_terminal {
            std::fs::write(installed.join("release-first-build"), b"release")
                .expect("release obsolete actual producer if still alive");
            let outcome = futures::future::select(
                Box::pin(
                    terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx)),
                ),
                Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
            )
            .await;
            match outcome {
                futures::future::Either::Left(_) => {}
                futures::future::Either::Right(_) => panic!("obsolete actual Build did not retire"),
            }
            cx.run_until_parked();
            assert_eq!(
                offers.borrow().len(),
                successful_routes,
                "obsolete Build completion cannot offer an extra Capture"
            );
            assert_eq!(
                std::fs::read_to_string(installed.join("capture.log"))
                    .expect("fresh Capture")
                    .lines()
                    .count(),
                successful_routes
            );
            assert!(!cx.has_pending_prompt());
        }
        cx.run_until_parked();
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[test]
    fn cargo_native_profile_active_mode_has_visible_configuration_and_action_name() {
        let mut preset = CargoPreset::ephemeral_default(crate::CargoSubcommand::Run);
        preset.id = "native".into();
        preset.native_profiler = Some(
            crate::CargoNativeProfiler::from_settings(
                &settings::CargoNativeProfilerSettingsContent {
                    collector_path: Some("/opt/tools/samply".into()),
                    profile_path: Some("target/capture.json".into()),
                },
            )
            .expect("explicit native declaration"),
        );
        let state = CargoPresetWorkspaceState {
            active_preset_id: Some(preset.id.clone()),
            ..Default::default()
        };
        let settings = CargoPresetSettings {
            presets: collections::BTreeMap::from_iter([(preset.id.clone(), preset)]),
            diagnostics: Vec::new(),
        };
        let active = active_configuration_from_preset_state(&state, &settings, None);
        assert!(active.native_profiling);
        let (snapshot, _) = CargoTreeProvider::project_with_configuration(&snapshot(), &active);
        let mode =
            find_node(&snapshot.roots, "Profiling mode").expect("visible native configuration");
        assert_eq!(mode.secondary_label.as_deref(), Some("macOS native"));
        assert!(mode.accessibility_label.contains("macOS native"));
        assert_eq!(
            settings
                .presets
                .get("native")
                .expect("current preset")
                .profile_action_label(),
            "Native Profile (macOS)"
        );
        assert_eq!(
            CargoPreset::ephemeral_default(crate::CargoSubcommand::Run).profile_action_label(),
            "Profile"
        );
    }
}
