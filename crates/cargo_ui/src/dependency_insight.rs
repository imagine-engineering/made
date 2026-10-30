use language_tools::language_tool_tree::{LanguageToolNode, LanguageToolNodeId};
use project::{
    ProjectPath,
    cargo_workspace::{
        CargoDependencyDeclarationOrigin, CargoDependencyFeatureCausality,
        CargoDependencyLockStatus, CargoDependencyModel, CargoDependencySourceKind,
    },
};

pub struct CargoDependencyInsightProjection {
    pub children: Vec<LanguageToolNode>,
    pub navigation: Vec<(LanguageToolNodeId, ProjectPath)>,
}

pub fn cargo_dependency_insight(
    parent: &LanguageToolNodeId,
    dependency: &CargoDependencyModel,
) -> CargoDependencyInsightProjection {
    let mut navigation = Vec::new();
    let declared = vec![
        fact(
            parent,
            "declared-origin",
            "Origin",
            match dependency.declaration_origin {
                CargoDependencyDeclarationOrigin::Direct => "direct declaration",
                CargoDependencyDeclarationOrigin::WorkspaceInherited => "workspace inherited",
                CargoDependencyDeclarationOrigin::Unknown => "unknown declaration origin",
            },
        ),
        fact(
            parent,
            "declared-version",
            "Version requirement",
            &dependency.version_requirement,
        ),
        fact(
            parent,
            "declared-target",
            "Target",
            dependency.target.as_deref().unwrap_or("all targets"),
        ),
    ];
    if let Some(path) = dependency.declaration_manifest.clone() {
        let id = id(parent, "declared-manifest");
        navigation.push((id.clone(), path.clone()));
        let mut manifest = fact_node(
            id,
            "Manifest",
            path.path.as_unix_str(),
            "Cargo dependency declaration manifest",
        );
        manifest.enabled = true;
        manifest.activation_label = Some("Open declaration manifest".to_string());
        let mut declared = declared;
        declared.push(manifest);
        return projection(parent, dependency, declared, navigation);
    }
    projection(parent, dependency, declared, navigation)
}

fn projection(
    parent: &LanguageToolNodeId,
    dependency: &CargoDependencyModel,
    declared: Vec<LanguageToolNode>,
    mut navigation: Vec<(LanguageToolNodeId, ProjectPath)>,
) -> CargoDependencyInsightProjection {
    let resolved = dependency
        .resolved_instances
        .iter()
        .enumerate()
        .map(|(index, resolved)| {
            let lock = lock_label(resolved.lock_status);
            let source = source_label(resolved.source_kind);
            let node_id = id(parent, &format!("resolved-{index}"));
            let local_path = resolved
                .workspace_member
                .clone()
                .or_else(|| resolved.local_manifest.clone());
            if let Some(path) = local_path.clone() {
                navigation.push((node_id.clone(), path));
            }
            LanguageToolNode {
                id: node_id,
                label: format!("{} {}", resolved.name, resolved.version),
                secondary_label: Some(format!("{source} · {lock}")),
                icon: None,
                accessibility_label: format!(
                    "Resolved Cargo dependency {} {}, {source}, {lock}",
                    resolved.name, resolved.version
                ),
                children: Vec::new(),
                enabled: local_path.is_some(),
                activation_label: local_path.map(|_| "Open resolved local manifest".to_string()),
            }
        })
        .collect::<Vec<_>>();
    let resolved = if resolved.is_empty() {
        vec![fact(
            parent,
            "resolved-unknown",
            "Resolution",
            "unavailable",
        )]
    } else {
        resolved
    };
    let requested_features = if dependency.requested_features.is_empty() {
        "none".to_string()
    } else {
        dependency.requested_features.join(", ")
    };
    let mut features = vec![fact(
        parent,
        "features-requested",
        "Requested",
        &requested_features,
    )];
    for (index, resolved) in dependency.resolved_instances.iter().enumerate() {
        let enabled = if resolved.enabled_features.is_empty() {
            "none".to_string()
        } else {
            resolved.enabled_features.join(", ")
        };
        features.push(fact(
            parent,
            &format!("features-resolved-{index}"),
            &format!("Enabled for {} {}", resolved.name, resolved.version),
            &enabled,
        ));
    }
    features.push(fact(
        parent,
        "features-causality",
        "Activation cause",
        match dependency.feature_causality {
            CargoDependencyFeatureCausality::Validated => "validated resolved facts",
            CargoDependencyFeatureCausality::Ambiguous => "ambiguous across resolved instances",
            CargoDependencyFeatureCausality::Unknown => "unknown",
        },
    ));
    let status = vec![
        fact(
            parent,
            "status-source",
            "Declared source",
            source_label(dependency.source_kind),
        ),
        fact(
            parent,
            "status-completeness",
            "Completeness",
            if dependency.resolution_truncated {
                "truncated"
            } else {
                "complete within available metadata"
            },
        ),
        fact(
            parent,
            "status-cycle",
            "Workspace cycle",
            if dependency.cycle_detected {
                "present"
            } else {
                "not observed"
            },
        ),
    ];
    CargoDependencyInsightProjection {
        children: vec![
            section(parent, "declared", "Declared", declared),
            section(parent, "resolved", "Resolved", resolved),
            section(parent, "features", "Features", features),
            section(parent, "source-lock", "Source and lock", status),
        ],
        navigation,
    }
}

fn section(
    parent: &LanguageToolNodeId,
    discriminator: &str,
    label: &str,
    children: Vec<LanguageToolNode>,
) -> LanguageToolNode {
    LanguageToolNode {
        id: id(parent, discriminator),
        label: label.to_string(),
        secondary_label: None,
        icon: None,
        accessibility_label: format!("Cargo dependency {label}"),
        children,
        enabled: false,
        activation_label: None,
    }
}

fn fact(
    parent: &LanguageToolNodeId,
    discriminator: &str,
    label: &str,
    value: &str,
) -> LanguageToolNode {
    fact_node(
        id(parent, discriminator),
        label,
        value,
        &format!("{label}, {value}"),
    )
}

fn fact_node(
    id: LanguageToolNodeId,
    label: &str,
    value: &str,
    accessibility_label: &str,
) -> LanguageToolNode {
    LanguageToolNode {
        id,
        label: label.to_string(),
        secondary_label: Some(value.to_string()),
        icon: None,
        accessibility_label: accessibility_label.to_string(),
        children: Vec::new(),
        enabled: false,
        activation_label: None,
    }
}

fn id(parent: &LanguageToolNodeId, discriminator: &str) -> LanguageToolNodeId {
    LanguageToolNodeId(format!("{}:insight:{discriminator}", parent.0))
}

fn source_label(source: CargoDependencySourceKind) -> &'static str {
    match source {
        CargoDependencySourceKind::Path => "path",
        CargoDependencySourceKind::Registry => "registry",
        CargoDependencySourceKind::Git => "git",
        CargoDependencySourceKind::Other => "other source",
    }
}

fn lock_label(status: CargoDependencyLockStatus) -> &'static str {
    match status {
        CargoDependencyLockStatus::Locked => "locked",
        CargoDependencyLockStatus::NotLocked => "not present in lockfile",
        CargoDependencyLockStatus::MissingLockfile => "lockfile missing",
        CargoDependencyLockStatus::Unknown => "lock status unknown",
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use project::cargo_workspace::{CargoDependencyKind, CargoResolvedDependencyModel};
    use settings::WorktreeId;
    use util::rel_path::RelPath;

    use super::*;

    fn path(value: &str) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_usize(1),
            path: Arc::from(RelPath::from_unix_str(value).expect("valid fixture path")),
        }
    }

    fn dependency(instance_count: usize) -> CargoDependencyModel {
        CargoDependencyModel {
            declaration_name: "serde".to_string(),
            rename: Some("serialization".to_string()),
            kind: CargoDependencyKind::Normal,
            version_requirement: "^1".to_string(),
            optional: true,
            uses_default_features: false,
            requested_features: vec!["derive".to_string()],
            target: Some("cfg(unix)".to_string()),
            source_kind: CargoDependencySourceKind::Registry,
            resolved_name: Some("serde".to_string()),
            resolved_version: Some("1.0.0".to_string()),
            resolved_workspace_member: None,
            local_manifest: None,
            declaration_manifest: Some(path("member/Cargo.toml")),
            declaration_origin: CargoDependencyDeclarationOrigin::WorkspaceInherited,
            resolved_instances: (0..instance_count)
                .map(|index| CargoResolvedDependencyModel {
                    name: "serde".to_string(),
                    version: format!("1.0.{index}"),
                    source_kind: CargoDependencySourceKind::Registry,
                    enabled_features: vec!["derive".to_string()],
                    lock_status: if index % 2 == 0 {
                        CargoDependencyLockStatus::Locked
                    } else {
                        CargoDependencyLockStatus::NotLocked
                    },
                    workspace_member: None,
                    local_manifest: None,
                })
                .collect(),
            resolution_truncated: false,
            feature_causality: if instance_count > 1 {
                CargoDependencyFeatureCausality::Ambiguous
            } else {
                CargoDependencyFeatureCausality::Validated
            },
            cycle_detected: true,
        }
    }

    #[test]
    fn cargo_dependency_insight_is_finite_and_navigates_only_visible_manifests() {
        let projection = cargo_dependency_insight(
            &LanguageToolNodeId("dependency".to_string()),
            &dependency(2),
        );
        assert_eq!(projection.children.len(), 4);
        assert_eq!(projection.navigation.len(), 1);
        assert_eq!(projection.navigation[0].1, path("member/Cargo.toml"));
        assert!(
            projection
                .children
                .iter()
                .all(|section| { section.children.iter().all(|fact| fact.children.is_empty()) })
        );
        let labels = projection
            .children
            .iter()
            .flat_map(|section| &section.children)
            .filter_map(|node| node.secondary_label.as_deref())
            .collect::<Vec<_>>();
        assert!(labels.contains(&"workspace inherited"));
        assert!(labels.contains(&"ambiguous across resolved instances"));
    }

    #[test]
    fn cargo_dependency_insight_large_stays_one_level_and_bounded() {
        let projection = cargo_dependency_insight(
            &LanguageToolNodeId("dependency".to_string()),
            &dependency(project::cargo_workspace::MAX_CARGO_DEPENDENCY_INSTANCES),
        );
        let node_count = projection
            .children
            .iter()
            .map(|section| 1 + section.children.len())
            .sum::<usize>();
        assert!(node_count < 100);
        assert!(
            projection
                .children
                .iter()
                .all(|section| { section.children.iter().all(|fact| fact.children.is_empty()) })
        );
    }
}
