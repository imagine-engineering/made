use crate::{AcceptedMigration, WorkflowFormat, WorkflowFormatDocument, WorkflowFormatError};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const FORMAT_MIGRATION_CATALOG: &[(&str, &str)] = &[
    ("COMFY-WORKFLOW-022", "workflow-json-0.4"),
    ("COMFY-WORKFLOW-023", "workflow-json-1"),
    ("COMFY-WORKFLOW-024", "api-prompt-json"),
    ("COMFY-WORKFLOW-025", "png"),
    ("COMFY-WORKFLOW-026", "avif"),
    ("COMFY-WORKFLOW-027", "webp"),
    ("COMFY-WORKFLOW-028", "mp3"),
    ("COMFY-WORKFLOW-029", "ogg-opus"),
    ("COMFY-WORKFLOW-030", "flac"),
    ("COMFY-WORKFLOW-031", "webm"),
    ("COMFY-WORKFLOW-032", "isobmff-video"),
    ("COMFY-WORKFLOW-033", "svg"),
    ("COMFY-WORKFLOW-034", "glb"),
    ("COMFY-WORKFLOW-035", "latent-safetensors"),
    ("COMFY-WORKFLOW-036", "a1111-parameters"),
    ("COMFY-GRAPH-016", "node-templates"),
    ("COMFY-WORKFLOW-037", "workflow-drafts-v1-v2"),
    ("COMFY-GRAPH-017", "legacy-reroute-native"),
    ("COMFY-GRAPH-018", "proxy-widget"),
    ("COMFY-FRONTEND-EXT-064", "node-def-v1-v2"),
    ("COMFY-SETTING-015", "lod-threshold-font-size"),
    ("COMFY-WORKFLOW-038", "ksampler-widget-values"),
    ("COMFY-GRAPH-019", "renderer-layout-scale"),
    ("COMFY-WORKFLOW-039", "unknown-version-0.4"),
];

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkflowMigrationId {
    LegacyRerouteNative,
    ProxyWidget,
    KSamplerWidgetValues,
    RendererLayoutScale,
    UnknownVersion04,
}

impl WorkflowMigrationId {
    pub fn identifier(self) -> &'static str {
        match self {
            Self::LegacyRerouteNative => "COMFY-GRAPH-017",
            Self::ProxyWidget => "COMFY-GRAPH-018",
            Self::KSamplerWidgetValues => "COMFY-WORKFLOW-038",
            Self::RendererLayoutScale => "COMFY-GRAPH-019",
            Self::UnknownVersion04 => "COMFY-WORKFLOW-039",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MigrationCandidate {
    pub identifier: WorkflowMigrationId,
    pub reason: String,
    pub paths: Vec<String>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkflowMigrationError {
    #[error("migration `{0:?}` does not apply to this workflow")]
    NotApplicable(WorkflowMigrationId),
    #[error("migration `{migration:?}` conflicts at {path}: {reason}")]
    Conflict {
        migration: WorkflowMigrationId,
        path: String,
        reason: String,
    },
    #[error("migration `{migration:?}` failed at {path}: {reason}")]
    Invalid {
        migration: WorkflowMigrationId,
        path: String,
        reason: String,
    },
    #[error(transparent)]
    Workflow(#[from] WorkflowFormatError),
}

pub fn detect_workflow_migrations(document: &WorkflowFormatDocument) -> Vec<MigrationCandidate> {
    let mut candidates = Vec::new();
    let value = document.value();
    let reroute_paths = find_legacy_reroute_paths(value);
    if !reroute_paths.is_empty() {
        candidates.push(MigrationCandidate {
            identifier: WorkflowMigrationId::LegacyRerouteNative,
            reason: "legacy Reroute nodes are present".to_owned(),
            paths: reroute_paths,
        });
    }
    let proxy_paths = find_node_paths(value, |node| {
        node.get("properties")
            .and_then(|properties| properties.get("proxyWidgets"))
            .is_some()
    });
    if !proxy_paths.is_empty() {
        candidates.push(MigrationCandidate {
            identifier: WorkflowMigrationId::ProxyWidget,
            reason: "legacy proxyWidgets data is present".to_owned(),
            paths: proxy_paths,
        });
    }
    let sampler_paths = find_node_paths(value, sampler_needs_migration);
    if !sampler_paths.is_empty() {
        candidates.push(MigrationCandidate {
            identifier: WorkflowMigrationId::KSamplerWidgetValues,
            reason: "legacy KSampler widget values are present".to_owned(),
            paths: sampler_paths,
        });
    }
    let renderer_paths = find_graph_paths(value, |graph| {
        graph
            .get("extra")
            .and_then(|extra| extra.get("workflowRendererVersion"))
            .and_then(Value::as_str)
            == Some("Vue")
    });
    if !renderer_paths.is_empty() {
        candidates.push(MigrationCandidate {
            identifier: WorkflowMigrationId::RendererLayoutScale,
            reason: "legacy Vue-scaled geometry is present".to_owned(),
            paths: renderer_paths,
        });
    }
    if document.format() == Some(WorkflowFormat::OtherNumeric) {
        candidates.push(MigrationCandidate {
            identifier: WorkflowMigrationId::UnknownVersion04,
            reason: "numeric versions other than 1 use the schema 0.4 compatibility adapter"
                .to_owned(),
            paths: vec!["$.version".to_owned()],
        });
    }
    candidates.sort_by_key(|candidate| candidate.identifier);
    candidates
}

fn find_legacy_reroute_paths(value: &Value) -> Vec<String> {
    let mut paths = Vec::new();
    visit_graphs(value, "$", &mut |graph, path| {
        let has_native_reroutes =
            graph
                .get("extra")
                .and_then(Value::as_object)
                .is_some_and(|extra| {
                    ["reroutes", "linkExtensions"].into_iter().any(|field| {
                        extra
                            .get(field)
                            .and_then(Value::as_array)
                            .is_some_and(|values| !values.is_empty())
                    })
                });
        if has_native_reroutes {
            return;
        }
        if let Some(nodes) = graph.get("nodes").and_then(Value::as_array) {
            for (index, node) in nodes.iter().enumerate() {
                if node.get("type").and_then(Value::as_str) == Some("Reroute") {
                    paths.push(format!("{path}.nodes[{index}]"));
                }
            }
        }
    });
    paths
}

pub fn apply_workflow_migrations(
    document: &WorkflowFormatDocument,
    accepted: &[WorkflowMigrationId],
) -> Result<WorkflowFormatDocument, WorkflowMigrationError> {
    let mut result = document.clone();
    let candidates = detect_workflow_migrations(document)
        .into_iter()
        .map(|candidate| candidate.identifier)
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for migration in accepted {
        if !seen.insert(*migration) {
            continue;
        }
        if !candidates.contains(migration) {
            return Err(WorkflowMigrationError::NotApplicable(*migration));
        }
        let mut value = result.value().clone();
        match migration {
            WorkflowMigrationId::LegacyRerouteNative => migrate_reroutes(&mut value)?,
            WorkflowMigrationId::ProxyWidget => migrate_proxy_widgets(&mut value)?,
            WorkflowMigrationId::KSamplerWidgetValues => {
                migrate_ksampler_widgets(&mut value)?;
            }
            WorkflowMigrationId::RendererLayoutScale => {
                migrate_renderer_scale(&mut value)?;
            }
            WorkflowMigrationId::UnknownVersion04 => {}
        }
        result = result.with_accepted_migration(
            value,
            AcceptedMigration {
                identifier: migration.identifier().to_owned(),
                version: 1,
                provenance: "frontend-formats-migrations.csv".to_owned(),
            },
        )?;
    }
    Ok(result)
}

fn find_node_paths(
    value: &Value,
    predicate: impl Fn(&Map<String, Value>) -> bool + Copy,
) -> Vec<String> {
    let mut paths = Vec::new();
    visit_graphs(value, "$", &mut |graph, path| {
        if let Some(nodes) = graph.get("nodes").and_then(Value::as_array) {
            for (index, node) in nodes.iter().enumerate() {
                if node.as_object().is_some_and(predicate) {
                    paths.push(format!("{path}.nodes[{index}]"));
                }
            }
        }
    });
    paths
}

fn find_graph_paths(
    value: &Value,
    predicate: impl Fn(&Map<String, Value>) -> bool + Copy,
) -> Vec<String> {
    let mut paths = Vec::new();
    visit_graphs(value, "$", &mut |graph, path| {
        if predicate(graph) {
            paths.push(path.to_owned());
        }
    });
    paths
}

fn visit_graphs(value: &Value, path: &str, visitor: &mut impl FnMut(&Map<String, Value>, &str)) {
    let Some(graph) = value.as_object() else {
        return;
    };
    visitor(graph, path);
    if let Some(definitions) = graph.get("definitions") {
        match definitions {
            Value::Array(definitions) => {
                for (index, definition) in definitions.iter().enumerate() {
                    visit_graphs(definition, &format!("{path}.definitions[{index}]"), visitor);
                }
            }
            Value::Object(definitions) => {
                if let Some(subgraphs) = definitions.get("subgraphs").and_then(Value::as_array) {
                    for (index, definition) in subgraphs.iter().enumerate() {
                        visit_graphs(
                            definition,
                            &format!("{path}.definitions.subgraphs[{index}]"),
                            visitor,
                        );
                    }
                } else {
                    for (identifier, definition) in definitions {
                        visit_graphs(
                            definition,
                            &format!("{path}.definitions.{identifier}"),
                            visitor,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

fn visit_graphs_mut(
    value: &mut Value,
    path: &str,
    visitor: &mut impl FnMut(&mut Map<String, Value>, &str) -> Result<(), WorkflowMigrationError>,
) -> Result<(), WorkflowMigrationError> {
    let Some(graph) = value.as_object_mut() else {
        return Ok(());
    };
    visitor(graph, path)?;
    if let Some(definitions) = graph.get_mut("definitions") {
        match definitions {
            Value::Array(definitions) => {
                for (index, definition) in definitions.iter_mut().enumerate() {
                    visit_graphs_mut(definition, &format!("{path}.definitions[{index}]"), visitor)?;
                }
            }
            Value::Object(definitions) => {
                if let Some(subgraphs) = definitions
                    .get_mut("subgraphs")
                    .and_then(Value::as_array_mut)
                {
                    for (index, definition) in subgraphs.iter_mut().enumerate() {
                        visit_graphs_mut(
                            definition,
                            &format!("{path}.definitions.subgraphs[{index}]"),
                            visitor,
                        )?;
                    }
                } else {
                    for (identifier, definition) in definitions {
                        visit_graphs_mut(
                            definition,
                            &format!("{path}.definitions.{identifier}"),
                            visitor,
                        )?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn sampler_needs_migration(node: &Map<String, Value>) -> bool {
    let node_type = node.get("type").and_then(Value::as_str);
    if !matches!(
        node_type,
        Some("KSampler" | "KSamplerAdvanced" | "PrimitiveNode")
    ) {
        return false;
    }
    node.get("widgets_values")
        .and_then(Value::as_array)
        .is_some_and(|widgets| {
            if node_type == Some("PrimitiveNode") {
                widgets.first().is_some_and(|widget| {
                    widget
                        .as_str()
                        .is_some_and(|value| value.starts_with("sample_"))
                }) || widgets.get(1).is_some_and(Value::is_boolean)
            } else {
                widgets.iter().any(|widget| {
                    widget
                        .as_str()
                        .is_some_and(|value| value.starts_with("sample_"))
                        || widget.is_boolean()
                })
            }
        })
}

fn migrate_ksampler_widgets(value: &mut Value) -> Result<(), WorkflowMigrationError> {
    visit_graphs_mut(value, "$", &mut |graph, path| {
        let Some(nodes) = graph.get_mut("nodes").and_then(Value::as_array_mut) else {
            return Ok(());
        };
        for (index, node) in nodes.iter_mut().enumerate() {
            let Some(node) = node.as_object_mut() else {
                continue;
            };
            let node_type = node
                .get("type")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let Some(widgets) = node.get_mut("widgets_values").and_then(Value::as_array_mut) else {
                continue;
            };
            let (control_index, sampler_index) = match node_type.as_deref() {
                Some("KSampler") => (Some(1usize), Some(4usize)),
                Some("KSamplerAdvanced") => (Some(2usize), Some(5usize)),
                Some("PrimitiveNode") => (
                    widgets.get(1).is_some_and(Value::is_boolean).then_some(1),
                    widgets
                        .first()
                        .is_some_and(|value| {
                            value
                                .as_str()
                                .is_some_and(|value| value.starts_with("sample_"))
                        })
                        .then_some(0),
                ),
                _ => continue,
            };
            if let Some(index) = sampler_index {
                if let Some(value) = widgets.get_mut(index) {
                    if let Some(sampler) = value
                        .as_str()
                        .and_then(|value| value.strip_prefix("sample_"))
                    {
                        *value = Value::String(sampler.to_owned());
                    }
                }
            }
            if let Some(control_index) = control_index {
                if let Some(value) = widgets.get_mut(control_index) {
                    if let Some(boolean) = value.as_bool() {
                        *value =
                            Value::String(if boolean { "randomize" } else { "fixed" }.to_owned());
                    }
                }
            }
            if widgets.len() > 100_000 {
                return Err(WorkflowMigrationError::Invalid {
                    migration: WorkflowMigrationId::KSamplerWidgetValues,
                    path: format!("{path}.nodes[{index}].widgets_values"),
                    reason: "widget count exceeds the migration limit".to_owned(),
                });
            }
        }
        Ok(())
    })
}

fn migrate_proxy_widgets(value: &mut Value) -> Result<(), WorkflowMigrationError> {
    visit_graphs_mut(value, "$", &mut |graph, path| {
        let Some(nodes) = graph.get_mut("nodes").and_then(Value::as_array_mut) else {
            return Ok(());
        };
        for (index, node) in nodes.iter_mut().enumerate() {
            let Some(node) = node.as_object_mut() else {
                continue;
            };
            let Some(properties) = node.get_mut("properties").and_then(Value::as_object_mut) else {
                continue;
            };
            let Some(proxy_widgets) = properties.remove("proxyWidgets") else {
                continue;
            };
            let tuples =
                proxy_widgets
                    .as_array()
                    .ok_or_else(|| WorkflowMigrationError::Invalid {
                        migration: WorkflowMigrationId::ProxyWidget,
                        path: format!("{path}.nodes[{index}].properties.proxyWidgets"),
                        reason: "proxyWidgets must be an array".to_owned(),
                    })?;
            let quarantine = properties
                .entry("proxyWidgetErrorQuarantine")
                .or_insert_with(|| json!({"version":1,"entries":[]}));
            let entries = quarantine
                .get_mut("entries")
                .and_then(Value::as_array_mut)
                .ok_or_else(|| WorkflowMigrationError::Conflict {
                    migration: WorkflowMigrationId::ProxyWidget,
                    path: format!("{path}.nodes[{index}].properties.proxyWidgetErrorQuarantine"),
                    reason: "existing quarantine is not versioned entry storage".to_owned(),
                })?;
            for tuple in tuples {
                entries.push(json!({
                    "source": tuple,
                    "reason": "requires-live-subgraph-resolution",
                    "hostValuePreserved": true,
                }));
            }
        }
        Ok(())
    })
}

#[derive(Clone, Debug)]
struct LegacyLink {
    identifier: Value,
    origin: Value,
    origin_slot: Value,
    target: Value,
    target_slot: Value,
    type_name: Value,
}

fn migrate_reroutes(value: &mut Value) -> Result<(), WorkflowMigrationError> {
    visit_graphs_mut(value, "$", &mut |graph, path| {
        migrate_graph_reroutes(graph, path)
    })
}

fn migrate_graph_reroutes(
    graph: &mut Map<String, Value>,
    path: &str,
) -> Result<(), WorkflowMigrationError> {
    let existing_native = graph
        .get("extra")
        .and_then(|extra| extra.get("reroutes"))
        .and_then(Value::as_array)
        .is_some_and(|reroutes| !reroutes.is_empty())
        || graph
            .get("extra")
            .and_then(|extra| extra.get("linkExtensions"))
            .and_then(Value::as_array)
            .is_some_and(|extensions| !extensions.is_empty());
    if existing_native {
        return Err(WorkflowMigrationError::Conflict {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.extra.reroutes"),
            reason: "native reroutes already exist".to_owned(),
        });
    }
    let nodes = graph
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.nodes"),
            reason: "nodes must be an array".to_owned(),
        })?;
    let node_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let identifier = node.get("id").and_then(identity).ok_or_else(|| {
                WorkflowMigrationError::Invalid {
                    migration: WorkflowMigrationId::LegacyRerouteNative,
                    path: format!("{path}.nodes[{index}].id"),
                    reason: "node ID must be a string or number".to_owned(),
                }
            })?;
            Ok((identifier, node.clone()))
        })
        .collect::<Result<BTreeMap<_, _>, WorkflowMigrationError>>()?;
    if node_by_id.len() != nodes.len() {
        return Err(WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.nodes"),
            reason: "node IDs must be unique".to_owned(),
        });
    }
    let reroute_nodes = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.get("type").and_then(Value::as_str) == Some("Reroute"))
        .map(|(index, node)| {
            let identifier = node.get("id").and_then(identity).ok_or_else(|| {
                WorkflowMigrationError::Invalid {
                    migration: WorkflowMigrationId::LegacyRerouteNative,
                    path: format!("{path}.nodes[{index}].id"),
                    reason: "reroute ID must be a string or number".to_owned(),
                }
            })?;
            let position = node.get("pos").and_then(Value::as_array).ok_or_else(|| {
                WorkflowMigrationError::Invalid {
                    migration: WorkflowMigrationId::LegacyRerouteNative,
                    path: format!("{path}.nodes[{index}].pos"),
                    reason: "reroute position must be a pair".to_owned(),
                }
            })?;
            let size = node.get("size").and_then(Value::as_array).ok_or_else(|| {
                WorkflowMigrationError::Invalid {
                    migration: WorkflowMigrationId::LegacyRerouteNative,
                    path: format!("{path}.nodes[{index}].size"),
                    reason: "reroute size must be a pair".to_owned(),
                }
            })?;
            let center = [
                pair_coordinate(position, 0, path, index, "pos")?
                    + pair_coordinate(size, 0, path, index, "size")? / 2.0,
                pair_coordinate(position, 1, path, index, "pos")?
                    + pair_coordinate(size, 1, path, index, "size")? / 2.0,
            ];
            Ok((
                identifier,
                Value::Array(vec![json_number(center[0])?, json_number(center[1])?]),
            ))
        })
        .collect::<Result<Vec<_>, WorkflowMigrationError>>()?;
    if reroute_nodes.is_empty() {
        return Ok(());
    }
    let reroute_keys = reroute_nodes
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<BTreeSet<_>>();
    let links = graph
        .get("links")
        .and_then(Value::as_array)
        .ok_or_else(|| WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.links"),
            reason: "links must be an array".to_owned(),
        })?
        .iter()
        .enumerate()
        .map(|(index, link)| parse_legacy_link(link, path, index))
        .collect::<Result<Vec<_>, _>>()?;
    let link_by_id = links
        .iter()
        .map(|link| {
            let identifier =
                identity(&link.identifier).ok_or_else(|| WorkflowMigrationError::Invalid {
                    migration: WorkflowMigrationId::LegacyRerouteNative,
                    path: format!("{path}.links"),
                    reason: "link ID must be a string or number".to_owned(),
                })?;
            Ok((identifier, link.clone()))
        })
        .collect::<Result<BTreeMap<_, _>, WorkflowMigrationError>>()?;
    if link_by_id.len() != links.len() {
        return Err(WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.links"),
            reason: "link IDs must be unique".to_owned(),
        });
    }
    let mut new_links = links
        .iter()
        .filter(|link| {
            !identity(&link.origin).is_some_and(|key| reroute_keys.contains(&key))
                && !identity(&link.target).is_some_and(|key| reroute_keys.contains(&key))
        })
        .map(link_value)
        .collect::<Vec<_>>();
    let ending_links = links
        .iter()
        .filter(|link| {
            identity(&link.origin).is_some_and(|key| reroute_keys.contains(&key))
                && !identity(&link.target).is_some_and(|key| reroute_keys.contains(&key))
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut floating_links = Vec::new();
    let mut link_extensions = Vec::new();
    let mut valid_reroutes = BTreeSet::new();
    let mut valid_reroute_order = Vec::new();
    let mut reroute_records = BTreeMap::<String, Value>::new();
    for (index, (key, position)) in reroute_nodes.iter().enumerate() {
        reroute_records.insert(
            key.clone(),
            json!({"id": index + 1, "pos": position, "linkIds": []}),
        );
    }
    let mut floating_identifier = reroute_nodes.len() + 1;
    for ending_link in &ending_links {
        let ending_key =
            identity(&ending_link.origin).ok_or_else(|| WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::LegacyRerouteNative,
                path: format!("{path}.links"),
                reason: "reroute origin ID is invalid".to_owned(),
            })?;
        let chain =
            legacy_reroute_chain(&ending_key, &node_by_id, &link_by_id, &reroute_keys, path)?;
        let first_reroute_id = connect_reroute_chain(
            &chain,
            &mut reroute_records,
            &mut valid_reroutes,
            &mut valid_reroute_order,
            path,
        )?;
        if let Some(starting_link) = chain_starting_link(&chain, &node_by_id, &link_by_id) {
            let mut projected = ending_link.clone();
            projected.origin = starting_link.origin.clone();
            projected.origin_slot = starting_link.origin_slot.clone();
            new_links.push(link_value(&projected));
            link_extensions.push(json!({
                "id": ending_link.identifier,
                "parentId": first_reroute_id,
            }));
            for key in &chain {
                let record = reroute_records
                    .get_mut(key)
                    .and_then(Value::as_object_mut)
                    .ok_or_else(|| missing_reroute_record(path))?;
                let link_ids = record
                    .get_mut("linkIds")
                    .and_then(Value::as_array_mut)
                    .ok_or_else(|| missing_reroute_record(path))?;
                link_ids.push(ending_link.identifier.clone());
                record.remove("floating");
            }
        } else {
            mark_floating_reroutes(&chain, &mut reroute_records, "input", path)?;
            floating_links.push(json!({
                "id": floating_identifier,
                "origin_id": -1,
                "origin_slot": -1,
                "target_id": ending_link.target,
                "target_slot": ending_link.target_slot,
                "type": ending_link.type_name,
                "parentId": first_reroute_id,
            }));
            floating_identifier += 1;
        }
    }

    for (reroute_key, _) in &reroute_nodes {
        let node = node_by_id
            .get(reroute_key)
            .ok_or_else(|| missing_reroute_record(path))?;
        let has_output_without_links = node
            .get("outputs")
            .and_then(Value::as_array)
            .and_then(|outputs| outputs.first())
            .is_some_and(|output| {
                output
                    .get("links")
                    .and_then(Value::as_array)
                    .is_none_or(Vec::is_empty)
            });
        if !has_output_without_links {
            continue;
        }
        let chain =
            legacy_reroute_chain(reroute_key, &node_by_id, &link_by_id, &reroute_keys, path)?;
        let Some(starting_link) = chain_starting_link(&chain, &node_by_id, &link_by_id) else {
            continue;
        };
        let first_reroute_id = connect_reroute_chain(
            &chain,
            &mut reroute_records,
            &mut valid_reroutes,
            &mut valid_reroute_order,
            path,
        )?;
        mark_floating_reroutes(&chain, &mut reroute_records, "output", path)?;
        floating_links.push(json!({
            "id": floating_identifier,
            "origin_id": starting_link.origin,
            "origin_slot": starting_link.origin_slot,
            "target_id": -1,
            "target_slot": -1,
            "type": starting_link.type_name,
            "parentId": first_reroute_id,
        }));
        floating_identifier += 1;
    }

    let retained_nodes = graph
        .get_mut("nodes")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.nodes"),
            reason: "nodes must be an array".to_owned(),
        })?;
    retained_nodes.retain(|node| node.get("type").and_then(Value::as_str) != Some("Reroute"));
    sort_like_javascript_object_values(retained_nodes);
    reconnect_legacy_node_slots(retained_nodes, &new_links, path)?;
    graph.insert("links".to_owned(), Value::Array(new_links));
    if floating_links.is_empty() {
        graph.remove("floatingLinks");
    } else {
        graph.insert("floatingLinks".to_owned(), Value::Array(floating_links));
    }
    let extra = graph
        .entry("extra")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| WorkflowMigrationError::Conflict {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.extra"),
            reason: "extra is not an object".to_owned(),
        })?;
    extra.insert(
        "reroutes".to_owned(),
        Value::Array(
            valid_reroute_order
                .iter()
                .filter_map(|key| reroute_records.remove(key))
                .collect(),
        ),
    );
    extra.insert("linkExtensions".to_owned(), Value::Array(link_extensions));
    Ok(())
}

fn pair_coordinate(
    pair: &[Value],
    coordinate: usize,
    path: &str,
    node_index: usize,
    field: &str,
) -> Result<f64, WorkflowMigrationError> {
    pair.get(coordinate)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.nodes[{node_index}].{field}"),
            reason: "reroute geometry must contain two finite numbers".to_owned(),
        })
}

fn legacy_reroute_chain(
    ending_key: &str,
    node_by_id: &BTreeMap<String, Value>,
    link_by_id: &BTreeMap<String, LegacyLink>,
    reroute_keys: &BTreeSet<String>,
    path: &str,
) -> Result<Vec<String>, WorkflowMigrationError> {
    let mut chain = Vec::new();
    let mut current = ending_key.to_owned();
    while reroute_keys.contains(&current) {
        if chain.contains(&current) {
            return Err(WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::LegacyRerouteNative,
                path: format!("{path}.links"),
                reason: "reroute links contain a cycle".to_owned(),
            });
        }
        chain.push(current.clone());
        let Some(node) = node_by_id.get(&current) else {
            break;
        };
        let Some(link_key) = reroute_input_link_key(node) else {
            break;
        };
        let Some(link) = link_by_id.get(&link_key) else {
            break;
        };
        let Some(origin) = identity(&link.origin) else {
            break;
        };
        current = origin;
    }
    Ok(chain)
}

fn chain_starting_link<'a>(
    chain: &[String],
    node_by_id: &BTreeMap<String, Value>,
    link_by_id: &'a BTreeMap<String, LegacyLink>,
) -> Option<&'a LegacyLink> {
    chain
        .last()
        .and_then(|key| node_by_id.get(key))
        .and_then(reroute_input_link_key)
        .and_then(|key| link_by_id.get(&key))
}

fn reroute_input_link_key(node: &Value) -> Option<String> {
    node.get("inputs")?
        .as_array()?
        .first()?
        .get("link")
        .filter(|link| !link.is_null())
        .and_then(identity)
}

fn connect_reroute_chain(
    chain: &[String],
    reroute_records: &mut BTreeMap<String, Value>,
    valid_reroutes: &mut BTreeSet<String>,
    valid_reroute_order: &mut Vec<String>,
    path: &str,
) -> Result<Value, WorkflowMigrationError> {
    let first = chain.first().ok_or_else(|| missing_reroute_record(path))?;
    for key in chain {
        if valid_reroutes.insert(key.clone()) {
            valid_reroute_order.push(key.clone());
        }
    }
    for pair in chain.windows(2) {
        let parent_id = reroute_records
            .get(&pair[1])
            .and_then(|record| record.get("id"))
            .cloned()
            .ok_or_else(|| missing_reroute_record(path))?;
        reroute_records
            .get_mut(&pair[0])
            .and_then(Value::as_object_mut)
            .ok_or_else(|| missing_reroute_record(path))?
            .insert("parentId".to_owned(), parent_id);
    }
    reroute_records
        .get(first)
        .and_then(|record| record.get("id"))
        .cloned()
        .ok_or_else(|| missing_reroute_record(path))
}

fn mark_floating_reroutes(
    chain: &[String],
    reroute_records: &mut BTreeMap<String, Value>,
    slot_type: &str,
    path: &str,
) -> Result<(), WorkflowMigrationError> {
    for key in chain {
        let record = reroute_records
            .get_mut(key)
            .and_then(Value::as_object_mut)
            .ok_or_else(|| missing_reroute_record(path))?;
        let has_links = record
            .get("linkIds")
            .and_then(Value::as_array)
            .is_some_and(|links| !links.is_empty());
        if !has_links {
            record.insert("floating".to_owned(), json!({"slotType": slot_type}));
        }
    }
    Ok(())
}

fn missing_reroute_record(path: &str) -> WorkflowMigrationError {
    WorkflowMigrationError::Invalid {
        migration: WorkflowMigrationId::LegacyRerouteNative,
        path: format!("{path}.extra.reroutes"),
        reason: "reroute conversion state is inconsistent".to_owned(),
    }
}

fn reconnect_legacy_node_slots(
    nodes: &mut [Value],
    links: &[Value],
    path: &str,
) -> Result<(), WorkflowMigrationError> {
    let mut node_indices = BTreeMap::new();
    for (index, node) in nodes.iter_mut().enumerate() {
        let Some(node) = node.as_object_mut() else {
            continue;
        };
        let identifier =
            node.get("id")
                .and_then(identity)
                .ok_or_else(|| WorkflowMigrationError::Invalid {
                    migration: WorkflowMigrationId::LegacyRerouteNative,
                    path: format!("{path}.nodes[{index}].id"),
                    reason: "node ID must be a string or number".to_owned(),
                })?;
        if node_indices.insert(identifier, index).is_some() {
            return Err(WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::LegacyRerouteNative,
                path: format!("{path}.nodes[{index}].id"),
                reason: "node ID is duplicated".to_owned(),
            });
        }
        if let Some(inputs) = node.get_mut("inputs").and_then(Value::as_array_mut) {
            for input in inputs {
                if let Some(input) = input.as_object_mut() {
                    input.insert("link".to_owned(), Value::Null);
                }
            }
        }
        if let Some(outputs) = node.get_mut("outputs").and_then(Value::as_array_mut) {
            for output in outputs {
                if let Some(output) = output.as_object_mut() {
                    output.insert("links".to_owned(), Value::Array(Vec::new()));
                }
            }
        }
    }
    for (link_index, link) in links.iter().enumerate() {
        let link = link
            .as_array()
            .ok_or_else(|| WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::LegacyRerouteNative,
                path: format!("{path}.links[{link_index}]"),
                reason: "link must be an array".to_owned(),
            })?;
        let origin =
            identity(&link[1]).and_then(|identifier| node_indices.get(&identifier).copied());
        let Some(origin) = origin else {
            continue;
        };
        let output_index = link[2]
            .as_u64()
            .and_then(|index| usize::try_from(index).ok());
        let Some(output) = nodes
            .get_mut(origin)
            .and_then(|node| node.get_mut("outputs"))
            .and_then(Value::as_array_mut)
            .and_then(|outputs| output_index.and_then(|index| outputs.get_mut(index)))
            .and_then(Value::as_object_mut)
        else {
            return Err(WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::LegacyRerouteNative,
                path: format!("{path}.links[{link_index}][2]"),
                reason: "origin slot is unavailable".to_owned(),
            });
        };
        let output_links = output
            .entry("links")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::LegacyRerouteNative,
                path: format!("{path}.nodes[{origin}].outputs"),
                reason: "output links are not an array".to_owned(),
            })?;
        output_links.push(link[0].clone());
    }
    for (link_index, link) in links.iter().enumerate() {
        let link = link
            .as_array()
            .ok_or_else(|| WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::LegacyRerouteNative,
                path: format!("{path}.links[{link_index}]"),
                reason: "link must be an array".to_owned(),
            })?;
        let target =
            identity(&link[3]).and_then(|identifier| node_indices.get(&identifier).copied());
        let Some(target) = target else {
            continue;
        };
        let input_index = link[4]
            .as_u64()
            .and_then(|index| usize::try_from(index).ok());
        let Some(input) = nodes
            .get_mut(target)
            .and_then(|node| node.get_mut("inputs"))
            .and_then(Value::as_array_mut)
            .and_then(|inputs| input_index.and_then(|index| inputs.get_mut(index)))
            .and_then(Value::as_object_mut)
        else {
            return Err(WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::LegacyRerouteNative,
                path: format!("{path}.links[{link_index}][4]"),
                reason: "target slot is unavailable".to_owned(),
            });
        };
        input.insert("link".to_owned(), link[0].clone());
    }
    Ok(())
}

fn parse_legacy_link(
    link: &Value,
    path: &str,
    index: usize,
) -> Result<LegacyLink, WorkflowMigrationError> {
    let link = link
        .as_array()
        .ok_or_else(|| WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.links[{index}]"),
            reason: "link must be an array".to_owned(),
        })?;
    if link.len() != 6 {
        return Err(WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::LegacyRerouteNative,
            path: format!("{path}.links[{index}]"),
            reason: "link must have six entries".to_owned(),
        });
    }
    Ok(LegacyLink {
        identifier: link[0].clone(),
        origin: link[1].clone(),
        origin_slot: link[2].clone(),
        target: link[3].clone(),
        target_slot: link[4].clone(),
        type_name: link[5].clone(),
    })
}

fn link_value(link: &LegacyLink) -> Value {
    json!([
        link.identifier,
        link.origin,
        link.origin_slot,
        link.target,
        link.target_slot,
        link.type_name
    ])
}

fn identity(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn sort_like_javascript_object_values(nodes: &mut Vec<Value>) {
    let mut indexed = nodes.drain(..).enumerate().collect::<Vec<_>>();
    indexed.sort_by_key(|(original_index, node)| {
        let array_index = node
            .get("id")
            .and_then(javascript_array_index)
            .unwrap_or(u64::MAX);
        let category = u8::from(array_index == u64::MAX);
        (category, array_index, *original_index)
    });
    nodes.extend(indexed.into_iter().map(|(_, node)| node));
}

fn javascript_array_index(value: &Value) -> Option<u64> {
    let text = match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        _ => return None,
    };
    let index = text.parse::<u64>().ok()?;
    (index < u64::from(u32::MAX) && index.to_string() == text).then_some(index)
}

fn json_number(value: f64) -> Result<Value, WorkflowMigrationError> {
    if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
        return Ok(Value::Number(Number::from(value as i64)));
    }
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::RendererLayoutScale,
            path: "$".to_owned(),
            reason: "migration produced non-finite geometry".to_owned(),
        })
}

fn migrate_renderer_scale(value: &mut Value) -> Result<(), WorkflowMigrationError> {
    visit_graphs_mut(value, "$", &mut |graph, path| {
        let renderer = graph
            .get("extra")
            .and_then(|extra| extra.get("workflowRendererVersion"))
            .and_then(Value::as_str);
        if renderer != Some("Vue") {
            return Ok(());
        }
        let anchor = graph
            .get("nodes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|node| node.get("pos").and_then(vector_pair))
            .fold(None, |anchor: Option<[f64; 2]>, position| {
                Some(match anchor {
                    Some(anchor) => [anchor[0].min(position[0]), anchor[1].min(position[1])],
                    None => position,
                })
            })
            .unwrap_or([0.0, 0.0]);
        if let Some(nodes) = graph.get_mut("nodes").and_then(Value::as_array_mut) {
            for (index, node) in nodes.iter_mut().enumerate() {
                scale_pair(node, "pos", path, "nodes", index, anchor)?;
                scale_pair(node, "size", path, "nodes", index, [0.0, 0.0])?;
            }
        }
        if let Some(groups) = graph.get_mut("groups").and_then(Value::as_array_mut) {
            for (index, group) in groups.iter_mut().enumerate() {
                if group.get("bounding").is_some() {
                    scale_bounds(group, "bounding", path, "groups", index, anchor)?;
                } else {
                    scale_pair(group, "pos", path, "groups", index, anchor)?;
                    scale_pair(group, "size", path, "groups", index, [0.0, 0.0])?;
                }
            }
        }
        for boundary in ["inputNode", "outputNode"] {
            if let Some(node) = graph.get_mut(boundary) {
                scale_bounds(node, "bounding", path, boundary, 0, anchor)?;
            }
        }
        scale_reroutes(graph, path, anchor)?;
        let extra = graph
            .get_mut("extra")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::RendererLayoutScale,
                path: format!("{path}.extra"),
                reason: "extra must be an object".to_owned(),
            })?;
        extra.insert(
            "workflowRendererVersion".to_owned(),
            Value::String("Vue-corrected".to_owned()),
        );
        Ok(())
    })
}

fn scale_reroutes(
    graph: &mut Map<String, Value>,
    path: &str,
    anchor: [f64; 2],
) -> Result<(), WorkflowMigrationError> {
    if let Some(reroutes) = graph.get_mut("reroutes").and_then(Value::as_array_mut) {
        for (index, reroute) in reroutes.iter_mut().enumerate() {
            scale_pair(reroute, "pos", path, "reroutes", index, anchor)?;
        }
    }
    if let Some(reroutes) = graph
        .get_mut("extra")
        .and_then(|extra| extra.get_mut("reroutes"))
        .and_then(Value::as_array_mut)
    {
        for (index, reroute) in reroutes.iter_mut().enumerate() {
            scale_pair(reroute, "pos", path, "extra.reroutes", index, anchor)?;
        }
    }
    Ok(())
}

fn vector_pair(value: &Value) -> Option<[f64; 2]> {
    if let Some(pair) = value.as_array() {
        return Some([pair.first()?.as_f64()?, pair.get(1)?.as_f64()?]);
    }
    let pair = value.as_object()?;
    Some([pair.get("0")?.as_f64()?, pair.get("1")?.as_f64()?])
}

fn scale_pair(
    item: &mut Value,
    field: &str,
    graph_path: &str,
    collection: &str,
    index: usize,
    anchor: [f64; 2],
) -> Result<(), WorkflowMigrationError> {
    let Some(value) = item.get_mut(field) else {
        return Ok(());
    };
    let pair = vector_pair(value).ok_or_else(|| WorkflowMigrationError::Invalid {
        migration: WorkflowMigrationId::RendererLayoutScale,
        path: format!("{graph_path}.{collection}[{index}].{field}"),
        reason: "geometry must contain two finite numbers".to_owned(),
    })?;
    let mut scaled = Vec::with_capacity(2);
    for (coordinate_index, value) in pair.into_iter().enumerate() {
        let coordinate = anchor[coordinate_index] + (value - anchor[coordinate_index]) / 1.2;
        scaled.push(
            json_number(coordinate).map_err(|_| WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::RendererLayoutScale,
                path: format!("{graph_path}.{collection}[{index}].{field}"),
                reason: "scaled geometry is not finite".to_owned(),
            })?,
        );
    }
    *value = Value::Array(scaled);
    Ok(())
}

fn scale_bounds(
    item: &mut Value,
    field: &str,
    graph_path: &str,
    collection: &str,
    index: usize,
    anchor: [f64; 2],
) -> Result<(), WorkflowMigrationError> {
    let Some(bounds) = item.get_mut(field).and_then(Value::as_array_mut) else {
        return Ok(());
    };
    if bounds.len() != 4 {
        return Err(WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::RendererLayoutScale,
            path: format!("{graph_path}.{collection}[{index}].{field}"),
            reason: "bounds must contain four numbers".to_owned(),
        });
    }
    for (coordinate_index, coordinate) in bounds.iter_mut().enumerate() {
        let value = coordinate
            .as_f64()
            .ok_or_else(|| WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::RendererLayoutScale,
                path: format!("{graph_path}.{collection}[{index}].{field}"),
                reason: "geometry must contain finite numbers".to_owned(),
            })?;
        let scaled = if coordinate_index < 2 {
            anchor[coordinate_index] + (value - anchor[coordinate_index]) / 1.2
        } else {
            value / 1.2
        };
        *coordinate = json_number(scaled).map_err(|_| WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::RendererLayoutScale,
            path: format!("{graph_path}.{collection}[{index}].{field}"),
            reason: "scaled geometry is not finite".to_owned(),
        })?;
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DraftV1 {
    pub data: String,
    pub updated_at: u64,
    pub name: String,
    pub is_temporary: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DraftStoreV1 {
    pub drafts: BTreeMap<String, DraftV1>,
    pub order: Vec<String>,
    pub open_paths: Vec<String>,
    pub active_index: usize,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DraftIndexEntryV2 {
    pub path: String,
    pub key: String,
    pub name: String,
    pub is_temporary: bool,
    pub updated_at: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DraftStoreV2 {
    pub version: u8,
    pub order: Vec<String>,
    pub entries: BTreeMap<String, DraftIndexEntryV2>,
    pub payloads: BTreeMap<String, String>,
    pub session_client_id: Option<String>,
    pub session_open_paths: Vec<String>,
    pub session_active_index: usize,
    pub retained_v1: DraftStoreV1,
}

pub fn migrate_draft_store_v1_to_v2(store: &DraftStoreV1, client_id: Option<&str>) -> DraftStoreV2 {
    const MAX_DRAFTS: usize = 32;
    let mut order = Vec::new();
    let mut entries = BTreeMap::new();
    let mut payloads = BTreeMap::new();
    for path in &store.order {
        let Some(draft) = store.drafts.get(path) else {
            continue;
        };
        let key = stable_path_key(path);
        entries.insert(
            key.clone(),
            DraftIndexEntryV2 {
                path: path.clone(),
                key: key.clone(),
                name: draft.name.clone(),
                is_temporary: draft.is_temporary,
                updated_at: draft.updated_at,
            },
        );
        order.retain(|candidate| candidate != &key);
        order.push(key.clone());
        while order.len() > MAX_DRAFTS {
            let evicted = order.remove(0);
            entries.remove(&evicted);
        }
        payloads.insert(key, draft.data.clone());
    }
    let session_client_id = client_id
        .filter(|client_id| !client_id.is_empty())
        .map(ToOwned::to_owned);
    let (session_open_paths, session_active_index) =
        if session_client_id.is_some() && !store.open_paths.is_empty() {
            (
                store.open_paths.clone(),
                store
                    .active_index
                    .min(store.open_paths.len().saturating_sub(1)),
            )
        } else {
            (Vec::new(), 0)
        };
    DraftStoreV2 {
        version: 2,
        order,
        entries,
        payloads,
        session_client_id,
        session_open_paths,
        session_active_index,
        retained_v1: store.clone(),
    }
}

fn stable_path_key(path: &str) -> String {
    let mut hash = 2_166_136_261u32;
    for code_unit in path.encode_utf16() {
        hash ^= u32::from(code_unit);
        hash = hash.wrapping_mul(16_777_619);
    }
    format!("{hash:08x}")
}

pub fn migrate_node_definition_v1_to_v2(value: &Value) -> Result<Value, WorkflowMigrationError> {
    let object = value
        .as_object()
        .ok_or_else(|| WorkflowMigrationError::Invalid {
            migration: WorkflowMigrationId::UnknownVersion04,
            path: "$".to_owned(),
            reason: "node definition must be an object".to_owned(),
        })?;
    let mut output = object.clone();
    let mut inputs = Map::new();
    if let Some(input) = object.get("input").and_then(Value::as_object) {
        for (section, optional) in [("required", false), ("optional", true)] {
            if let Some(section) = input.get(section).and_then(Value::as_object) {
                for (name, specification) in section {
                    inputs.insert(
                        name.clone(),
                        migrate_input_specification(name, specification, optional),
                    );
                }
            }
        }
    }
    let mut outputs = Vec::new();
    if let Some(types) = object.get("output").and_then(Value::as_array) {
        let names = object.get("output_name").and_then(Value::as_array);
        let lists = object.get("output_is_list").and_then(Value::as_array);
        let tooltips = object.get("output_tooltips").and_then(Value::as_array);
        for (index, type_name) in types.iter().enumerate() {
            let combo = type_name.as_array();
            let mut descriptor = Map::from_iter([
                ("index".to_owned(), Value::Number(Number::from(index))),
                (
                    "name".to_owned(),
                    names
                        .and_then(|names| names.get(index))
                        .and_then(Value::as_str)
                        .map(|name| Value::String(name.to_owned()))
                        .unwrap_or_else(|| Value::String(format!("output_{index}"))),
                ),
                (
                    "type".to_owned(),
                    combo
                        .map(|_| Value::String("COMBO".to_owned()))
                        .unwrap_or_else(|| type_name.clone()),
                ),
                (
                    "is_list".to_owned(),
                    lists
                        .and_then(|lists| lists.get(index))
                        .and_then(Value::as_bool)
                        .map(Value::Bool)
                        .unwrap_or(Value::Bool(false)),
                ),
            ]);
            if let Some(options) = combo {
                descriptor.insert("options".to_owned(), Value::Array(options.clone()));
            }
            if let Some(tooltip) = tooltips.and_then(|tooltips| tooltips.get(index)) {
                descriptor.insert("tooltip".to_owned(), tooltip.clone());
            }
            outputs.push(Value::Object(descriptor));
        }
    }
    output.remove("input");
    output.remove("output");
    output.remove("output_name");
    output.remove("output_is_list");
    output.remove("output_tooltips");
    output.insert("inputs".to_owned(), Value::Object(inputs));
    output.insert("outputs".to_owned(), Value::Array(outputs));
    if let Some(hidden) = object
        .get("input")
        .and_then(Value::as_object)
        .and_then(|input| input.get("hidden"))
    {
        output.insert("hidden".to_owned(), hidden.clone());
    }
    Ok(Value::Object(output))
}

fn migrate_input_specification(name: &str, value: &Value, optional: bool) -> Value {
    let Some(array) = value.as_array() else {
        return json!({"name":name,"isOptional":optional,"type":"UNKNOWN"});
    };
    let type_name = if array.first().is_some_and(Value::is_array) {
        "COMBO".to_owned()
    } else {
        array
            .first()
            .and_then(Value::as_str)
            .unwrap_or("UNKNOWN")
            .to_owned()
    };
    let mut output = Map::from_iter([
        ("type".to_owned(), Value::String(type_name)),
        ("name".to_owned(), Value::String(name.to_owned())),
        ("isOptional".to_owned(), Value::Bool(optional)),
    ]);
    if let Some(options) = array.get(1).and_then(Value::as_object) {
        output.extend(options.clone());
    }
    if let Some(combo) = array.first().and_then(Value::as_array) {
        output.insert("options".to_owned(), Value::Array(combo.clone()));
    }
    Value::Object(output)
}

pub fn migrate_lod_setting(settings: &mut BTreeMap<String, Value>) -> bool {
    const OLD: &str = "LiteGraph.Canvas.LowQualityRenderingZoomThreshold";
    const NEW: &str = "LiteGraph.Canvas.MinFontSizeForLOD";
    if settings.contains_key(NEW) {
        return false;
    }
    let Some(value) = settings.get(OLD).and_then(Value::as_f64) else {
        return false;
    };
    let font_size = (14.0 * value).round().clamp(1.0, 24.0);
    let Some(number) = Number::from_f64(font_size) else {
        return false;
    };
    settings.insert(NEW.to_owned(), Value::Number(number));
    settings.remove(OLD);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkflowFormatDocument;

    #[test]
    fn migration_catalog_has_every_normative_row() {
        assert_eq!(FORMAT_MIGRATION_CATALOG.len(), 24);
        assert_eq!(
            FORMAT_MIGRATION_CATALOG.first(),
            Some(&("COMFY-WORKFLOW-022", "workflow-json-0.4"))
        );
        assert_eq!(
            FORMAT_MIGRATION_CATALOG.last(),
            Some(&("COMFY-WORKFLOW-039", "unknown-version-0.4"))
        );
    }

    #[test]
    fn accepted_sampler_migration_preserves_unknown_values() {
        let source = br#"{"version":0.4,"nodes":[{"id":"01","type":"KSampler","widgets_values":[4,true,20,7,"sample_euler","normal",1],"future":{"keep":1}}],"links":[],"future_top":true}"#;
        let document = WorkflowFormatDocument::parse(source).expect("workflow");
        let migrated =
            apply_workflow_migrations(&document, &[WorkflowMigrationId::KSamplerWidgetValues])
                .expect("accepted migration");
        assert_eq!(document.serialized_bytes().expect("source bytes"), source);
        assert_eq!(
            migrated.value()["nodes"][0]["widgets_values"][1],
            "randomize"
        );
        assert_eq!(migrated.value()["nodes"][0]["widgets_values"][4], "euler");
        assert_eq!(migrated.value()["nodes"][0]["future"]["keep"], 1);
        assert_eq!(migrated.value()["future_top"], true);
        assert_eq!(
            migrated.accepted_migrations()[0].identifier,
            "COMFY-WORKFLOW-038"
        );
    }

    #[test]
    fn primitive_sampler_migration_does_not_rewrite_boolean_primary_values() {
        let boolean_source = br#"{"version":0.4,"nodes":[{"id":1,"type":"PrimitiveNode","widgets_values":[true]}],"links":[]}"#;
        let boolean = WorkflowFormatDocument::parse(boolean_source).expect("boolean primitive");
        assert!(
            detect_workflow_migrations(&boolean)
                .iter()
                .all(|candidate| candidate.identifier != WorkflowMigrationId::KSamplerWidgetValues)
        );

        let control_source = br#"{"version":0.4,"nodes":[{"id":1,"type":"PrimitiveNode","widgets_values":[7,true]}],"links":[]}"#;
        let control = WorkflowFormatDocument::parse(control_source).expect("controlled primitive");
        let migrated =
            apply_workflow_migrations(&control, &[WorkflowMigrationId::KSamplerWidgetValues])
                .expect("controlled primitive migration");
        assert_eq!(
            migrated.value()["nodes"][0]["widgets_values"],
            json!([7, "randomize"])
        );
    }

    #[test]
    fn rejected_migration_is_atomic_and_keeps_original_bytes() {
        let source = br#"{"version":0.4,"nodes":[],"links":[]}"#;
        let document = WorkflowFormatDocument::parse(source).expect("workflow");
        assert_eq!(
            apply_workflow_migrations(&document, &[WorkflowMigrationId::KSamplerWidgetValues]),
            Err(WorkflowMigrationError::NotApplicable(
                WorkflowMigrationId::KSamplerWidgetValues
            ))
        );
        assert_eq!(document.original_bytes(), source);
        assert!(!document.is_modified());
    }

    #[test]
    fn proxy_widget_data_moves_to_lossless_quarantine() {
        let source = br#"{"version":0.4,"nodes":[{"id":1,"type":"Subgraph","properties":{"proxyWidgets":[[2,"seed"]]},"widgets_values":[9]}],"links":[]}"#;
        let document = WorkflowFormatDocument::parse(source).expect("workflow");
        let migrated = apply_workflow_migrations(&document, &[WorkflowMigrationId::ProxyWidget])
            .expect("proxy migration");
        let properties = &migrated.value()["nodes"][0]["properties"];
        assert!(properties.get("proxyWidgets").is_none());
        assert_eq!(
            properties["proxyWidgetErrorQuarantine"]["entries"][0]["source"],
            json!([2, "seed"])
        );
        assert_eq!(migrated.value()["nodes"][0]["widgets_values"][0], 9);
    }

    #[test]
    fn legacy_reroute_migration_matches_all_source_fixtures() {
        let fixtures = [
            (
                "single_connected",
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/legacy/single_connected.json"
                ),
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/native/single_connected.json"
                ),
            ),
            (
                "branching",
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/legacy/branching.json"
                ),
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/native/branching.json"
                ),
            ),
            (
                "floating",
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/legacy/floating.json"
                ),
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/native/floating.json"
                ),
            ),
            (
                "floating_branch",
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/legacy/floating_branch.json"
                ),
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/native/floating_branch.json"
                ),
            ),
        ];
        for (name, legacy, native) in fixtures {
            let document = WorkflowFormatDocument::parse(legacy).expect("legacy source fixture");
            let migrated =
                apply_workflow_migrations(&document, &[WorkflowMigrationId::LegacyRerouteNative])
                    .unwrap_or_else(|error| panic!("source fixture {name}: {error:?}"));
            let expected: Value = serde_json::from_str(native).expect("native source fixture");
            assert_eq!(migrated.value(), &expected, "source fixture {name}");
            assert_eq!(document.original_bytes(), legacy.as_bytes());
        }
    }

    #[test]
    fn malformed_reroute_rejection_is_atomic() {
        let source = br#"{"version":0.4,"nodes":[{"id":1,"type":"Reroute","size":[10,10],"inputs":[],"outputs":[]}],"links":[],"future":true}"#;
        let document = WorkflowFormatDocument::parse(source).expect("workflow");
        let result =
            apply_workflow_migrations(&document, &[WorkflowMigrationId::LegacyRerouteNative]);
        assert!(matches!(
            result,
            Err(WorkflowMigrationError::Invalid {
                migration: WorkflowMigrationId::LegacyRerouteNative,
                ..
            })
        ));
        assert_eq!(document.serialized_bytes().expect("exact source"), source);
        assert_eq!(document.value()["future"], true);
    }

    #[test]
    fn native_reroutes_suppress_the_destructive_legacy_candidate() {
        let source = br#"{"version":0.4,"nodes":[{"id":1,"type":"Reroute"}],"links":[],"extra":{"reroutes":[{"id":1,"pos":[0,0]}]}}"#;
        let document = WorkflowFormatDocument::parse(source).expect("mixed reroute workflow");
        assert!(
            detect_workflow_migrations(&document)
                .iter()
                .all(|candidate| candidate.identifier != WorkflowMigrationId::LegacyRerouteNative)
        );
        assert_eq!(document.serialized_bytes().expect("source bytes"), source);
    }

    #[test]
    fn renderer_and_unknown_numeric_version_migrations_are_explicit() {
        let source = br#"{"version":9,"nodes":[{"id":1,"type":"A","pos":[100,100],"size":[120,60]},{"id":2,"type":"B","pos":[220,220],"size":[240,120]}],"links":[],"groups":[{"pos":[340,340],"size":[120,120]}],"extra":{"workflowRendererVersion":"Vue","reroutes":[{"id":1,"pos":[460,460]}]},"future":"keep"}"#;
        let document = WorkflowFormatDocument::parse(source).expect("workflow");
        let candidates = detect_workflow_migrations(&document)
            .into_iter()
            .map(|candidate| candidate.identifier)
            .collect::<BTreeSet<_>>();
        assert!(candidates.contains(&WorkflowMigrationId::RendererLayoutScale));
        assert!(candidates.contains(&WorkflowMigrationId::UnknownVersion04));
        let migrated = apply_workflow_migrations(
            &document,
            &[
                WorkflowMigrationId::RendererLayoutScale,
                WorkflowMigrationId::UnknownVersion04,
            ],
        )
        .expect("accepted migrations");
        assert_eq!(migrated.value()["nodes"][0]["pos"], json!([100, 100]));
        assert_eq!(migrated.value()["nodes"][0]["size"], json!([100, 50]));
        assert_eq!(migrated.value()["nodes"][1]["pos"], json!([200, 200]));
        assert_eq!(migrated.value()["groups"][0]["pos"], json!([300, 300]));
        assert_eq!(
            migrated.value()["extra"]["reroutes"][0]["pos"],
            json!([400, 400])
        );
        assert_eq!(migrated.value()["future"], "keep");
        assert_eq!(migrated.value()["version"], 9);
        assert_eq!(migrated.accepted_migrations().len(), 2);
        assert_eq!(document.serialized_bytes().expect("exact source"), source);
    }

    #[test]
    fn renderer_migration_scales_serialized_bounds_object_vectors_and_schema_one_reroutes() {
        let source = br#"{
            "version":1,
            "state":{"lastGroupId":0,"lastNodeId":0,"lastLinkId":0,"lastRerouteId":0},
            "nodes":[{"id":1,"type":"A","pos":{"0":100,"1":100},"size":[120,60],"flags":{},"order":0,"mode":0,"properties":{}}],
            "groups":[{"title":"G","bounding":[220,220,120,60]}],
            "reroutes":[{"id":1,"pos":{"0":340,"1":340}}],
            "links":[],"extra":{"workflowRendererVersion":"Vue"},
            "definitions":{"subgraphs":[{
                "id":"00000000-0000-0000-0000-000000000001","revision":1,"version":1,
                "state":{"lastGroupId":0,"lastNodeId":0,"lastLinkId":0,"lastRerouteId":0},
                "name":"Nested","inputNode":{"id":-1,"bounding":[100,100,120,60]},
                "outputNode":{"id":-2,"bounding":[340,100,120,60]},
                "nodes":[{"id":2,"type":"B","pos":[100,100],"size":[120,60],"flags":{},"order":0,"mode":0,"properties":{}}],
                "links":[],"extra":{"workflowRendererVersion":"Vue"}
            }]}
        }"#;
        let document = WorkflowFormatDocument::parse(source).expect("renderer workflow");
        let migrated =
            apply_workflow_migrations(&document, &[WorkflowMigrationId::RendererLayoutScale])
                .expect("renderer migration");
        assert_eq!(migrated.value()["nodes"][0]["pos"], json!([100, 100]));
        assert_eq!(migrated.value()["nodes"][0]["size"], json!([100, 50]));
        assert_eq!(
            migrated.value()["groups"][0]["bounding"],
            json!([200, 200, 100, 50])
        );
        assert_eq!(migrated.value()["reroutes"][0]["pos"], json!([300, 300]));
        let nested = &migrated.value()["definitions"]["subgraphs"][0];
        assert_eq!(nested["inputNode"]["bounding"], json!([100, 100, 100, 50]));
        assert_eq!(nested["outputNode"]["bounding"], json!([300, 100, 100, 50]));
        assert_eq!(nested["extra"]["workflowRendererVersion"], "Vue-corrected");
    }

    #[test]
    fn draft_migration_retains_v1_and_clamps_active_tab() {
        let v1 = DraftStoreV1 {
            drafts: BTreeMap::from([(
                "a.json".to_owned(),
                DraftV1 {
                    data: "{}".to_owned(),
                    updated_at: 4,
                    name: "A".to_owned(),
                    is_temporary: false,
                },
            )]),
            order: vec!["missing.json".to_owned(), "a.json".to_owned()],
            open_paths: vec!["a.json".to_owned()],
            active_index: 9,
            unknown: BTreeMap::from([("future".to_owned(), Value::Bool(true))]),
        };
        let v2 = migrate_draft_store_v1_to_v2(&v1, Some("client"));
        assert_eq!(v2.entries.len(), 1);
        assert_eq!(v2.session_active_index, 0);
        assert_eq!(v2.session_client_id.as_deref(), Some("client"));
        assert_eq!(v2.order, [stable_path_key("a.json")]);
        assert_eq!(stable_path_key(""), "811c9dc5");
        assert_eq!(v2.retained_v1, v1);

        let paths = (0..40)
            .map(|index| format!("{index}.json"))
            .collect::<Vec<_>>();
        let bounded = DraftStoreV1 {
            drafts: paths
                .iter()
                .enumerate()
                .map(|(index, path)| {
                    (
                        path.clone(),
                        DraftV1 {
                            data: format!("{{\"index\":{index}}}"),
                            updated_at: index as u64,
                            name: path.clone(),
                            is_temporary: true,
                        },
                    )
                })
                .collect(),
            order: paths,
            open_paths: vec!["0.json".to_owned()],
            active_index: 0,
            unknown: BTreeMap::new(),
        };
        let bounded_v2 = migrate_draft_store_v1_to_v2(&bounded, None);
        assert_eq!(bounded_v2.entries.len(), 32);
        assert_eq!(bounded_v2.order.len(), 32);
        assert_eq!(bounded_v2.payloads.len(), 40);
        assert!(bounded_v2.session_open_paths.is_empty());
        assert!(bounded_v2.session_client_id.is_none());
    }

    #[test]
    fn node_definition_and_lod_migrations_are_typed() {
        let v1 = json!({
            "name":"Example",
            "input":{"required":{"seed":["INT",{"default":1}]},"optional":{"mode":[["a","b"],{}]},"hidden":{"token":"SECRET"}},
            "output":["IMAGE",["x","y"]],
            "output_name":["image","choice"],
            "future":true
        });
        let v2 = migrate_node_definition_v1_to_v2(&v1).expect("node definition");
        assert_eq!(v2["inputs"]["seed"]["type"], "INT");
        assert_eq!(v2["inputs"]["mode"]["type"], "COMBO");
        assert_eq!(v2["outputs"][1]["options"], json!(["x", "y"]));
        assert_eq!(v2["hidden"]["token"], "SECRET");
        assert_eq!(v2["future"], true);

        let mut settings = BTreeMap::from([(
            "LiteGraph.Canvas.LowQualityRenderingZoomThreshold".to_owned(),
            json!(0.5),
        )]);
        assert!(migrate_lod_setting(&mut settings));
        assert_eq!(settings["LiteGraph.Canvas.MinFontSizeForLOD"], 7.0);
        assert!(!settings.contains_key("LiteGraph.Canvas.LowQualityRenderingZoomThreshold"));

        let mut clamped = BTreeMap::from([(
            "LiteGraph.Canvas.LowQualityRenderingZoomThreshold".to_owned(),
            json!(99),
        )]);
        assert!(migrate_lod_setting(&mut clamped));
        assert_eq!(clamped["LiteGraph.Canvas.MinFontSizeForLOD"], 24.0);
    }
}
