use crate::graph::NATIVE_WIDGETS_FIELD;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};
use sim_media::{ComfyMetadata, MetadataDocument};
use sim_model::ArtifactKey;
use sim_nodes::NodeDescriptor;
use sim_types::{
    ApiPrompt, NodeId, NonFiniteJsonToken, PromptNode, PromptSubmission, normalize_json_non_finite,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;
use uuid::Uuid;

pub const MAX_WORKFLOW_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_WORKFLOW_NODES: usize = 100_000;
const MAX_SAVE_JOURNAL_BYTES: usize = MAX_WORKFLOW_BYTES * 6;
const MAX_JSON_DEPTH: usize = 256;
const MAX_MIGRATIONS: usize = 1_024;
const MAX_WORKFLOW_IDENTITY_BYTES: usize = 16 * 1024;
const MAX_PROVIDER_IDENTIFIER_BYTES: usize = 4 * 1024;
const CONTROL_VALUE_LIMIT: i64 = 1_125_899_906_842_624;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkflowFormat {
    Schema04,
    Schema1,
    OtherNumeric,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkflowValidationIssue {
    pub path: String,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcceptedMigration {
    pub identifier: String,
    pub version: u32,
    pub provenance: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PluginMappingProjection {
    pub legacy_identifier: String,
    pub resolved_identifier: String,
    pub plugin_identifier: String,
    pub mapping_version: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkflowFormatDocument {
    original_bytes: Arc<[u8]>,
    original_value: Value,
    value: Value,
    format: Option<WorkflowFormat>,
    non_finite_tokens: Vec<NonFiniteJsonToken>,
    validation_issues: Vec<WorkflowValidationIssue>,
    accepted_migrations: Vec<AcceptedMigration>,
    plugin_projections: Vec<PluginMappingProjection>,
    modified: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApiPromptDocument {
    original_bytes: Arc<[u8]>,
    value: Value,
    non_finite_tokens: Vec<NonFiniteJsonToken>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TemplateDocument {
    original_bytes: Arc<[u8]>,
    root: Value,
    templates: Value,
    non_finite_tokens: Vec<NonFiniteJsonToken>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum JsonImport {
    Templates(TemplateDocument),
    ApiPrompt(ApiPromptDocument),
    Workflow(WorkflowFormatDocument),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkflowFormatError {
    #[error("workflow input is {actual} bytes, exceeding the {limit}-byte limit")]
    InputTooLarge { actual: usize, limit: usize },
    #[error("workflow JSON nesting exceeds {MAX_JSON_DEPTH}")]
    TooDeep,
    #[error("workflow JSON is invalid at {path}: {reason}")]
    InvalidJson { path: String, reason: String },
    #[error("workflow root must be an object")]
    RootNotObject,
    #[error("workflow version is missing or is not numeric")]
    InvalidVersion,
    #[error("API prompt is invalid at {path}: {reason}")]
    InvalidPrompt { path: String, reason: String },
    #[error("graph-to-prompt conversion failed at {path}: {reason}")]
    PromptProjection { path: String, reason: String },
    #[error("embedded content has no importable representation")]
    NoImportableRepresentation,
    #[error("A1111 parameters are invalid: {0}")]
    InvalidA1111(String),
    #[error("file locator is invalid: {0}")]
    InvalidLocator(String),
    #[error("model path is invalid: {0}")]
    InvalidModelPath(String),
    #[error("serialization failed: {0}")]
    Serialization(String),
    #[error("too many accepted migrations")]
    TooManyMigrations,
}

impl WorkflowFormatDocument {
    pub fn parse(bytes: impl AsRef<[u8]>) -> Result<Self, WorkflowFormatError> {
        let bytes = bytes.as_ref();
        let (value, non_finite_tokens) = parse_lossless_json(bytes)?;
        Self::from_parts(bytes, value, non_finite_tokens)
    }

    fn from_parts(
        bytes: &[u8],
        value: Value,
        non_finite_tokens: Vec<NonFiniteJsonToken>,
    ) -> Result<Self, WorkflowFormatError> {
        let object = value
            .as_object()
            .ok_or(WorkflowFormatError::RootNotObject)?;
        let format = object
            .get("version")
            .and_then(Value::as_number)
            .map(|version| {
                if version.as_u64() == Some(1) {
                    WorkflowFormat::Schema1
                } else if version.as_f64() == Some(0.4) {
                    WorkflowFormat::Schema04
                } else {
                    WorkflowFormat::OtherNumeric
                }
            });
        let validation_issues = validate_workflow(&value, format);
        Ok(Self {
            original_bytes: Arc::from(bytes),
            original_value: value.clone(),
            value,
            format,
            non_finite_tokens,
            validation_issues,
            accepted_migrations: Vec::new(),
            plugin_projections: Vec::new(),
            modified: false,
        })
    }

    pub fn original_bytes(&self) -> &[u8] {
        &self.original_bytes
    }

    pub fn original_value(&self) -> &Value {
        &self.original_value
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn format(&self) -> Option<WorkflowFormat> {
        self.format
    }

    pub fn non_finite_tokens(&self) -> &[NonFiniteJsonToken] {
        &self.non_finite_tokens
    }

    pub fn validation_issues(&self) -> &[WorkflowValidationIssue] {
        &self.validation_issues
    }

    pub fn accepted_migrations(&self) -> &[AcceptedMigration] {
        &self.accepted_migrations
    }

    pub fn plugin_projections(&self) -> &[PluginMappingProjection] {
        &self.plugin_projections
    }

    pub fn is_modified(&self) -> bool {
        self.modified
    }

    pub fn serialized_bytes(&self) -> Result<Vec<u8>, WorkflowFormatError> {
        if !self.modified {
            return Ok(self.original_bytes.to_vec());
        }
        serde_json::to_vec(&self.value)
            .map_err(|error| WorkflowFormatError::Serialization(error.to_string()))
    }

    pub fn with_plugin_projection(mut self, projection: PluginMappingProjection) -> Self {
        self.plugin_projections.push(projection);
        self
    }

    pub(crate) fn with_accepted_migration(
        &self,
        value: Value,
        migration: AcceptedMigration,
    ) -> Result<Self, WorkflowFormatError> {
        if self.accepted_migrations.len() >= MAX_MIGRATIONS {
            return Err(WorkflowFormatError::TooManyMigrations);
        }
        let object = value
            .as_object()
            .ok_or(WorkflowFormatError::RootNotObject)?;
        let version = object
            .get("version")
            .and_then(Value::as_number)
            .ok_or(WorkflowFormatError::InvalidVersion)?;
        let format = if version.as_u64() == Some(1) {
            WorkflowFormat::Schema1
        } else if version.as_f64() == Some(0.4) {
            WorkflowFormat::Schema04
        } else {
            WorkflowFormat::OtherNumeric
        };
        let mut accepted_migrations = self.accepted_migrations.clone();
        accepted_migrations.push(migration);
        Ok(Self {
            original_bytes: self.original_bytes.clone(),
            original_value: self.original_value.clone(),
            validation_issues: validate_workflow(&value, Some(format)),
            value,
            format: Some(format),
            non_finite_tokens: self.non_finite_tokens.clone(),
            accepted_migrations,
            plugin_projections: self.plugin_projections.clone(),
            modified: true,
        })
    }
}

impl ApiPromptDocument {
    pub fn original_bytes(&self) -> &[u8] {
        &self.original_bytes
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn non_finite_tokens(&self) -> &[NonFiniteJsonToken] {
        &self.non_finite_tokens
    }

    pub fn to_submission(&self) -> Result<PromptSubmission, WorkflowFormatError> {
        let prompt: ApiPrompt = serde_json::from_value(self.value.clone()).map_err(|error| {
            WorkflowFormatError::InvalidPrompt {
                path: "$".to_owned(),
                reason: error.to_string(),
            }
        })?;
        Ok(PromptSubmission {
            prompt,
            prompt_id: None,
            client_id: None,
            number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
        })
    }
}

impl TemplateDocument {
    pub fn original_bytes(&self) -> &[u8] {
        &self.original_bytes
    }

    pub fn root(&self) -> &Value {
        &self.root
    }

    pub fn templates(&self) -> &Value {
        &self.templates
    }

    pub fn non_finite_tokens(&self) -> &[NonFiniteJsonToken] {
        &self.non_finite_tokens
    }
}

pub fn import_json(bytes: impl AsRef<[u8]>) -> Result<JsonImport, WorkflowFormatError> {
    let bytes = bytes.as_ref();
    let (value, non_finite_tokens) = parse_lossless_json(bytes)?;
    let object = value
        .as_object()
        .ok_or(WorkflowFormatError::RootNotObject)?;
    if let Some(templates) = object.get("templates").filter(|value| !value.is_null()) {
        let templates = templates.clone();
        return Ok(JsonImport::Templates(TemplateDocument {
            original_bytes: Arc::from(bytes),
            root: value,
            templates,
            non_finite_tokens,
        }));
    }
    if is_api_prompt_value(&value) {
        return Ok(JsonImport::ApiPrompt(ApiPromptDocument {
            original_bytes: Arc::from(bytes),
            value,
            non_finite_tokens,
        }));
    }
    Ok(JsonImport::Workflow(WorkflowFormatDocument::from_parts(
        bytes,
        value,
        non_finite_tokens,
    )?))
}

fn parse_lossless_json(
    bytes: &[u8],
) -> Result<(Value, Vec<NonFiniteJsonToken>), WorkflowFormatError> {
    if bytes.len() > MAX_WORKFLOW_BYTES {
        return Err(WorkflowFormatError::InputTooLarge {
            actual: bytes.len(),
            limit: MAX_WORKFLOW_BYTES,
        });
    }
    validate_json_depth(bytes)?;
    let (normalized, tokens) = normalize_json_non_finite(bytes);
    let mut deserializer = serde_json::Deserializer::from_slice(&normalized);
    let value = serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
        WorkflowFormatError::InvalidJson {
            path: error.path().to_string(),
            reason: error.inner().to_string(),
        }
    })?;
    Ok((value, tokens))
}

fn validate_json_depth(bytes: &[u8]) -> Result<(), WorkflowFormatError> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match *byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth = depth.checked_add(1).ok_or(WorkflowFormatError::TooDeep)?;
                if depth > MAX_JSON_DEPTH {
                    return Err(WorkflowFormatError::TooDeep);
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

fn is_api_prompt_value(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    !object.is_empty()
        && object.values().all(|node| {
            node.as_object().is_some_and(|node| {
                node.get("class_type").and_then(Value::as_str).is_some()
                    && node.get("inputs").and_then(Value::as_object).is_some()
            })
        })
}

fn validate_workflow(
    value: &Value,
    format: Option<WorkflowFormat>,
) -> Vec<WorkflowValidationIssue> {
    let mut issues = Vec::new();
    let Some(object) = value.as_object() else {
        issues.push(issue("$", "root is not an object"));
        return issues;
    };
    match format {
        Some(WorkflowFormat::Schema1) => {
            validate_schema_one_graph(object, "$", false, &mut issues);
        }
        Some(WorkflowFormat::Schema04 | WorkflowFormat::OtherNumeric) => {
            validate_schema_zero_four_graph(object, "$", &mut issues);
        }
        None => issues.push(issue(
            "$.version",
            "workflow version is missing or is not numeric",
        )),
    }
    if object
        .get("nodes")
        .and_then(Value::as_array)
        .is_some_and(|nodes| nodes.len() > MAX_WORKFLOW_NODES)
    {
        issues.push(issue("$.nodes", "node count exceeds the configured limit"));
    }
    issues
}

fn validate_schema_zero_four_graph(
    object: &Map<String, Value>,
    path: &str,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    if object
        .get("last_node_id")
        .and_then(GraphNodeIdentifier::parse)
        .is_none()
    {
        issues.push(issue(
            format!("{path}.last_node_id"),
            "schema 0.4 last node ID must be an integer or string",
        ));
    }
    if !object.get("last_link_id").is_some_and(Value::is_number) {
        issues.push(issue(
            format!("{path}.last_link_id"),
            "schema 0.4 last link ID must be numeric",
        ));
    }
    validate_optional_graph_identity_and_revision(object, path, issues);
    match object.get("nodes").and_then(Value::as_array) {
        Some(nodes) => {
            for (index, node) in nodes.iter().enumerate() {
                validate_schema_one_node(node, &format!("{path}.nodes[{index}]"), issues);
            }
        }
        None => issues.push(issue(
            format!("{path}.nodes"),
            "schema 0.4 nodes must be an array",
        )),
    }
    match object.get("links").and_then(Value::as_array) {
        Some(links) => {
            for (index, link) in links.iter().enumerate() {
                validate_schema_zero_four_link(link, &format!("{path}.links[{index}]"), issues);
            }
        }
        None => issues.push(issue(
            format!("{path}.links"),
            "schema 0.4 links must be an array",
        )),
    }
    if let Some(floating_links) = object.get("floatingLinks") {
        match floating_links.as_array() {
            Some(floating_links) => {
                for (index, link) in floating_links.iter().enumerate() {
                    validate_schema_one_link(
                        link,
                        &format!("{path}.floatingLinks[{index}]"),
                        issues,
                    );
                }
            }
            None => issues.push(issue(
                format!("{path}.floatingLinks"),
                "schema 0.4 floating links must be an array",
            )),
        }
    }
    validate_schema_one_groups(object.get("groups"), path, issues);
    validate_optional_graph_fields(object, path, issues);
    validate_schema_one_instances(object.get("subgraphs"), path, issues);
    validate_definitions(object.get("definitions"), path, issues);
}

fn validate_schema_zero_four_link(
    value: &Value,
    path: &str,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    let Some(link) = value.as_array() else {
        issues.push(issue(path, "schema 0.4 link must be an array"));
        return;
    };
    if link.len() != 6 {
        issues.push(issue(path, "schema 0.4 link must have six entries"));
        return;
    }
    if !link[0].is_number() {
        issues.push(issue(
            format!("{path}[0]"),
            "schema 0.4 link ID must be numeric",
        ));
    }
    for (index, label) in [(1, "origin"), (3, "target")] {
        if GraphNodeIdentifier::parse(&link[index]).is_none() {
            issues.push(issue(
                format!("{path}[{index}]"),
                format!("schema 0.4 {label} node ID is invalid"),
            ));
        }
    }
    for (index, label) in [(2, "origin"), (4, "target")] {
        if !valid_slot_index(&link[index]) {
            issues.push(issue(
                format!("{path}[{index}]"),
                format!("schema 0.4 {label} slot is invalid"),
            ));
        }
    }
    if !valid_data_type(&link[5]) {
        issues.push(issue(
            format!("{path}[5]"),
            "schema 0.4 link type is invalid",
        ));
    }
}

fn validate_schema_one_graph(
    object: &Map<String, Value>,
    path: &str,
    definition: bool,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    if object.get("version").and_then(Value::as_u64) != Some(1) {
        issues.push(issue(
            format!("{path}.version"),
            "schema 1 version must equal 1",
        ));
    }
    if let Some(identifier) = object.get("id") {
        if !identifier
            .as_str()
            .is_some_and(|identifier| Uuid::parse_str(identifier).is_ok())
        {
            issues.push(issue(
                format!("{path}.id"),
                "schema 1 graph id must be a UUID",
            ));
        }
    } else if definition {
        issues.push(issue(
            format!("{path}.id"),
            "schema 1 subgraph id is required",
        ));
    }
    if let Some(revision) = object.get("revision") {
        if !revision.is_number() {
            issues.push(issue(
                format!("{path}.revision"),
                "schema 1 revision must be numeric",
            ));
        }
    } else if definition {
        issues.push(issue(
            format!("{path}.revision"),
            "schema 1 subgraph revision is required",
        ));
    }
    validate_optional_graph_fields(object, path, issues);
    match object.get("state").and_then(Value::as_object) {
        Some(state) => {
            for field in ["lastGroupId", "lastNodeId", "lastLinkId", "lastRerouteId"] {
                if !state.get(field).is_some_and(Value::is_number) {
                    issues.push(issue(
                        format!("{path}.state.{field}"),
                        "schema 1 graph counter is required and must be numeric",
                    ));
                }
            }
        }
        None => issues.push(issue(
            format!("{path}.state"),
            "schema 1 state must be an object",
        )),
    }
    match object.get("nodes").and_then(Value::as_array) {
        Some(nodes) => {
            for (index, node) in nodes.iter().enumerate() {
                validate_schema_one_node(node, &format!("{path}.nodes[{index}]"), issues);
            }
        }
        None => issues.push(issue(
            format!("{path}.nodes"),
            "schema 1 nodes must be an array",
        )),
    }
    validate_schema_one_groups(object.get("groups"), path, issues);
    validate_schema_one_instances(object.get("subgraphs"), path, issues);
    if let Some(links) = object.get("links") {
        match links.as_array() {
            Some(links) => {
                for (index, link) in links.iter().enumerate() {
                    validate_schema_one_link(link, &format!("{path}.links[{index}]"), issues);
                }
            }
            None => issues.push(issue(
                format!("{path}.links"),
                "schema 1 links must be an array",
            )),
        }
    }
    if let Some(floating_links) = object.get("floatingLinks") {
        match floating_links.as_array() {
            Some(floating_links) => {
                for (index, link) in floating_links.iter().enumerate() {
                    validate_schema_one_link(
                        link,
                        &format!("{path}.floatingLinks[{index}]"),
                        issues,
                    );
                }
            }
            None => issues.push(issue(
                format!("{path}.floatingLinks"),
                "schema 1 floating links must be an array",
            )),
        }
    }
    if let Some(reroutes) = object.get("reroutes") {
        match reroutes.as_array() {
            Some(reroutes) => {
                for (index, reroute) in reroutes.iter().enumerate() {
                    let reroute_path = format!("{path}.reroutes[{index}]");
                    let Some(reroute) = reroute.as_object() else {
                        issues.push(issue(reroute_path, "schema 1 reroute must be an object"));
                        continue;
                    };
                    if !reroute.get("id").is_some_and(Value::is_number) {
                        issues.push(issue(
                            format!("{reroute_path}.id"),
                            "schema 1 reroute id must be numeric",
                        ));
                    }
                    if !reroute.get("pos").is_some_and(valid_vector_two) {
                        issues.push(issue(
                            format!("{reroute_path}.pos"),
                            "schema 1 reroute position must contain two numbers",
                        ));
                    }
                    if reroute
                        .get("parentId")
                        .is_some_and(|parent| !parent.is_number())
                    {
                        issues.push(issue(
                            format!("{reroute_path}.parentId"),
                            "schema 1 reroute parent must be numeric",
                        ));
                    }
                    if reroute.get("linkIds").is_some_and(|link_ids| {
                        !link_ids.is_null()
                            && !link_ids
                                .as_array()
                                .is_some_and(|link_ids| link_ids.iter().all(Value::is_number))
                    }) {
                        issues.push(issue(
                            format!("{reroute_path}.linkIds"),
                            "schema 1 reroute link ids must be numeric",
                        ));
                    }
                    if let Some(floating) = reroute.get("floating") {
                        let valid = floating.as_object().is_some_and(|floating| {
                            matches!(
                                floating.get("slotType").and_then(Value::as_str),
                                Some("input" | "output")
                            )
                        });
                        if !valid {
                            issues.push(issue(
                                format!("{reroute_path}.floating.slotType"),
                                "schema 1 floating reroute slot type must be input or output",
                            ));
                        }
                    }
                }
            }
            None => issues.push(issue(
                format!("{path}.reroutes"),
                "schema 1 reroutes must be an array",
            )),
        }
    }
    if definition {
        if !object.get("name").is_some_and(Value::is_string) {
            issues.push(issue(
                format!("{path}.name"),
                "schema 1 subgraph name is required",
            ));
        }
        for field in ["inputNode", "outputNode"] {
            let field_path = format!("{path}.{field}");
            let Some(boundary) = object.get(field).and_then(Value::as_object) else {
                issues.push(issue(
                    field_path,
                    "schema 1 subgraph boundary node is required",
                ));
                continue;
            };
            if boundary
                .get("id")
                .and_then(GraphNodeIdentifier::parse)
                .is_none()
            {
                issues.push(issue(
                    format!("{path}.{field}.id"),
                    "schema 1 subgraph boundary id is invalid",
                ));
            }
            if !boundary.get("bounding").is_some_and(valid_bounding) {
                issues.push(issue(
                    format!("{path}.{field}.bounding"),
                    "schema 1 subgraph boundary must contain four numbers",
                ));
            }
            if boundary
                .get("pinned")
                .is_some_and(|pinned| !pinned.is_boolean())
            {
                issues.push(issue(
                    format!("{path}.{field}.pinned"),
                    "schema 1 subgraph boundary pinned flag must be boolean",
                ));
            }
        }
        for field in ["inputs", "outputs"] {
            if let Some(ports) = object.get(field) {
                match ports.as_array() {
                    Some(ports) => {
                        for (index, port) in ports.iter().enumerate() {
                            validate_schema_one_subgraph_port(
                                port,
                                &format!("{path}.{field}[{index}]"),
                                issues,
                            );
                        }
                    }
                    None => issues.push(issue(
                        format!("{path}.{field}"),
                        "schema 1 subgraph ports must be an array",
                    )),
                }
            }
        }
        if let Some(widgets) = object.get("widgets") {
            match widgets.as_array() {
                Some(widgets) => {
                    for (index, widget) in widgets.iter().enumerate() {
                        let widget_path = format!("{path}.widgets[{index}]");
                        let Some(widget) = widget.as_object() else {
                            issues.push(issue(
                                widget_path,
                                "schema 1 exposed widget must be an object",
                            ));
                            continue;
                        };
                        for field in ["id", "name"] {
                            if !widget.get(field).is_some_and(Value::is_string) {
                                issues.push(issue(
                                    format!("{path}.widgets[{index}].{field}"),
                                    "schema 1 exposed widget field must be a string",
                                ));
                            }
                        }
                    }
                }
                None => issues.push(issue(
                    format!("{path}.widgets"),
                    "schema 1 exposed widgets must be an array",
                )),
            }
        }
    }
    validate_definitions(object.get("definitions"), path, issues);
}

fn validate_optional_graph_fields(
    object: &Map<String, Value>,
    path: &str,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    for field in ["config", "extra"] {
        if object
            .get(field)
            .is_some_and(|value| !value.is_null() && !value.is_object())
        {
            issues.push(issue(
                format!("{path}.{field}"),
                "workflow graph field must be an object or null",
            ));
        }
    }
    if let Some(models) = object.get("models") {
        let Some(models) = models.as_array() else {
            issues.push(issue(
                format!("{path}.models"),
                "workflow models must be an array",
            ));
            return;
        };
        for (index, model) in models.iter().enumerate() {
            validate_model_file(model, &format!("{path}.models[{index}]"), issues);
        }
    }
}

fn validate_optional_graph_identity_and_revision(
    object: &Map<String, Value>,
    path: &str,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    if object.get("id").is_some_and(|identifier| {
        !identifier
            .as_str()
            .is_some_and(|identifier| Uuid::parse_str(identifier).is_ok())
    }) {
        issues.push(issue(
            format!("{path}.id"),
            "workflow graph ID must be a UUID",
        ));
    }
    if object
        .get("revision")
        .is_some_and(|revision| !revision.is_number())
    {
        issues.push(issue(
            format!("{path}.revision"),
            "workflow graph revision must be numeric",
        ));
    }
}

fn validate_model_file(value: &Value, path: &str, issues: &mut Vec<WorkflowValidationIssue>) {
    let Some(model) = value.as_object() else {
        issues.push(issue(path, "workflow model must be an object"));
        return;
    };
    for field in ["name", "directory"] {
        if !model.get(field).is_some_and(Value::is_string) {
            issues.push(issue(
                format!("{path}.{field}"),
                "workflow model field must be a string",
            ));
        }
    }
    if !model
        .get("url")
        .and_then(Value::as_str)
        .is_some_and(|url| url::Url::parse(url).is_ok())
    {
        issues.push(issue(
            format!("{path}.url"),
            "workflow model URL is invalid",
        ));
    }
    for field in ["hash", "hash_type"] {
        if model.get(field).is_some_and(|value| !value.is_string()) {
            issues.push(issue(
                format!("{path}.{field}"),
                "workflow model field must be a string",
            ));
        }
    }
}

fn validate_definitions(
    value: Option<&Value>,
    path: &str,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    let Some(value) = value else {
        return;
    };
    let Some(subgraphs) = value
        .as_object()
        .and_then(|definitions| definitions.get("subgraphs"))
        .and_then(Value::as_array)
    else {
        issues.push(issue(
            format!("{path}.definitions.subgraphs"),
            "schema 1 subgraph definitions must be an array",
        ));
        return;
    };
    for (index, subgraph) in subgraphs.iter().enumerate() {
        let Some(subgraph) = subgraph.as_object() else {
            issues.push(issue(
                format!("{path}.definitions.subgraphs[{index}]"),
                "schema 1 subgraph definition must be an object",
            ));
            continue;
        };
        validate_schema_one_graph(
            subgraph,
            &format!("{path}.definitions.subgraphs[{index}]"),
            true,
            issues,
        );
    }
}

struct GraphNodeIdentifier;

impl GraphNodeIdentifier {
    fn parse(value: &Value) -> Option<()> {
        (value.as_i64().is_some() || value.as_str().is_some()).then_some(())
    }
}

fn validate_schema_one_node(value: &Value, path: &str, issues: &mut Vec<WorkflowValidationIssue>) {
    let Some(node) = value.as_object() else {
        issues.push(issue(path, "schema 1 node must be an object"));
        return;
    };
    validate_schema_one_node_base(node, path, issues);
    validate_schema_one_node_slots(node, path, false, issues);
    if !node.get("properties").is_some_and(Value::is_object) {
        issues.push(issue(
            format!("{path}.properties"),
            "schema 1 node properties must be an object",
        ));
    }
}

fn validate_schema_one_node_base(
    node: &Map<String, Value>,
    path: &str,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    if node
        .get("id")
        .and_then(GraphNodeIdentifier::parse)
        .is_none()
    {
        issues.push(issue(format!("{path}.id"), "schema 1 node id is invalid"));
    }
    for field in ["type"] {
        if !node.get(field).is_some_and(Value::is_string) {
            issues.push(issue(
                format!("{path}.{field}"),
                "schema 1 node field must be a string",
            ));
        }
    }
    for field in ["pos", "size"] {
        if !node.get(field).is_some_and(valid_vector_two) {
            issues.push(issue(
                format!("{path}.{field}"),
                "schema 1 node vector must contain two numbers",
            ));
        }
    }
    if !node.get("flags").is_some_and(Value::is_object) {
        issues.push(issue(
            format!("{path}.flags"),
            "schema 1 node flags must be an object",
        ));
    }
    for field in ["order", "mode"] {
        if !node.get(field).is_some_and(Value::is_number) {
            issues.push(issue(
                format!("{path}.{field}"),
                "schema 1 node field must be numeric",
            ));
        }
    }
    if node
        .get("widgets_values")
        .is_some_and(|widgets| !widgets.is_array() && !widgets.is_object())
    {
        issues.push(issue(
            format!("{path}.widgets_values"),
            "schema 1 widget values must be an array or object",
        ));
    }
}

fn validate_schema_one_node_slots(
    node: &Map<String, Value>,
    path: &str,
    subgraph_ports: bool,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    for field in ["inputs", "outputs"] {
        let Some(slots) = node.get(field) else {
            continue;
        };
        let Some(slots) = slots.as_array() else {
            issues.push(issue(
                format!("{path}.{field}"),
                "schema 1 node slots must be an array",
            ));
            continue;
        };
        for (index, slot) in slots.iter().enumerate() {
            let slot_path = format!("{path}.{field}[{index}]");
            if subgraph_ports {
                validate_schema_one_subgraph_port(slot, &slot_path, issues);
            } else {
                validate_schema_one_node_slot(slot, &slot_path, field == "outputs", issues);
            }
        }
    }
}

fn validate_schema_one_node_slot(
    value: &Value,
    path: &str,
    output: bool,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    let Some(slot) = value.as_object() else {
        issues.push(issue(path, "schema 1 node slot must be an object"));
        return;
    };
    if !slot.get("name").is_some_and(Value::is_string) {
        issues.push(issue(
            format!("{path}.name"),
            "schema 1 node slot name must be a string",
        ));
    }
    if !slot.get("type").is_some_and(valid_data_type) {
        issues.push(issue(
            format!("{path}.type"),
            "schema 1 node slot type is invalid",
        ));
    }
    if slot
        .get("slot_index")
        .is_some_and(|index| !valid_slot_index(index))
    {
        issues.push(issue(
            format!("{path}.slot_index"),
            "schema 1 node slot index is invalid",
        ));
    }
    if output {
        if slot.get("links").is_some_and(|links| {
            !links.is_null()
                && !links
                    .as_array()
                    .is_some_and(|links| links.iter().all(Value::is_number))
        }) {
            issues.push(issue(
                format!("{path}.links"),
                "schema 1 output links must be numeric",
            ));
        }
    } else if slot
        .get("link")
        .is_some_and(|link| !link.is_null() && !link.is_number())
    {
        issues.push(issue(
            format!("{path}.link"),
            "schema 1 input link must be numeric",
        ));
    }
}

fn validate_schema_one_groups(
    value: Option<&Value>,
    path: &str,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    let Some(value) = value else {
        return;
    };
    let Some(groups) = value.as_array() else {
        issues.push(issue(
            format!("{path}.groups"),
            "schema 1 groups must be an array",
        ));
        return;
    };
    for (index, group) in groups.iter().enumerate() {
        let group_path = format!("{path}.groups[{index}]");
        let Some(group) = group.as_object() else {
            issues.push(issue(group_path, "schema 1 group must be an object"));
            continue;
        };
        if group
            .get("id")
            .is_some_and(|identifier| !identifier.is_number())
        {
            issues.push(issue(
                format!("{path}.groups[{index}].id"),
                "schema 1 group id must be numeric",
            ));
        }
        if !group.get("title").is_some_and(Value::is_string) {
            issues.push(issue(
                format!("{path}.groups[{index}].title"),
                "schema 1 group title must be a string",
            ));
        }
        if !group.get("bounding").is_some_and(valid_bounding) {
            issues.push(issue(
                format!("{path}.groups[{index}].bounding"),
                "schema 1 group bounds must contain four numbers",
            ));
        }
        for field in ["color"] {
            if group.get(field).is_some_and(|value| !value.is_string()) {
                issues.push(issue(
                    format!("{path}.groups[{index}].{field}"),
                    "schema 1 group field must be a string",
                ));
            }
        }
        if group
            .get("font_size")
            .is_some_and(|value| !value.is_number())
        {
            issues.push(issue(
                format!("{path}.groups[{index}].font_size"),
                "schema 1 group font size must be numeric",
            ));
        }
        if group.get("locked").is_some_and(|value| !value.is_boolean()) {
            issues.push(issue(
                format!("{path}.groups[{index}].locked"),
                "schema 1 group locked flag must be boolean",
            ));
        }
    }
}

fn validate_schema_one_instances(
    value: Option<&Value>,
    path: &str,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    let Some(value) = value else {
        return;
    };
    let Some(instances) = value.as_array() else {
        issues.push(issue(
            format!("{path}.subgraphs"),
            "schema 1 subgraph instances must be an array",
        ));
        return;
    };
    for (index, instance) in instances.iter().enumerate() {
        let instance_path = format!("{path}.subgraphs[{index}]");
        let Some(instance) = instance.as_object() else {
            issues.push(issue(
                instance_path,
                "schema 1 subgraph instance must be an object",
            ));
            continue;
        };
        validate_schema_one_node_base(instance, &instance_path, issues);
        validate_schema_one_node_slots(instance, &instance_path, true, issues);
        if !instance
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|identifier| Uuid::parse_str(identifier).is_ok())
        {
            issues.push(issue(
                format!("{instance_path}.type"),
                "schema 1 subgraph instance type must be a UUID",
            ));
        }
    }
}

fn validate_schema_one_link(value: &Value, path: &str, issues: &mut Vec<WorkflowValidationIssue>) {
    let Some(link) = value.as_object() else {
        issues.push(issue(path, "schema 1 link must be an object"));
        return;
    };
    if !link.get("id").is_some_and(Value::is_number) {
        issues.push(issue(
            format!("{path}.id"),
            "schema 1 link id must be numeric",
        ));
    }
    for field in ["origin_id", "target_id"] {
        if link
            .get(field)
            .and_then(GraphNodeIdentifier::parse)
            .is_none()
        {
            issues.push(issue(
                format!("{path}.{field}"),
                "schema 1 link node id is invalid",
            ));
        }
    }
    for field in ["origin_slot", "target_slot"] {
        let valid = link.get(field).is_some_and(valid_slot_index);
        if !valid {
            issues.push(issue(
                format!("{path}.{field}"),
                "schema 1 link slot is invalid",
            ));
        }
    }
    if !link.get("type").is_some_and(valid_data_type) {
        issues.push(issue(
            format!("{path}.type"),
            "schema 1 link type is invalid",
        ));
    }
    if link
        .get("parentId")
        .is_some_and(|parent| !parent.is_number())
    {
        issues.push(issue(
            format!("{path}.parentId"),
            "schema 1 link parent reroute must be numeric",
        ));
    }
}

fn validate_schema_one_subgraph_port(
    value: &Value,
    path: &str,
    issues: &mut Vec<WorkflowValidationIssue>,
) {
    let Some(port) = value.as_object() else {
        issues.push(issue(path, "schema 1 subgraph port must be an object"));
        return;
    };
    if !port
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|identifier| Uuid::parse_str(identifier).is_ok())
    {
        issues.push(issue(
            format!("{path}.id"),
            "schema 1 subgraph port id must be a UUID",
        ));
    }
    validate_schema_one_node_slot(value, path, false, issues);
    if port.get("linkIds").is_some_and(|link_ids| {
        !link_ids
            .as_array()
            .is_some_and(|link_ids| link_ids.iter().all(Value::is_number))
    }) {
        issues.push(issue(
            format!("{path}.linkIds"),
            "schema 1 subgraph link ids must be numeric",
        ));
    }
}

fn valid_slot_index(value: &Value) -> bool {
    value.as_i64().is_some()
        || value
            .as_str()
            .is_some_and(|value| value.trim().parse::<i64>().is_ok())
}

fn valid_vector_two(value: &Value) -> bool {
    value
        .as_array()
        .is_some_and(|values| values.len() == 2 && values.iter().all(Value::is_number))
        || value.as_object().is_some_and(|values| {
            values.get("0").is_some_and(Value::is_number)
                && values.get("1").is_some_and(Value::is_number)
        })
}

fn valid_bounding(value: &Value) -> bool {
    value
        .as_array()
        .is_some_and(|values| values.len() == 4 && values.iter().all(Value::is_number))
}

fn valid_data_type(value: &Value) -> bool {
    value.is_string()
        || value.is_number()
        || value
            .as_array()
            .is_some_and(|values| values.iter().all(Value::is_string))
}

fn issue(path: impl Into<String>, reason: impl Into<String>) -> WorkflowValidationIssue {
    WorkflowValidationIssue {
        path: path.into(),
        reason: reason.into(),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddedImport {
    pub templates: Vec<TemplateDocument>,
    pub primary: EmbeddedPrimary,
    pub rejected: Vec<WorkflowValidationIssue>,
    pub executes_on_import: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EmbeddedPrimary {
    Workflow(WorkflowFormatDocument),
    ApiPrompt(ApiPromptDocument),
    A1111(WorkflowFormatDocument),
    MediaOnly,
}

pub fn import_embedded_metadata(
    document: &MetadataDocument,
) -> Result<EmbeddedImport, WorkflowFormatError> {
    let ComfyMetadata {
        templates,
        workflow,
        prompt,
        parameters,
        ..
    } = document.sim_metadata();
    let mut template_documents = Vec::new();
    let mut rejected = Vec::new();
    for template in templates {
        match import_json(template.as_bytes()) {
            Ok(JsonImport::Templates(template)) => template_documents.push(template),
            Ok(_) => rejected.push(issue(
                "metadata.templates",
                "value is not a template container",
            )),
            Err(error) => rejected.push(issue("metadata.templates", error.to_string())),
        }
    }
    if let Some(workflow) = workflow {
        match WorkflowFormatDocument::parse(workflow.as_bytes()) {
            Ok(workflow) => {
                return Ok(EmbeddedImport {
                    templates: template_documents,
                    primary: EmbeddedPrimary::Workflow(workflow),
                    rejected,
                    executes_on_import: false,
                });
            }
            Err(error) => rejected.push(issue("metadata.workflow", error.to_string())),
        }
    }
    if let Some(prompt) = prompt {
        match import_json(prompt.as_bytes()) {
            Ok(JsonImport::ApiPrompt(prompt)) => {
                return Ok(EmbeddedImport {
                    templates: template_documents,
                    primary: EmbeddedPrimary::ApiPrompt(prompt),
                    rejected,
                    executes_on_import: false,
                });
            }
            Ok(_) => rejected.push(issue("metadata.prompt", "value is not an API prompt")),
            Err(error) => rejected.push(issue("metadata.prompt", error.to_string())),
        }
    }
    if let Some(parameters) = parameters {
        match convert_a1111_parameters(&parameters) {
            Ok(workflow) => {
                return Ok(EmbeddedImport {
                    templates: template_documents,
                    primary: EmbeddedPrimary::A1111(workflow),
                    rejected,
                    executes_on_import: false,
                });
            }
            Err(error) => rejected.push(issue("metadata.parameters", error.to_string())),
        }
    }
    if !template_documents.is_empty() {
        return Ok(EmbeddedImport {
            templates: template_documents,
            primary: EmbeddedPrimary::MediaOnly,
            rejected,
            executes_on_import: false,
        });
    }
    Ok(EmbeddedImport {
        templates: Vec::new(),
        primary: EmbeddedPrimary::MediaOnly,
        rejected,
        executes_on_import: false,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LinkProjection {
    origin: String,
    origin_slot: u64,
    target: String,
    target_slot: u64,
    type_name: String,
}

struct PromptWidgetValue {
    identifier: Option<String>,
    prompt_value: Value,
}

fn prompt_widget_values(
    node: &Map<String, Value>,
    node_index: usize,
) -> Result<Option<Vec<PromptWidgetValue>>, WorkflowFormatError> {
    let workflow_values = node.get("widgets_values").and_then(Value::as_array);
    let Some(native) = node.get(NATIVE_WIDGETS_FIELD) else {
        return Ok(workflow_values.map(|values| {
            values
                .iter()
                .cloned()
                .map(|prompt_value| PromptWidgetValue {
                    identifier: None,
                    prompt_value,
                })
                .collect()
        }));
    };
    let native = native
        .as_object()
        .ok_or_else(|| WorkflowFormatError::PromptProjection {
            path: format!("$.nodes[{node_index}].{NATIVE_WIDGETS_FIELD}"),
            reason: "native widget metadata must be an object".to_owned(),
        })?;
    if native.get("version").and_then(Value::as_u64) != Some(1) {
        return Err(WorkflowFormatError::PromptProjection {
            path: format!("$.nodes[{node_index}].{NATIVE_WIDGETS_FIELD}.version"),
            reason: "native widget metadata version is unsupported".to_owned(),
        });
    }
    let widgets = native
        .get("widgets")
        .and_then(Value::as_array)
        .ok_or_else(|| WorkflowFormatError::PromptProjection {
            path: format!("$.nodes[{node_index}].{NATIVE_WIDGETS_FIELD}.widgets"),
            reason: "native widget metadata widgets must be an array".to_owned(),
        })?;
    let workflow_values = workflow_values.ok_or_else(|| WorkflowFormatError::PromptProjection {
        path: format!("$.nodes[{node_index}].widgets_values"),
        reason: "native widget metadata requires workflow widget values".to_owned(),
    })?;
    if widgets.len() != workflow_values.len() {
        return Err(WorkflowFormatError::PromptProjection {
            path: format!("$.nodes[{node_index}].{NATIVE_WIDGETS_FIELD}.widgets"),
            reason: format!(
                "native widget count {} does not match workflow widget count {}",
                widgets.len(),
                workflow_values.len()
            ),
        });
    }
    let mut identifiers = BTreeSet::new();
    widgets
        .iter()
        .enumerate()
        .map(|(widget_index, widget)| {
            let widget = widget.as_object().ok_or_else(|| {
                WorkflowFormatError::PromptProjection {
                    path: format!(
                        "$.nodes[{node_index}].{NATIVE_WIDGETS_FIELD}.widgets[{widget_index}]"
                    ),
                    reason: "native widget must be an object".to_owned(),
                }
            })?;
            let identifier = widget
                .get("identifier")
                .and_then(Value::as_str)
                .filter(|identifier| !identifier.trim().is_empty())
                .ok_or_else(|| WorkflowFormatError::PromptProjection {
                    path: format!(
                        "$.nodes[{node_index}].{NATIVE_WIDGETS_FIELD}.widgets[{widget_index}].identifier"
                    ),
                    reason: "native widget identifier is required".to_owned(),
                })?;
            if !identifiers.insert(identifier) {
                return Err(WorkflowFormatError::PromptProjection {
                    path: format!(
                        "$.nodes[{node_index}].{NATIVE_WIDGETS_FIELD}.widgets[{widget_index}].identifier"
                    ),
                    reason: "native widget identifier is duplicated".to_owned(),
                });
            }
            let prompt_value = widget.get("prompt_value").cloned().ok_or_else(|| {
                WorkflowFormatError::PromptProjection {
                    path: format!(
                        "$.nodes[{node_index}].{NATIVE_WIDGETS_FIELD}.widgets[{widget_index}].prompt_value"
                    ),
                    reason: "native widget prompt value is required".to_owned(),
                }
            })?;
            Ok(PromptWidgetValue {
                identifier: Some(identifier.to_owned()),
                prompt_value,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

pub fn graph_to_prompt(
    workflow: &WorkflowFormatDocument,
    descriptors: &BTreeMap<String, NodeDescriptor>,
    frontend_version: &str,
) -> Result<PromptSubmission, WorkflowFormatError> {
    let root = workflow
        .value()
        .as_object()
        .ok_or(WorkflowFormatError::RootNotObject)?;
    let nodes = root.get("nodes").and_then(Value::as_array).ok_or_else(|| {
        WorkflowFormatError::PromptProjection {
            path: "$.nodes".to_owned(),
            reason: "nodes must be an array".to_owned(),
        }
    })?;
    if nodes.len() > MAX_WORKFLOW_NODES {
        return Err(WorkflowFormatError::PromptProjection {
            path: "$.nodes".to_owned(),
            reason: "node count exceeds the configured limit".to_owned(),
        });
    }
    let links = project_links(root.get("links"))?;
    let skipped = nodes
        .iter()
        .filter(|node| should_skip_node(node))
        .filter_map(|node| node.get("id").and_then(node_id_text))
        .collect::<BTreeSet<_>>();
    let bypassed = nodes
        .iter()
        .filter(|node| node.get("mode").and_then(Value::as_i64) == Some(4))
        .filter_map(|node| node.get("id").and_then(node_id_text))
        .collect::<BTreeSet<_>>();
    let mut prompt_nodes = BTreeMap::new();
    for (node_index, node) in nodes.iter().enumerate() {
        if should_skip_node(node) {
            if is_virtual_node(node) {
                if let Some(virtual_prompt) = node.get("virtual_prompt") {
                    let virtual_nodes: ApiPrompt = serde_json::from_value(virtual_prompt.clone())
                        .map_err(|error| {
                        WorkflowFormatError::PromptProjection {
                            path: format!("$.nodes[{node_index}].virtual_prompt"),
                            reason: error.to_string(),
                        }
                    })?;
                    for (identifier, node) in virtual_nodes.0 {
                        if prompt_nodes.insert(identifier.clone(), node).is_some() {
                            return Err(WorkflowFormatError::PromptProjection {
                                path: format!("$.nodes[{node_index}].virtual_prompt"),
                                reason: format!(
                                    "virtual prompt repeats node ID `{}`",
                                    identifier.0
                                ),
                            });
                        }
                    }
                }
            }
            continue;
        }
        let node = node
            .as_object()
            .ok_or_else(|| WorkflowFormatError::PromptProjection {
                path: format!("$.nodes[{node_index}]"),
                reason: "node must be an object".to_owned(),
            })?;
        let identifier = node.get("id").and_then(node_id_text).ok_or_else(|| {
            WorkflowFormatError::PromptProjection {
                path: format!("$.nodes[{node_index}].id"),
                reason: "node ID must be a string or number".to_owned(),
            }
        })?;
        let class_type = node
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| WorkflowFormatError::PromptProjection {
                path: format!("$.nodes[{node_index}].type"),
                reason: "node type must be a string".to_owned(),
            })?
            .to_owned();
        let descriptor = descriptors.get(&class_type);
        let mut inputs = BTreeMap::new();
        let mut linked_names = BTreeSet::new();
        if let Some(node_inputs) = node.get("inputs").and_then(Value::as_array) {
            for (input_index, input) in node_inputs.iter().enumerate() {
                let Some(input) = input.as_object() else {
                    continue;
                };
                let Some(name) = input.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let Some(link_id) = input.get("link").filter(|link| !link.is_null()) else {
                    continue;
                };
                let key = json_identity(link_id).ok_or_else(|| {
                    WorkflowFormatError::PromptProjection {
                        path: format!("$.nodes[{node_index}].inputs[{input_index}].link"),
                        reason: "link ID must be a string or number".to_owned(),
                    }
                })?;
                if let Some(link) = links.get(&key) {
                    if link.target != identifier
                        || usize::try_from(link.target_slot).ok() != Some(input_index)
                    {
                        return Err(WorkflowFormatError::PromptProjection {
                            path: format!("$.nodes[{node_index}].inputs[{input_index}].link"),
                            reason: "link target does not match the input slot".to_owned(),
                        });
                    }
                    let Some((origin, origin_slot)) = resolve_link_origin(link, &links, &bypassed)
                    else {
                        continue;
                    };
                    if skipped.contains(&origin) {
                        continue;
                    }
                    inputs.insert(name.to_owned(), json!([origin, origin_slot]));
                    linked_names.insert(name.to_owned());
                }
            }
        }
        if let Some(widget_values) = prompt_widget_values(node, node_index)? {
            let widget_names = widget_names(node, descriptor);
            for (index, widget) in widget_values.iter().enumerate() {
                let name = if let Some(identifier) = &widget.identifier {
                    if descriptor.is_some_and(|descriptor| {
                        !descriptor
                            .inputs
                            .iter()
                            .any(|input| input.name == *identifier)
                    }) {
                        return Err(WorkflowFormatError::PromptProjection {
                            path: format!(
                                "$.nodes[{node_index}].{NATIVE_WIDGETS_FIELD}.widgets[{index}].identifier"
                            ),
                            reason: format!(
                                "native widget identifier `{identifier}` is not an input of `{class_type}`"
                            ),
                        });
                    }
                    identifier
                } else {
                    let Some(name) = widget_names.get(index) else {
                        break;
                    };
                    name
                };
                if linked_names.contains(name) {
                    continue;
                }
                let type_name = descriptor
                    .and_then(|descriptor| {
                        descriptor.inputs.iter().find(|input| input.name == *name)
                    })
                    .map(|input| input.type_name.as_str());
                inputs.insert(
                    name.clone(),
                    wrap_literal(widget.prompt_value.clone(), type_name),
                );
            }
        }
        let mut unknown = BTreeMap::new();
        if let Some(title) = node.get("title").and_then(Value::as_str) {
            unknown.insert("_meta".to_owned(), json!({ "title": title }));
        }
        let previous = prompt_nodes.insert(
            NodeId(identifier),
            PromptNode {
                class_type,
                inputs,
                unknown,
            },
        );
        if previous.is_some() {
            return Err(WorkflowFormatError::PromptProjection {
                path: format!("$.nodes[{node_index}].id"),
                reason: "node ID is duplicated".to_owned(),
            });
        }
    }
    let mut extra_data = BTreeMap::new();
    extra_data.insert(
        "extra_pnginfo".to_owned(),
        json!({
            "workflow": workflow.value(),
            "frontendVersion": frontend_version,
        }),
    );
    Ok(PromptSubmission {
        prompt: ApiPrompt(prompt_nodes),
        prompt_id: None,
        client_id: None,
        number: None,
        extra_data,
        unknown: BTreeMap::new(),
    })
}

fn project_links(
    value: Option<&Value>,
) -> Result<BTreeMap<String, LinkProjection>, WorkflowFormatError> {
    let mut links = BTreeMap::new();
    let Some(value) = value else {
        return Ok(links);
    };
    let array = value
        .as_array()
        .ok_or_else(|| WorkflowFormatError::PromptProjection {
            path: "$.links".to_owned(),
            reason: "links must be an array".to_owned(),
        })?;
    for (index, link) in array.iter().enumerate() {
        let (key, projection, identifier_path) = project_link(index, link)?;
        let previous = links.insert(key, projection);
        if previous.is_some() {
            return Err(WorkflowFormatError::PromptProjection {
                path: identifier_path,
                reason: "link ID is duplicated".to_owned(),
            });
        }
    }
    Ok(links)
}

fn project_link(
    index: usize,
    value: &Value,
) -> Result<(String, LinkProjection, String), WorkflowFormatError> {
    if let Some(link) = value.as_array() {
        if link.len() != 6 {
            return Err(WorkflowFormatError::PromptProjection {
                path: format!("$.links[{index}]"),
                reason: "link must have six entries".to_owned(),
            });
        }
        let identifier_path = format!("$.links[{index}][0]");
        let key = json_identity(&link[0]).ok_or_else(|| WorkflowFormatError::PromptProjection {
            path: identifier_path.clone(),
            reason: "link ID must be a string or number".to_owned(),
        })?;
        let origin =
            node_id_text(&link[1]).ok_or_else(|| WorkflowFormatError::PromptProjection {
                path: format!("$.links[{index}][1]"),
                reason: "origin ID must be a string or number".to_owned(),
            })?;
        let origin_slot =
            projection_slot(&link[2]).ok_or_else(|| WorkflowFormatError::PromptProjection {
                path: format!("$.links[{index}][2]"),
                reason: "origin slot must be an unsigned integer".to_owned(),
            })?;
        let target =
            node_id_text(&link[3]).ok_or_else(|| WorkflowFormatError::PromptProjection {
                path: format!("$.links[{index}][3]"),
                reason: "target ID must be a string or number".to_owned(),
            })?;
        let target_slot =
            projection_slot(&link[4]).ok_or_else(|| WorkflowFormatError::PromptProjection {
                path: format!("$.links[{index}][4]"),
                reason: "target slot must be an unsigned integer".to_owned(),
            })?;
        let type_name =
            data_type_text(&link[5]).ok_or_else(|| WorkflowFormatError::PromptProjection {
                path: format!("$.links[{index}][5]"),
                reason: "link type must be a string, number, or string array".to_owned(),
            })?;
        return Ok((
            key,
            LinkProjection {
                origin,
                origin_slot,
                target,
                target_slot,
                type_name,
            },
            identifier_path,
        ));
    }

    let Some(link) = value.as_object() else {
        return Err(WorkflowFormatError::PromptProjection {
            path: format!("$.links[{index}]"),
            reason: "link must be a six-entry array or an object".to_owned(),
        });
    };
    let identifier_path = format!("$.links[{index}].id");
    let key = link.get("id").and_then(json_identity).ok_or_else(|| {
        WorkflowFormatError::PromptProjection {
            path: identifier_path.clone(),
            reason: "link ID must be a string or number".to_owned(),
        }
    })?;
    let origin = link
        .get("origin_id")
        .and_then(node_id_text)
        .ok_or_else(|| WorkflowFormatError::PromptProjection {
            path: format!("$.links[{index}].origin_id"),
            reason: "origin ID must be a string or number".to_owned(),
        })?;
    let origin_slot = link
        .get("origin_slot")
        .and_then(projection_slot)
        .ok_or_else(|| WorkflowFormatError::PromptProjection {
            path: format!("$.links[{index}].origin_slot"),
            reason: "origin slot must be an unsigned integer".to_owned(),
        })?;
    let target = link
        .get("target_id")
        .and_then(node_id_text)
        .ok_or_else(|| WorkflowFormatError::PromptProjection {
            path: format!("$.links[{index}].target_id"),
            reason: "target ID must be a string or number".to_owned(),
        })?;
    let target_slot = link
        .get("target_slot")
        .and_then(projection_slot)
        .ok_or_else(|| WorkflowFormatError::PromptProjection {
            path: format!("$.links[{index}].target_slot"),
            reason: "target slot must be an unsigned integer".to_owned(),
        })?;
    let type_name = link.get("type").and_then(data_type_text).ok_or_else(|| {
        WorkflowFormatError::PromptProjection {
            path: format!("$.links[{index}].type"),
            reason: "link type must be a string, number, or string array".to_owned(),
        }
    })?;
    Ok((
        key,
        LinkProjection {
            origin,
            origin_slot,
            target,
            target_slot,
            type_name,
        },
        identifier_path,
    ))
}

fn should_skip_node(node: &Value) -> bool {
    let mode = node.get("mode").and_then(Value::as_i64);
    matches!(mode, Some(1 | 2 | 4)) || is_virtual_node(node)
}

fn is_virtual_node(node: &Value) -> bool {
    node.get("properties")
        .and_then(|properties| properties.get("virtualNode"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn resolve_link_origin(
    link: &LinkProjection,
    links: &BTreeMap<String, LinkProjection>,
    bypassed: &BTreeSet<String>,
) -> Option<(String, u64)> {
    let mut origin = link.origin.clone();
    let mut origin_slot = link.origin_slot;
    let mut visited = BTreeSet::new();
    while bypassed.contains(&origin) {
        if !visited.insert(origin.clone()) {
            return None;
        }
        let incoming = links
            .values()
            .filter(|candidate| candidate.target == origin)
            .find(|candidate| candidate.target_slot == origin_slot)
            .or_else(|| {
                links
                    .values()
                    .filter(|candidate| candidate.target == origin)
                    .find(|candidate| candidate.type_name == link.type_name)
            })
            .or_else(|| links.values().find(|candidate| candidate.target == origin))?;
        origin = incoming.origin.clone();
        origin_slot = incoming.origin_slot;
    }
    Some((origin, origin_slot))
}

fn node_id_text(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn json_identity(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(format!("s:{value}")),
        Value::Number(value) => Some(format!("n:{value}")),
        _ => None,
    }
}

fn projection_slot(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| {
        value
            .as_str()
            .and_then(|value| value.trim().parse::<u64>().ok())
    })
}

fn data_type_text(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Array(values) if values.iter().all(Value::is_string) => Some(value.to_string()),
        _ => None,
    }
}

fn widget_names(node: &Map<String, Value>, descriptor: Option<&NodeDescriptor>) -> Vec<String> {
    if let Some(names) = node
        .get("properties")
        .and_then(|properties| properties.get("widget_input_names"))
        .and_then(Value::as_array)
    {
        return names
            .iter()
            .filter_map(Value::as_str)
            .map(ToOwned::to_owned)
            .collect();
    }
    descriptor
        .map(|descriptor| {
            descriptor
                .inputs
                .iter()
                .map(|input| input.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

pub fn wrap_literal(value: Value, type_name: Option<&str>) -> Value {
    if value.is_array() {
        let mut object = Map::from_iter([("__value__".to_owned(), value)]);
        if type_name == Some("CURVE") {
            object.insert("__type__".to_owned(), Value::String("CURVE".to_owned()));
        }
        Value::Object(object)
    } else {
        value
    }
}

pub fn unwrap_literal(value: &Value) -> &Value {
    value
        .as_object()
        .and_then(|object| object.get("__value__"))
        .unwrap_or(value)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ControlMode {
    Fixed,
    Increment,
    Decrement,
    Randomize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ControlledValue {
    Integer(i64),
    Combo(String),
}

pub fn next_controlled_value(
    current: ControlledValue,
    mode: ControlMode,
    bounds: Option<(i64, i64)>,
    combo_values: &[String],
    random_value: i64,
    partial_execution: bool,
) -> ControlledValue {
    if partial_execution || mode == ControlMode::Fixed {
        return current;
    }
    match current {
        ControlledValue::Integer(value) => {
            let (minimum, maximum) = bounds.unwrap_or((-CONTROL_VALUE_LIMIT, CONTROL_VALUE_LIMIT));
            let minimum = minimum.max(-CONTROL_VALUE_LIMIT);
            let maximum = maximum.min(CONTROL_VALUE_LIMIT).max(minimum);
            let value = match mode {
                ControlMode::Fixed => value,
                ControlMode::Increment => value.saturating_add(1),
                ControlMode::Decrement => value.saturating_sub(1),
                ControlMode::Randomize => random_value,
            };
            ControlledValue::Integer(value.clamp(minimum, maximum))
        }
        ControlledValue::Combo(value) => {
            if mode != ControlMode::Increment || combo_values.is_empty() {
                return ControlledValue::Combo(value);
            }
            let next = combo_values
                .iter()
                .position(|candidate| candidate == &value)
                .map(|index| (index + 1) % combo_values.len())
                .unwrap_or(0);
            ControlledValue::Combo(combo_values[next].clone())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileNamespace {
    Input,
    Output,
    Temp,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SavedResultLocator {
    pub filename: String,
    #[serde(default)]
    pub subfolder: String,
    #[serde(rename = "type")]
    pub namespace: FileNamespace,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animated: Option<bool>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

impl SavedResultLocator {
    pub fn asset_namespace(&self) -> crate::AssetNamespace {
        match self.namespace {
            FileNamespace::Input => crate::AssetNamespace::Input,
            FileNamespace::Output => crate::AssetNamespace::Output,
            FileNamespace::Temp => crate::AssetNamespace::Temporary,
        }
    }

    pub fn artifact_key(
        &self,
        root_id: impl Into<String>,
    ) -> Result<ArtifactKey, WorkflowFormatError> {
        let relative_path = if self.subfolder.is_empty() {
            PathBuf::from(&self.filename)
        } else {
            Path::new(&self.subfolder).join(&self.filename)
        };
        ArtifactKey::new(root_id, relative_path)
            .map_err(|error| WorkflowFormatError::InvalidLocator(error.to_string()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkflowModelReference {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory: Option<String>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

pub fn normalize_model_name(name: &str) -> Result<String, WorkflowFormatError> {
    let key = ArtifactKey::new("workflow-model-reference", name)
        .map_err(|error| WorkflowFormatError::InvalidModelPath(error.to_string()))?;
    key.relative_path
        .to_str()
        .map(|path| path.replace('\\', "/"))
        .ok_or_else(|| {
            WorkflowFormatError::InvalidModelPath(
                "canonical model path is not valid UTF-8".to_owned(),
            )
        })
}

pub fn convert_a1111_parameters(
    parameters: &str,
) -> Result<WorkflowFormatDocument, WorkflowFormatError> {
    let negative_marker = "\nNegative prompt:";
    let negative_start = parameters.find(negative_marker).ok_or_else(|| {
        WorkflowFormatError::InvalidA1111("Negative prompt is missing".to_owned())
    })?;
    let positive = parameters[..negative_start].trim();
    let remainder = &parameters[negative_start + negative_marker.len()..];
    let options_start = remainder
        .rfind("\nSteps:")
        .ok_or_else(|| WorkflowFormatError::InvalidA1111("Steps are missing".to_owned()))?;
    let negative = remainder[..options_start].trim();
    let options = parse_a1111_options(&remainder[options_start + 1..]);
    let steps = parse_option_u64(&options, "Steps")?;
    let cfg = parse_option_f64(&options, "CFG scale")?;
    let seed = parse_option_u64(&options, "Seed")?;
    let sampler = options
        .get("Sampler")
        .ok_or_else(|| WorkflowFormatError::InvalidA1111("Sampler is missing".to_owned()))?;
    let size = options
        .get("Size")
        .ok_or_else(|| WorkflowFormatError::InvalidA1111("Size is missing".to_owned()))?;
    let (width, height) = size
        .split_once('x')
        .ok_or_else(|| WorkflowFormatError::InvalidA1111("Size must be WIDTHxHEIGHT".to_owned()))?;
    let width = width
        .trim()
        .parse::<u64>()
        .map_err(|error| WorkflowFormatError::InvalidA1111(format!("invalid width: {error}")))?;
    let height = height
        .trim()
        .parse::<u64>()
        .map_err(|error| WorkflowFormatError::InvalidA1111(format!("invalid height: {error}")))?;
    let model = options
        .get("Model")
        .cloned()
        .unwrap_or_else(|| "model.safetensors".to_owned());
    let sampler = sampler
        .strip_prefix("sample_")
        .unwrap_or(sampler)
        .to_ascii_lowercase()
        .replace(' ', "_");
    let scheduler = options
        .get("Schedule type")
        .cloned()
        .unwrap_or_else(|| "normal".to_owned())
        .to_ascii_lowercase();
    let (positive, loras) = extract_a1111_loras(positive)?;
    let mut nodes = vec![
        json!({"id":1,"type":"CheckpointLoaderSimple","pos":[0,0],"size":[320,100],"inputs":[],"widgets_values":[model]}),
        json!({"id":2,"type":"CLIPTextEncode","pos":[400,0],"size":[320,100],"inputs":[{"name":"clip","link":1}],"widgets_values":[positive]}),
        json!({"id":3,"type":"CLIPTextEncode","pos":[400,160],"size":[320,100],"inputs":[{"name":"clip","link":2}],"widgets_values":[negative]}),
        json!({"id":4,"type":"EmptyLatentImage","pos":[400,320],"size":[320,100],"inputs":[],"widgets_values":[width,height,1]}),
        json!({"id":5,"type":"KSampler","pos":[800,120],"size":[320,260],"inputs":[{"name":"model","link":3},{"name":"positive","link":4},{"name":"negative","link":5},{"name":"latent_image","link":6}],"widgets_values":[seed,"fixed",steps,cfg,sampler,scheduler,1.0]}),
        json!({"id":6,"type":"VAEDecode","pos":[1200,120],"size":[260,100],"inputs":[{"name":"samples","link":7},{"name":"vae","link":8}],"widgets_values":[]}),
        json!({"id":7,"type":"SaveImage","pos":[1520,120],"size":[260,100],"inputs":[{"name":"images","link":9}],"widgets_values":["ComfyUI"]}),
    ];
    let mut links = vec![
        json!([1, 1, 1, 2, 0, "CLIP"]),
        json!([2, 1, 1, 3, 0, "CLIP"]),
        json!([3, 1, 0, 5, 0, "MODEL"]),
        json!([4, 2, 0, 5, 1, "CONDITIONING"]),
        json!([5, 3, 0, 5, 2, "CONDITIONING"]),
        json!([6, 4, 0, 5, 3, "LATENT"]),
        json!([7, 5, 0, 6, 0, "LATENT"]),
        json!([8, 1, 2, 6, 1, "VAE"]),
        json!([9, 6, 0, 7, 0, "IMAGE"]),
    ];
    let mut next_node_id = 8u64;
    let mut next_link_id = 10u64;
    let mut model_source = 1u64;
    let mut clip_source = 1u64;
    for (name, model_weight, clip_weight) in loras {
        let model_link = next_link_id;
        next_link_id += 1;
        let clip_link = next_link_id;
        next_link_id += 1;
        links.push(json!([
            model_link,
            model_source,
            0,
            next_node_id,
            0,
            "MODEL"
        ]));
        links.push(json!([clip_link, clip_source, 1, next_node_id, 1, "CLIP"]));
        nodes.push(json!({
            "id":next_node_id,
            "type":"LoraLoader",
            "pos":[200, next_node_id * 80],
            "size":[320,120],
            "inputs":[{"name":"model","link":model_link},{"name":"clip","link":clip_link}],
            "widgets_values":[name,model_weight,clip_weight]
        }));
        model_source = next_node_id;
        clip_source = next_node_id;
        next_node_id += 1;
    }
    if let Some(clip_skip) = options.get("Clip skip") {
        let clip_skip = clip_skip.parse::<i64>().map_err(|error| {
            WorkflowFormatError::InvalidA1111(format!("invalid Clip skip: {error}"))
        })?;
        let clip_link = next_link_id;
        next_link_id += 1;
        links.push(json!([clip_link, clip_source, 1, next_node_id, 0, "CLIP"]));
        nodes.push(json!({
            "id":next_node_id,
            "type":"CLIPSetLastLayer",
            "pos":[360,320],
            "size":[320,100],
            "inputs":[{"name":"clip","link":clip_link}],
            "widgets_values":[-clip_skip.abs()]
        }));
        clip_source = next_node_id;
        next_node_id += 1;
    }
    links[0][1] = Value::Number(Number::from(clip_source));
    links[0][2] = Value::Number(Number::from(1));
    links[1][1] = Value::Number(Number::from(clip_source));
    links[1][2] = Value::Number(Number::from(1));
    links[2][1] = Value::Number(Number::from(model_source));
    if let Some(upscale) = options.get("Hires upscale") {
        let upscale = upscale.parse::<f64>().map_err(|error| {
            WorkflowFormatError::InvalidA1111(format!("invalid Hires upscale: {error}"))
        })?;
        if !upscale.is_finite() || upscale <= 0.0 {
            return Err(WorkflowFormatError::InvalidA1111(
                "Hires upscale must be positive and finite".to_owned(),
            ));
        }
        let denoise = options
            .get("Denoising strength")
            .map(|value| {
                value.parse::<f64>().map_err(|error| {
                    WorkflowFormatError::InvalidA1111(format!(
                        "invalid Denoising strength: {error}"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(0.7);
        if !(0.0..=1.0).contains(&denoise) {
            return Err(WorkflowFormatError::InvalidA1111(
                "Denoising strength must be between zero and one".to_owned(),
            ));
        }
        let hires_steps = options
            .get("Hires steps")
            .map(|value| {
                value.parse::<u64>().map_err(|error| {
                    WorkflowFormatError::InvalidA1111(format!("invalid Hires steps: {error}"))
                })
            })
            .transpose()?
            .unwrap_or(steps);
        let upscale_node = next_node_id;
        next_node_id += 1;
        let upscale_input = next_link_id;
        next_link_id += 1;
        links.push(json!([upscale_input, 5, 0, upscale_node, 0, "LATENT"]));
        nodes.push(json!({
            "id":upscale_node,
            "type":"LatentUpscaleBy",
            "pos":[1120,420],
            "size":[300,120],
            "inputs":[{"name":"samples","link":upscale_input}],
            "widgets_values":["nearest-exact",upscale]
        }));
        let hires_sampler = next_node_id;
        next_node_id += 1;
        let model_link = next_link_id;
        next_link_id += 1;
        let positive_link = next_link_id;
        next_link_id += 1;
        let negative_link = next_link_id;
        next_link_id += 1;
        let latent_link = next_link_id;
        next_link_id += 1;
        links.push(json!([
            model_link,
            model_source,
            0,
            hires_sampler,
            0,
            "MODEL"
        ]));
        links.push(json!([
            positive_link,
            2,
            0,
            hires_sampler,
            1,
            "CONDITIONING"
        ]));
        links.push(json!([
            negative_link,
            3,
            0,
            hires_sampler,
            2,
            "CONDITIONING"
        ]));
        links.push(json!([
            latent_link,
            upscale_node,
            0,
            hires_sampler,
            3,
            "LATENT"
        ]));
        nodes.push(json!({
            "id":hires_sampler,
            "type":"KSampler",
            "pos":[1440,420],
            "size":[320,260],
            "inputs":[
                {"name":"model","link":model_link},
                {"name":"positive","link":positive_link},
                {"name":"negative","link":negative_link},
                {"name":"latent_image","link":latent_link}
            ],
            "widgets_values":[seed,"fixed",hires_steps,cfg,sampler,scheduler,denoise]
        }));
        links[6][1] = Value::Number(Number::from(hires_sampler));
    }
    complete_generated_schema_zero_four_nodes(&mut nodes, &links)?;
    let value = json!({
        "version": 0.4,
        "last_node_id": next_node_id - 1,
        "last_link_id": next_link_id - 1,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"a1111": {"source": parameters, "options": options}},
    });
    let bytes = serde_json::to_vec(&value)
        .map_err(|error| WorkflowFormatError::Serialization(error.to_string()))?;
    let document = WorkflowFormatDocument::from_parts(&bytes, value, Vec::new())?;
    if let Some(issue) = document.validation_issues().first() {
        return Err(WorkflowFormatError::InvalidA1111(format!(
            "generated workflow is invalid at {}: {}",
            issue.path, issue.reason
        )));
    }
    Ok(document)
}

fn complete_generated_schema_zero_four_nodes(
    nodes: &mut [Value],
    links: &[Value],
) -> Result<(), WorkflowFormatError> {
    let mut input_types = BTreeMap::new();
    for link in links {
        let Some(link) = link.as_array().filter(|link| link.len() == 6) else {
            return Err(WorkflowFormatError::InvalidA1111(
                "generated link is malformed".to_owned(),
            ));
        };
        let target = node_id_text(&link[3]).ok_or_else(|| {
            WorkflowFormatError::InvalidA1111("generated link target is invalid".to_owned())
        })?;
        let target_slot = projection_slot(&link[4]).ok_or_else(|| {
            WorkflowFormatError::InvalidA1111("generated link target slot is invalid".to_owned())
        })?;
        input_types.insert((target, target_slot), link[5].clone());
    }
    for (order, node) in nodes.iter_mut().enumerate() {
        let node = node.as_object_mut().ok_or_else(|| {
            WorkflowFormatError::InvalidA1111("generated node is not an object".to_owned())
        })?;
        let identifier = node.get("id").and_then(node_id_text).ok_or_else(|| {
            WorkflowFormatError::InvalidA1111("generated node ID is invalid".to_owned())
        })?;
        node.entry("flags").or_insert_with(|| json!({}));
        node.entry("order").or_insert_with(|| json!(order));
        node.entry("mode").or_insert_with(|| json!(0));
        node.entry("properties").or_insert_with(|| json!({}));
        if let Some(inputs) = node.get_mut("inputs").and_then(Value::as_array_mut) {
            for (slot, input) in inputs.iter_mut().enumerate() {
                let input = input.as_object_mut().ok_or_else(|| {
                    WorkflowFormatError::InvalidA1111(
                        "generated node input is not an object".to_owned(),
                    )
                })?;
                if !input.contains_key("type") {
                    let slot = u64::try_from(slot).map_err(|_| {
                        WorkflowFormatError::InvalidA1111(
                            "generated node input slot is too large".to_owned(),
                        )
                    })?;
                    let type_name = input_types
                        .get(&(identifier.clone(), slot))
                        .cloned()
                        .ok_or_else(|| {
                            WorkflowFormatError::InvalidA1111(format!(
                                "generated node {identifier} input {slot} has no link type"
                            ))
                        })?;
                    input.insert("type".to_owned(), type_name);
                }
            }
        }
    }
    Ok(())
}

fn extract_a1111_loras(
    prompt: &str,
) -> Result<(String, Vec<(String, f64, f64)>), WorkflowFormatError> {
    let mut clean = String::with_capacity(prompt.len());
    let mut loras = Vec::new();
    let mut position = 0usize;
    while let Some(relative) = prompt[position..].find("<lora:") {
        let start = position + relative;
        clean.push_str(&prompt[position..start]);
        let end = prompt[start..]
            .find('>')
            .map(|offset| start + offset)
            .ok_or_else(|| WorkflowFormatError::InvalidA1111("unterminated LoRA tag".to_owned()))?;
        let fields = prompt[start + "<lora:".len()..end]
            .split(':')
            .map(str::trim)
            .collect::<Vec<_>>();
        if !(2..=3).contains(&fields.len()) || fields[0].is_empty() {
            return Err(WorkflowFormatError::InvalidA1111(
                "LoRA tags must be <lora:name:model-weight[:clip-weight]>".to_owned(),
            ));
        }
        let model_weight = fields[1].parse::<f64>().map_err(|error| {
            WorkflowFormatError::InvalidA1111(format!("invalid LoRA model weight: {error}"))
        })?;
        let clip_weight = fields
            .get(2)
            .map(|value| {
                value.parse::<f64>().map_err(|error| {
                    WorkflowFormatError::InvalidA1111(format!("invalid LoRA CLIP weight: {error}"))
                })
            })
            .transpose()?
            .unwrap_or(model_weight);
        if !model_weight.is_finite() || !clip_weight.is_finite() {
            return Err(WorkflowFormatError::InvalidA1111(
                "LoRA weights must be finite".to_owned(),
            ));
        }
        let name = if fields[0].contains('.') {
            fields[0].to_owned()
        } else {
            format!("{}.safetensors", fields[0])
        };
        loras.push((name, model_weight, clip_weight));
        position = end + 1;
    }
    clean.push_str(&prompt[position..]);
    Ok((clean.trim().to_owned(), loras))
}

fn parse_a1111_options(line: &str) -> BTreeMap<String, String> {
    let mut options = BTreeMap::new();
    let mut start = 0usize;
    let mut quoted = false;
    let mut nesting = 0usize;
    let bytes = line.as_bytes();
    for (index, byte) in bytes.iter().copied().enumerate() {
        if byte == b'"' {
            quoted = !quoted;
        } else if !quoted && matches!(byte, b'{' | b'[') {
            nesting = nesting.saturating_add(1);
        } else if !quoted && matches!(byte, b'}' | b']') {
            nesting = nesting.saturating_sub(1);
        } else if byte == b',' && !quoted && nesting == 0 {
            insert_a1111_option(&line[start..index], &mut options);
            start = index + 1;
        }
    }
    insert_a1111_option(&line[start..], &mut options);
    options
}

fn insert_a1111_option(part: &str, options: &mut BTreeMap<String, String>) {
    if let Some((key, value)) = part.trim().split_once(':') {
        options.insert(
            key.trim().to_owned(),
            value.trim().trim_matches('"').to_owned(),
        );
    }
}

fn parse_option_u64(
    options: &BTreeMap<String, String>,
    key: &str,
) -> Result<u64, WorkflowFormatError> {
    options
        .get(key)
        .ok_or_else(|| WorkflowFormatError::InvalidA1111(format!("{key} is missing")))?
        .parse::<u64>()
        .map_err(|error| WorkflowFormatError::InvalidA1111(format!("invalid {key}: {error}")))
}

fn parse_option_f64(
    options: &BTreeMap<String, String>,
    key: &str,
) -> Result<f64, WorkflowFormatError> {
    let value = options
        .get(key)
        .ok_or_else(|| WorkflowFormatError::InvalidA1111(format!("{key} is missing")))?
        .parse::<f64>()
        .map_err(|error| WorkflowFormatError::InvalidA1111(format!("invalid {key}: {error}")))?;
    if !value.is_finite() {
        return Err(WorkflowFormatError::InvalidA1111(format!(
            "{key} must be finite"
        )));
    }
    Ok(value)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TemplateProvenance {
    LocalFile {
        path: String,
    },
    Bundled {
        identifier: String,
    },
    Provider {
        provider: String,
        identifier: String,
    },
    Url {
        url: String,
    },
    SignedPlugin {
        plugin: String,
        digest: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TemplateRequirement {
    #[serde(default = "default_template_requirement_kind")]
    pub kind: String,
    #[serde(alias = "name")]
    pub identifier: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

fn default_template_requirement_kind() -> String {
    "model".to_owned()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TemplateInstantiation {
    pub document_identity: Uuid,
    pub provenance: TemplateProvenance,
    pub thumbnail: Option<String>,
    pub requirements: Vec<TemplateRequirement>,
    pub missing_node_identifiers: Vec<String>,
    pub workflow: Value,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

impl TemplateDocument {
    pub fn instantiate(
        &self,
        index: usize,
        provenance: TemplateProvenance,
        installed_nodes: &BTreeSet<String>,
        document_identity: Uuid,
    ) -> Result<TemplateInstantiation, WorkflowFormatError> {
        let templates =
            self.templates
                .as_array()
                .ok_or_else(|| WorkflowFormatError::InvalidJson {
                    path: "$.templates".to_owned(),
                    reason: "templates must be an array".to_owned(),
                })?;
        let template = templates
            .get(index)
            .and_then(Value::as_object)
            .ok_or_else(|| WorkflowFormatError::InvalidJson {
                path: format!("$.templates[{index}]"),
                reason: "template must be an object".to_owned(),
            })?;
        let workflow = template
            .get("workflow")
            .or_else(|| template.get("data"))
            .cloned()
            .ok_or_else(|| WorkflowFormatError::InvalidJson {
                path: format!("$.templates[{index}].workflow"),
                reason: "template workflow is missing".to_owned(),
            })?;
        let requirements = if let Some(requirements) = template
            .get("models")
            .or_else(|| template.get("requirements"))
        {
            let requirements =
                requirements
                    .as_array()
                    .ok_or_else(|| WorkflowFormatError::InvalidJson {
                        path: format!("$.templates[{index}].requirements"),
                        reason: "template requirements must be an array".to_owned(),
                    })?;
            requirements
                .iter()
                .enumerate()
                .map(|(requirement_index, requirement)| {
                    serde_json::from_value::<TemplateRequirement>(requirement.clone()).map_err(
                        |error| WorkflowFormatError::InvalidJson {
                            path: format!("$.templates[{index}].requirements[{requirement_index}]"),
                            reason: error.to_string(),
                        },
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let mut node_identifiers = BTreeSet::new();
        collect_workflow_node_types(&workflow, &mut node_identifiers);
        let missing_node_identifiers = node_identifiers
            .difference(installed_nodes)
            .cloned()
            .collect();
        let known = ["workflow", "data", "models", "requirements", "thumbnail"];
        let unknown = template
            .iter()
            .filter(|(key, _)| !known.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        Ok(TemplateInstantiation {
            document_identity,
            provenance,
            thumbnail: template
                .get("thumbnail")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            requirements,
            missing_node_identifiers,
            workflow,
            unknown,
        })
    }
}

fn collect_workflow_node_types(value: &Value, identifiers: &mut BTreeSet<String>) {
    if let Some(nodes) = value.get("nodes").and_then(Value::as_array) {
        for node in nodes {
            if let Some(identifier) = node.get("type").and_then(Value::as_str) {
                identifiers.insert(identifier.to_owned());
            }
        }
    }
    if let Some(definitions) = value.get("definitions") {
        match definitions {
            Value::Array(definitions) => {
                for definition in definitions {
                    collect_workflow_node_types(definition, identifiers);
                }
            }
            Value::Object(definitions) => {
                for definition in definitions.values() {
                    collect_workflow_node_types(definition, identifiers);
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppModePort {
    pub identifier: String,
    pub node_identifier: String,
    pub port: String,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AppModeConfiguration {
    pub inputs: Vec<AppModePort>,
    pub outputs: Vec<AppModePort>,
    pub allow_restore_editing: bool,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AppModeDocument {
    workflow: WorkflowFormatDocument,
    configuration: AppModeConfiguration,
    editing_hidden: bool,
}

impl AppModeDocument {
    pub fn new(workflow: WorkflowFormatDocument, configuration: AppModeConfiguration) -> Self {
        Self {
            workflow,
            configuration,
            editing_hidden: true,
        }
    }

    pub fn workflow(&self) -> &WorkflowFormatDocument {
        &self.workflow
    }

    pub fn configuration(&self) -> &AppModeConfiguration {
        &self.configuration
    }

    pub fn editing_hidden(&self) -> bool {
        self.editing_hidden
    }

    pub fn restore_editing(&mut self) -> bool {
        if !self.configuration.allow_restore_editing {
            return false;
        }
        self.editing_hidden = false;
        true
    }

    pub fn hide_editing(&mut self) {
        self.editing_hidden = true;
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkflowStorageProvider {
    LocalFile,
    Draft,
    Provider { identifier: String },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentRevision(pub String);

impl ContentRevision {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(format!("{:x}", Sha256::digest(bytes)))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkflowAuthority {
    InSync,
    LocalDirty,
    ExternalNewer,
    ExternalMissing,
    Conflict,
    SavePrepared,
    Interrupted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VersionedWorkflowContent {
    pub revision: ContentRevision,
    #[serde(with = "base64_bytes")]
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparedWorkflowSave {
    pub operation_id: Uuid,
    pub expected_revision: ContentRevision,
    pub target_identity: String,
    #[serde(with = "base64_bytes")]
    pub bytes: Vec<u8>,
    pub save_copy: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WorkflowSaveCoordinator {
    schema_version: u16,
    document_identity: String,
    provider: WorkflowStorageProvider,
    base: VersionedWorkflowContent,
    #[serde(with = "base64_bytes")]
    local_bytes: Vec<u8>,
    external: Option<VersionedWorkflowContent>,
    external_missing: bool,
    missing_recreation_approved: bool,
    authority: WorkflowAuthority,
    prepared: Option<PreparedWorkflowSave>,
    #[serde(flatten)]
    unknown: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct WorkflowSaveCoordinatorDocument {
    schema_version: u16,
    document_identity: String,
    provider: WorkflowStorageProvider,
    base: VersionedWorkflowContent,
    #[serde(with = "base64_bytes")]
    local_bytes: Vec<u8>,
    external: Option<VersionedWorkflowContent>,
    #[serde(default)]
    external_missing: bool,
    #[serde(default)]
    missing_recreation_approved: bool,
    authority: WorkflowAuthority,
    prepared: Option<PreparedWorkflowSave>,
    #[serde(flatten)]
    unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowComparison<'a> {
    pub base: &'a [u8],
    pub local: &'a [u8],
    pub external: Option<&'a [u8]>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkflowSaveError {
    #[error("workflow content is {actual} bytes, exceeding the {limit}-byte limit")]
    InputTooLarge { actual: usize, limit: usize },
    #[error("save journal is {actual} bytes, exceeding the {limit}-byte limit")]
    JournalTooLarge { actual: usize, limit: usize },
    #[error("external revision changed from {expected:?} to {actual:?}")]
    Conflict {
        expected: ContentRevision,
        actual: ContentRevision,
    },
    #[error("save operation `{0}` is not the prepared operation")]
    UnknownOperation(Uuid),
    #[error("no external version is available for reload")]
    NoExternalVersion,
    #[error("save coordinator schema version {0} is unsupported")]
    UnsupportedSchema(u16),
    #[error("save coordinator state is inconsistent: {0}")]
    InvalidState(String),
    #[error("save coordinator persistence failed: {0}")]
    Persistence(String),
}

impl WorkflowSaveCoordinator {
    pub const SCHEMA_VERSION: u16 = 2;

    pub fn new(
        document_identity: impl Into<String>,
        provider: WorkflowStorageProvider,
        bytes: Vec<u8>,
    ) -> Result<Self, WorkflowSaveError> {
        validate_save_bytes(&bytes)?;
        let document_identity = document_identity.into();
        validate_document_identity(&document_identity)?;
        validate_storage_provider(&provider)?;
        let revision = ContentRevision::from_bytes(&bytes);
        Ok(Self {
            schema_version: Self::SCHEMA_VERSION,
            document_identity,
            provider,
            base: VersionedWorkflowContent {
                revision,
                bytes: bytes.clone(),
            },
            local_bytes: bytes,
            external: None,
            external_missing: false,
            missing_recreation_approved: false,
            authority: WorkflowAuthority::InSync,
            prepared: None,
            unknown: BTreeMap::new(),
        })
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn document_identity(&self) -> &str {
        &self.document_identity
    }

    pub fn provider(&self) -> &WorkflowStorageProvider {
        &self.provider
    }

    pub fn base(&self) -> &VersionedWorkflowContent {
        &self.base
    }

    pub fn local_bytes(&self) -> &[u8] {
        &self.local_bytes
    }

    pub fn external(&self) -> Option<&VersionedWorkflowContent> {
        self.external.as_ref()
    }

    pub const fn external_missing(&self) -> bool {
        self.external_missing
    }

    pub const fn missing_recreation_approved(&self) -> bool {
        self.missing_recreation_approved
    }

    pub const fn authority(&self) -> WorkflowAuthority {
        self.authority
    }

    pub fn prepared(&self) -> Option<&PreparedWorkflowSave> {
        self.prepared.as_ref()
    }

    pub fn unknown(&self) -> &BTreeMap<String, Value> {
        &self.unknown
    }

    pub fn retarget_local_file(
        &mut self,
        document_identity: impl Into<String>,
    ) -> Result<(), WorkflowSaveError> {
        if self.provider != WorkflowStorageProvider::LocalFile {
            return Err(WorkflowSaveError::InvalidState(
                "only a local-file workflow can be retargeted after rename".to_owned(),
            ));
        }
        if self.prepared.is_some() {
            return Err(WorkflowSaveError::InvalidState(
                "a workflow with a prepared save cannot be retargeted".to_owned(),
            ));
        }
        let document_identity = document_identity.into();
        validate_document_identity(&document_identity)?;
        self.document_identity = document_identity;
        Ok(())
    }

    pub fn switch_provider_after_committed_save(
        &mut self,
        provider: WorkflowStorageProvider,
    ) -> Result<(), WorkflowSaveError> {
        validate_storage_provider(&provider)?;
        if self.authority != WorkflowAuthority::InSync
            || self.prepared.is_some()
            || self.external.is_some()
            || self.external_missing
            || self.missing_recreation_approved
            || self.local_bytes != self.base.bytes
        {
            return Err(WorkflowSaveError::InvalidState(
                "storage provider can change only after a fully committed save".to_owned(),
            ));
        }
        self.provider = provider;
        Ok(())
    }

    pub fn detach_local_file_to_draft(
        &mut self,
        document_identity: impl Into<String>,
    ) -> Result<(), WorkflowSaveError> {
        if self.provider != WorkflowStorageProvider::LocalFile {
            return Err(WorkflowSaveError::InvalidState(
                "only a local-file workflow can detach to a draft".to_owned(),
            ));
        }
        let document_identity = document_identity.into();
        validate_document_identity(&document_identity)?;
        self.document_identity = document_identity;
        self.provider = WorkflowStorageProvider::Draft;
        self.external = None;
        self.external_missing = false;
        self.missing_recreation_approved = false;
        self.prepared = None;
        self.authority = WorkflowAuthority::LocalDirty;
        validate_coordinator_authority(self)
    }

    pub fn edit(&mut self, bytes: Vec<u8>) -> Result<(), WorkflowSaveError> {
        validate_save_bytes(&bytes)?;
        self.local_bytes = bytes;
        self.authority = match (
            &self.external,
            self.external_missing,
            self.missing_recreation_approved,
        ) {
            (None, true, true) => WorkflowAuthority::LocalDirty,
            (None, true, false) if self.local_bytes == self.base.bytes => {
                WorkflowAuthority::ExternalMissing
            }
            (None, true, false) => WorkflowAuthority::Conflict,
            (Some(_), false, false) if self.local_bytes == self.base.bytes => {
                WorkflowAuthority::ExternalNewer
            }
            (Some(_), false, false) => WorkflowAuthority::Conflict,
            (None, false, false) if self.local_bytes == self.base.bytes => {
                WorkflowAuthority::InSync
            }
            (None, false, false) => WorkflowAuthority::LocalDirty,
            (Some(_), true, _) | (Some(_), false, true) | (None, false, true) => {
                return Err(WorkflowSaveError::InvalidState(
                    "missing-file recreation approval has an inconsistent external state"
                        .to_owned(),
                ));
            }
        };
        self.prepared = None;
        Ok(())
    }

    pub fn observe_external_change(&mut self, bytes: Vec<u8>) -> Result<(), WorkflowSaveError> {
        validate_save_bytes(&bytes)?;
        self.external_missing = false;
        self.missing_recreation_approved = false;
        let revision = ContentRevision::from_bytes(&bytes);
        if revision == self.base.revision {
            self.external = None;
            self.prepared = None;
            self.authority = if self.local_bytes == self.base.bytes {
                WorkflowAuthority::InSync
            } else {
                WorkflowAuthority::LocalDirty
            };
            return Ok(());
        }
        if bytes == self.local_bytes {
            self.base = VersionedWorkflowContent { revision, bytes };
            self.external = None;
            self.prepared = None;
            self.authority = WorkflowAuthority::InSync;
            return Ok(());
        }
        self.external = Some(VersionedWorkflowContent { revision, bytes });
        self.prepared = None;
        self.authority = if self.local_bytes == self.base.bytes {
            WorkflowAuthority::ExternalNewer
        } else {
            WorkflowAuthority::Conflict
        };
        Ok(())
    }

    pub fn observe_external_deletion(&mut self) -> Result<(), WorkflowSaveError> {
        if self.provider != WorkflowStorageProvider::LocalFile {
            return Err(WorkflowSaveError::InvalidState(
                "only a local-file workflow can observe external deletion".to_owned(),
            ));
        }
        if self.external_missing {
            return validate_coordinator_authority(self);
        }
        self.external = None;
        self.external_missing = true;
        self.missing_recreation_approved = false;
        self.prepared = None;
        self.authority = if self.local_bytes == self.base.bytes {
            WorkflowAuthority::ExternalMissing
        } else {
            WorkflowAuthority::Conflict
        };
        validate_coordinator_authority(self)
    }

    pub fn prepare_save(
        &mut self,
        operation_id: Uuid,
        observed_revision: ContentRevision,
        target_identity: impl Into<String>,
        save_copy: bool,
    ) -> Result<PreparedWorkflowSave, WorkflowSaveError> {
        let target_identity = target_identity.into();
        validate_document_identity(&target_identity)?;
        if let Some(prepared) = &self.prepared {
            if prepared.operation_id == operation_id
                && prepared.expected_revision == observed_revision
                && prepared.target_identity == target_identity
                && prepared.bytes == self.local_bytes
                && prepared.save_copy == save_copy
            {
                return Ok(prepared.clone());
            }
            return Err(WorkflowSaveError::InvalidState(
                "a different workflow save is already prepared".to_owned(),
            ));
        }
        if !save_copy && self.external_missing && !self.missing_recreation_approved {
            self.authority = if self.local_bytes == self.base.bytes {
                WorkflowAuthority::ExternalMissing
            } else {
                WorkflowAuthority::Conflict
            };
            return Err(WorkflowSaveError::InvalidState(
                "external workflow is missing; keep the local version before recreating it"
                    .to_owned(),
            ));
        }
        let authoritative_revision = self
            .external
            .as_ref()
            .map(|external| &external.revision)
            .unwrap_or(&self.base.revision);
        if !save_copy && observed_revision != *authoritative_revision {
            self.authority = WorkflowAuthority::Conflict;
            return Err(WorkflowSaveError::Conflict {
                expected: authoritative_revision.clone(),
                actual: observed_revision,
            });
        }
        if !save_copy && self.external.is_some() {
            self.authority = WorkflowAuthority::Conflict;
            return Err(WorkflowSaveError::Conflict {
                expected: self.base.revision.clone(),
                actual: authoritative_revision.clone(),
            });
        }
        self.prepared = Some(PreparedWorkflowSave {
            operation_id,
            expected_revision: observed_revision,
            target_identity,
            bytes: self.local_bytes.clone(),
            save_copy,
        });
        self.authority = WorkflowAuthority::SavePrepared;
        self.prepared
            .clone()
            .ok_or_else(|| WorkflowSaveError::InvalidState("prepared save disappeared".to_owned()))
    }

    pub fn commit_save(
        &mut self,
        operation_id: Uuid,
        observed_revision: ContentRevision,
        committed_revision: ContentRevision,
    ) -> Result<(), WorkflowSaveError> {
        let prepared = self
            .prepared
            .as_ref()
            .filter(|prepared| prepared.operation_id == operation_id)
            .cloned()
            .ok_or(WorkflowSaveError::UnknownOperation(operation_id))?;
        if !prepared.save_copy && observed_revision != prepared.expected_revision {
            self.prepared = None;
            self.authority = WorkflowAuthority::Interrupted;
            return Err(WorkflowSaveError::Conflict {
                expected: prepared.expected_revision,
                actual: observed_revision,
            });
        }
        let actual_committed_revision = ContentRevision::from_bytes(&prepared.bytes);
        if committed_revision != actual_committed_revision {
            self.prepared = None;
            self.authority = WorkflowAuthority::Interrupted;
            return Err(WorkflowSaveError::InvalidState(
                "committed revision does not match prepared bytes".to_owned(),
            ));
        }
        self.document_identity = prepared.target_identity;
        self.base = VersionedWorkflowContent {
            revision: committed_revision,
            bytes: prepared.bytes.clone(),
        };
        self.local_bytes = prepared.bytes;
        self.external = None;
        self.external_missing = false;
        self.missing_recreation_approved = false;
        self.prepared = None;
        self.authority = WorkflowAuthority::InSync;
        Ok(())
    }

    pub fn recover_after_restart(&mut self) {
        if self.prepared.take().is_some() {
            self.authority = WorkflowAuthority::Interrupted;
        }
    }

    pub fn comparison(&self) -> WorkflowComparison<'_> {
        WorkflowComparison {
            base: &self.base.bytes,
            local: &self.local_bytes,
            external: self
                .external
                .as_ref()
                .map(|external| external.bytes.as_slice()),
        }
    }

    pub fn reload_external(&mut self) -> Result<(), WorkflowSaveError> {
        if self.external_missing {
            return Err(WorkflowSaveError::NoExternalVersion);
        }
        let external = self
            .external
            .take()
            .ok_or(WorkflowSaveError::NoExternalVersion)?;
        self.base = external.clone();
        self.local_bytes = external.bytes;
        self.prepared = None;
        self.authority = WorkflowAuthority::InSync;
        Ok(())
    }

    pub fn keep_local(&mut self) -> Result<(), WorkflowSaveError> {
        if self.external_missing {
            self.external = None;
            self.missing_recreation_approved = true;
            self.prepared = None;
            self.authority = WorkflowAuthority::LocalDirty;
            return validate_coordinator_authority(self);
        }
        let external = self
            .external
            .take()
            .ok_or(WorkflowSaveError::NoExternalVersion)?;
        self.base = external;
        self.prepared = None;
        self.authority = if self.local_bytes == self.base.bytes {
            WorkflowAuthority::InSync
        } else {
            WorkflowAuthority::LocalDirty
        };
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, WorkflowSaveError> {
        let encoded = serde_json::to_vec(self)
            .map_err(|error| WorkflowSaveError::Persistence(error.to_string()))?;
        validate_journal_bytes(&encoded)?;
        Ok(encoded)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, WorkflowSaveError> {
        validate_journal_bytes(bytes)?;
        let document: WorkflowSaveCoordinatorDocument = serde_json::from_slice(bytes)
            .map_err(|error| WorkflowSaveError::Persistence(error.to_string()))?;
        let mut coordinator = Self {
            schema_version: document.schema_version,
            document_identity: document.document_identity,
            provider: document.provider,
            base: document.base,
            local_bytes: document.local_bytes,
            external: document.external,
            external_missing: document.external_missing,
            missing_recreation_approved: document.missing_recreation_approved,
            authority: document.authority,
            prepared: document.prepared,
            unknown: document.unknown,
        };
        if !matches!(coordinator.schema_version, 1 | Self::SCHEMA_VERSION) {
            return Err(WorkflowSaveError::UnsupportedSchema(
                coordinator.schema_version,
            ));
        }
        coordinator.schema_version = Self::SCHEMA_VERSION;
        validate_document_identity(&coordinator.document_identity)?;
        validate_storage_provider(&coordinator.provider)?;
        if coordinator.base.revision != ContentRevision::from_bytes(&coordinator.base.bytes) {
            return Err(WorkflowSaveError::InvalidState(
                "base revision does not match its bytes".to_owned(),
            ));
        }
        validate_save_bytes(&coordinator.base.bytes)?;
        validate_save_bytes(&coordinator.local_bytes)?;
        if let Some(external) = &coordinator.external {
            validate_save_bytes(&external.bytes)?;
            if external.revision != ContentRevision::from_bytes(&external.bytes) {
                return Err(WorkflowSaveError::InvalidState(
                    "external revision does not match its bytes".to_owned(),
                ));
            }
        }
        if coordinator.external_missing && coordinator.external.is_some() {
            return Err(WorkflowSaveError::InvalidState(
                "external workflow cannot be both changed and missing".to_owned(),
            ));
        }
        if coordinator.missing_recreation_approved && !coordinator.external_missing {
            return Err(WorkflowSaveError::InvalidState(
                "missing-file recreation approval requires a missing external workflow".to_owned(),
            ));
        }
        if coordinator.external_missing
            && coordinator.provider != WorkflowStorageProvider::LocalFile
        {
            return Err(WorkflowSaveError::InvalidState(
                "only a local-file workflow can persist external deletion".to_owned(),
            ));
        }
        if let Some(prepared) = &coordinator.prepared {
            validate_save_bytes(&prepared.bytes)?;
            if prepared.bytes != coordinator.local_bytes {
                return Err(WorkflowSaveError::InvalidState(
                    "prepared bytes do not match local bytes".to_owned(),
                ));
            }
            validate_document_identity(&prepared.target_identity)?;
        }
        if coordinator.authority == WorkflowAuthority::SavePrepared
            && coordinator.prepared.is_none()
        {
            return Err(WorkflowSaveError::InvalidState(
                "SavePrepared state has no operation".to_owned(),
            ));
        }
        if coordinator.prepared.is_some()
            && coordinator.authority != WorkflowAuthority::SavePrepared
        {
            return Err(WorkflowSaveError::InvalidState(
                "prepared operation is present outside SavePrepared state".to_owned(),
            ));
        }
        validate_coordinator_authority(&coordinator)?;
        coordinator.recover_after_restart();
        Ok(coordinator)
    }
}

fn validate_coordinator_authority(
    coordinator: &WorkflowSaveCoordinator,
) -> Result<(), WorkflowSaveError> {
    let valid = match coordinator.authority {
        WorkflowAuthority::InSync => {
            coordinator.local_bytes == coordinator.base.bytes
                && coordinator.external.is_none()
                && !coordinator.external_missing
                && !coordinator.missing_recreation_approved
                && coordinator.prepared.is_none()
        }
        WorkflowAuthority::LocalDirty => {
            coordinator.external.is_none()
                && ((!coordinator.external_missing && !coordinator.missing_recreation_approved)
                    || (coordinator.external_missing && coordinator.missing_recreation_approved))
                && coordinator.prepared.is_none()
        }
        WorkflowAuthority::ExternalNewer => {
            coordinator.local_bytes == coordinator.base.bytes
                && coordinator.external.is_some()
                && !coordinator.external_missing
                && !coordinator.missing_recreation_approved
                && coordinator.prepared.is_none()
        }
        WorkflowAuthority::Conflict => {
            (coordinator.external.is_some() || coordinator.external_missing)
                && !coordinator.missing_recreation_approved
                && coordinator.prepared.is_none()
        }
        WorkflowAuthority::ExternalMissing => {
            coordinator.local_bytes == coordinator.base.bytes
                && coordinator.external.is_none()
                && coordinator.external_missing
                && !coordinator.missing_recreation_approved
                && coordinator.prepared.is_none()
        }
        WorkflowAuthority::SavePrepared => coordinator.prepared.is_some(),
        WorkflowAuthority::Interrupted => coordinator.prepared.is_none(),
    };
    if !valid {
        return Err(WorkflowSaveError::InvalidState(
            "authority does not match persisted versions".to_owned(),
        ));
    }
    Ok(())
}

fn validate_document_identity(document_identity: &str) -> Result<(), WorkflowSaveError> {
    if document_identity.trim().is_empty() {
        return Err(WorkflowSaveError::InvalidState(
            "workflow document identity is empty".to_owned(),
        ));
    }
    if document_identity.len() > MAX_WORKFLOW_IDENTITY_BYTES {
        return Err(WorkflowSaveError::InvalidState(format!(
            "workflow document identity exceeds {MAX_WORKFLOW_IDENTITY_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_storage_provider(provider: &WorkflowStorageProvider) -> Result<(), WorkflowSaveError> {
    if let WorkflowStorageProvider::Provider { identifier } = provider {
        if identifier.trim().is_empty() {
            return Err(WorkflowSaveError::InvalidState(
                "workflow storage provider identifier is empty".to_owned(),
            ));
        }
        if identifier.len() > MAX_PROVIDER_IDENTIFIER_BYTES {
            return Err(WorkflowSaveError::InvalidState(format!(
                "workflow storage provider identifier exceeds {MAX_PROVIDER_IDENTIFIER_BYTES} bytes"
            )));
        }
    }
    Ok(())
}

mod base64_bytes {
    use super::BASE64;
    use base64::Engine as _;
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&BASE64.encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        BASE64.decode(encoded).map_err(D::Error::custom)
    }
}

fn validate_save_bytes(bytes: &[u8]) -> Result<(), WorkflowSaveError> {
    if bytes.len() > MAX_WORKFLOW_BYTES {
        return Err(WorkflowSaveError::InputTooLarge {
            actual: bytes.len(),
            limit: MAX_WORKFLOW_BYTES,
        });
    }
    Ok(())
}

fn validate_journal_bytes(bytes: &[u8]) -> Result<(), WorkflowSaveError> {
    validate_journal_length(bytes.len())
}

fn validate_journal_length(length: usize) -> Result<(), WorkflowSaveError> {
    if length > MAX_SAVE_JOURNAL_BYTES {
        return Err(WorkflowSaveError::JournalTooLarge {
            actual: length,
            limit: MAX_SAVE_JOURNAL_BYTES,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use sim_nodes::PortDescriptor;
    use std::{fs, path::PathBuf};

    const LEGACY: &[u8] = br#"{
      "version": 0.4,
      "last_node_id": "02",
      "last_link_id": 1,
      "nodes": [
        {"id":"001","type":"Source","pos":[0,0],"size":[1,1],"inputs":[],"widgets_values":[[1,2]],"future":{"x":1}},
        {"id":"02","type":"Sink","pos":[0,0],"size":[1,1],"inputs":[{"name":"value","link":1}],"widgets_values":[]}
      ],
      "links": [[1,"001",0,"02",0,"VALUE"]],
      "groups": [], "config": {}, "extra": {"future": NaN}
    }"#;

    const SCHEMA_ONE_RECURSIVE: &[u8] = br#"{
      "version":1,
      "state":{"lastGroupId":1,"lastNodeId":1,"lastLinkId":1,"lastRerouteId":1,"future":true},
      "nodes":[{"id":"source","type":"Source","pos":{"0":0,"1":0},"size":[100,60],"flags":{"collapsed":false},"order":0,"mode":0,"inputs":[{"name":"value","type":["IMAGE","MASK"],"link":null,"slot_index":"0"}],"outputs":[{"name":"value","type":1,"links":[1],"slot_index":0}],"properties":{},"widgets_values":{"future":1}}],
      "groups":[{"id":1,"title":"Fixture","bounding":[0,0,120,80],"locked":false}],
      "links":[{"id":1,"origin_id":"source","origin_slot":"0","target_id":"source","target_slot":0,"type":["IMAGE","MASK"],"parentId":1}],
      "floatingLinks":[],
      "reroutes":[{"id":1,"pos":[50,20],"linkIds":[1],"floating":{"slotType":"output"}}],
      "subgraphs":[{"id":"instance","type":"00000000-0000-0000-0000-000000000002","pos":[0,0],"size":[100,60],"flags":{},"order":1,"mode":0,"inputs":[{"id":"00000000-0000-0000-0000-000000000010","name":"in","type":"IMAGE","linkIds":[1]}],"outputs":[]}],
      "definitions":{"subgraphs":[{
        "id":"00000000-0000-0000-0000-000000000002","revision":3,"version":1,
        "state":{"lastGroupId":0,"lastNodeId":0,"lastLinkId":0,"lastRerouteId":0},
        "name":"Outer","inputNode":{"id":-10,"bounding":[0,0,120,60],"pinned":true},"outputNode":{"id":-20,"bounding":[200,0,120,60]},
        "nodes":[],"links":[],
        "inputs":[{"id":"00000000-0000-0000-0000-000000000011","name":"in","type":"IMAGE"}],
        "outputs":[{"id":"00000000-0000-0000-0000-000000000012","name":"out","type":"IMAGE","linkIds":[1]}],
        "widgets":[{"id":"widget-1","name":"Strength"}],
        "definitions":{"subgraphs":[{
          "id":"00000000-0000-0000-0000-000000000003","revision":1,"version":1,
          "state":{"lastGroupId":0,"lastNodeId":0,"lastLinkId":0,"lastRerouteId":0},
          "name":"Inner","inputNode":{"id":-30,"bounding":[0,0,120,60]},"outputNode":{"id":-40,"bounding":[200,0,120,60]},
          "nodes":[],"links":[],"future":{"keep":1}
        }]},
        "future":{"keep":1}
      }]},
      "futureTop":"keep"
    }"#;

    const MIXED_LINKS: &[u8] = br#"{
      "version":1,
      "nodes":[
        {"id":"source","type":"Source","inputs":[],"outputs":[{"name":"value","type":"VALUE","links":[1,"two"]}],"widgets_values":[7]},
        {"id":"legacy-sink","type":"Sink","inputs":[{"name":"value","type":"VALUE","link":1}],"outputs":[],"widgets_values":[]},
        {"id":"schema-sink","type":"Sink","inputs":[{"name":"value","type":"VALUE","link":"two"}],"outputs":[],"widgets_values":[]}
      ],
      "links":[[1,"source","0","legacy-sink","0",7],{"id":"two","origin_id":"source","origin_slot":"0","target_id":"schema-sink","target_slot":"0","type":["VALUE"]}]
    }"#;

    const INVALID_SCHEMA_ONE_RECURSIVE: &[u8] = br#"{
      "version":1,
      "state":{"lastNodeId":"not-a-number"},
      "nodes":[],
      "links":[{"id":"bad","origin_id":1,"origin_slot":0,"target_id":2,"target_slot":0,"type":"IMAGE"}],
      "definitions":{"subgraphs":[{
        "id":"00000000-0000-0000-0000-000000000002","revision":1,"version":1,
        "state":{"lastGroupId":0,"lastNodeId":0,"lastLinkId":0,"lastRerouteId":0},
        "name":"Outer","inputNode":{"id":-10,"bounding":[0,0,120,60]},"outputNode":{"id":-20,"bounding":[200,0,120,60]},"nodes":[],
        "definitions":{"subgraphs":[{"id":"not-a-uuid","version":1,"nodes":[],"inputs":[{"id":"bad","name":1,"type":false,"linkIds":["bad"]}]}]}
      }]}
    }"#;

    fn descriptors() -> BTreeMap<String, NodeDescriptor> {
        BTreeMap::from([
            (
                "Source".to_owned(),
                NodeDescriptor {
                    type_name: "Source".to_owned(),
                    display_name: "Source".to_owned(),
                    inputs: vec![
                        PortDescriptor {
                            name: "literal".to_owned(),
                            type_name: "VALUE".to_owned(),
                            required: true,
                        },
                        PortDescriptor {
                            name: "second".to_owned(),
                            type_name: "VALUE".to_owned(),
                            required: true,
                        },
                    ],
                    outputs: vec![PortDescriptor {
                        name: "value".to_owned(),
                        type_name: "VALUE".to_owned(),
                        required: true,
                    }],
                },
            ),
            (
                "Sink".to_owned(),
                NodeDescriptor {
                    type_name: "Sink".to_owned(),
                    display_name: "Sink".to_owned(),
                    inputs: vec![PortDescriptor {
                        name: "value".to_owned(),
                        type_name: "VALUE".to_owned(),
                        required: true,
                    }],
                    outputs: vec![],
                },
            ),
        ])
    }

    #[test]
    fn lossless_workflow_preserves_exact_source_and_nonfinite_strings() {
        let document = WorkflowFormatDocument::parse(LEGACY).expect("legacy workflow");
        assert_eq!(document.original_bytes(), LEGACY);
        assert_eq!(document.serialized_bytes().expect("lossless bytes"), LEGACY);
        assert_eq!(document.non_finite_tokens().len(), 1);
        assert_eq!(document.value()["extra"]["future"], Value::Null);

        let quoted = import_json(
            br#"{"version":0.4,"nodes":[],"links":[],"text":"NaN Infinity -Infinity"}"#,
        )
        .expect("quoted tokens");
        let JsonImport::Workflow(quoted) = quoted else {
            panic!("expected workflow");
        };
        assert!(quoted.non_finite_tokens().is_empty());
        assert_eq!(quoted.value()["text"], "NaN Infinity -Infinity");
    }

    #[test]
    fn schema_one_round_trips_recursive_unknown_state_exactly() {
        let document =
            WorkflowFormatDocument::parse(SCHEMA_ONE_RECURSIVE).expect("schema one workflow");
        assert_eq!(document.format(), Some(WorkflowFormat::Schema1));
        assert!(document.validation_issues().is_empty());
        assert_eq!(
            document.serialized_bytes().expect("exact bytes"),
            SCHEMA_ONE_RECURSIVE
        );
        assert_eq!(document.value()["futureTop"], "keep");
        assert_eq!(
            document.value()["definitions"]["subgraphs"][0]["definitions"]["subgraphs"][0]["future"]
                ["keep"],
            1
        );
    }

    #[test]
    fn schema_one_validation_reports_required_recursive_paths() {
        let document = WorkflowFormatDocument::parse(INVALID_SCHEMA_ONE_RECURSIVE)
            .expect("schema one workflow");
        let paths = document
            .validation_issues()
            .iter()
            .map(|issue| issue.path.as_str())
            .collect::<BTreeSet<_>>();
        for expected in [
            "$.state.lastGroupId",
            "$.state.lastNodeId",
            "$.state.lastLinkId",
            "$.state.lastRerouteId",
            "$.links[0].id",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].id",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].revision",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].state",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].name",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].inputNode",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].outputNode",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].inputs[0].id",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].inputs[0].name",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].inputs[0].type",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].inputs[0].linkIds",
        ] {
            assert!(paths.contains(expected), "missing issue for {expected}");
        }
    }

    #[test]
    fn discriminator_preserves_templates_prompt_and_empty_object_edges() {
        assert!(matches!(
            import_json(br#"{"templates":[],"future":1}"#).expect("templates"),
            JsonImport::Templates(_)
        ));
        assert!(matches!(
            import_json(br#"{"001":{"class_type":"A","inputs":{},"future":true}}"#)
                .expect("prompt"),
            JsonImport::ApiPrompt(_)
        ));
        let JsonImport::Workflow(empty) = import_json(br#"{}"#).expect("empty workflow") else {
            panic!("expected workflow");
        };
        assert_eq!(empty.validation_issues()[0].path, "$.version");
    }

    #[test]
    fn embedded_metadata_priority_falls_through_without_execution() {
        let svg = br#"<svg><metadata><![CDATA[{"templates":{"templates":[{"workflow":{"version":0.4,"nodes":[],"links":[]}}]},"workflow":{"version":0.4,"nodes":[],"links":[],"chosen":"workflow"},"prompt":{"1":{"class_type":"A","inputs":{}}},"parameters":"cat\nSteps: 4, Seed: 1, Size: 64x64"}]]></metadata></svg>"#;
        let metadata = MetadataDocument::parse(
            svg,
            Some("fixture.svg"),
            None,
            sim_media::MetadataLimits::default(),
        )
        .expect("embedded SVG metadata");
        let imported = import_embedded_metadata(&metadata).expect("embedded import");
        assert_eq!(imported.templates.len(), 1);
        let EmbeddedPrimary::Workflow(workflow) = imported.primary else {
            panic!("workflow must have priority");
        };
        assert_eq!(workflow.value()["chosen"], "workflow");
        assert!(!imported.executes_on_import);

        let fallback = br#"<svg><metadata><![CDATA[{"workflow":"{","prompt":{"1":{"class_type":"A","inputs":{}}},"parameters":"cat\nSteps: 4, Seed: 1, Size: 64x64"}]]></metadata></svg>"#;
        let metadata = MetadataDocument::parse(
            fallback,
            Some("fallback.svg"),
            None,
            sim_media::MetadataLimits::default(),
        )
        .expect("fallback SVG metadata");
        let imported = import_embedded_metadata(&metadata).expect("fallback import");
        assert!(matches!(imported.primary, EmbeddedPrimary::ApiPrompt(_)));
        assert_eq!(imported.rejected.len(), 1);
        assert!(!imported.executes_on_import);
    }

    #[test]
    fn graph_projection_preserves_textual_ids_and_wraps_array_literals() {
        let workflow = WorkflowFormatDocument::parse(LEGACY).expect("workflow");
        let prompt =
            graph_to_prompt(&workflow, &descriptors(), "1.0.0").expect("prompt projection");
        assert!(prompt.prompt.0.contains_key(&NodeId("001".to_owned())));
        assert_eq!(
            prompt.prompt.0[&NodeId("001".to_owned())].inputs["literal"]["__value__"],
            json!([1, 2])
        );
        assert_eq!(
            prompt.prompt.0[&NodeId("02".to_owned())].inputs["value"],
            json!(["001", 0])
        );
        assert_eq!(
            prompt.extra_data["extra_pnginfo"]["workflow"]["nodes"][0]["future"]["x"],
            1
        );
    }

    #[test]
    fn graph_projection_accepts_legacy_and_schema_one_link_entries() {
        let workflow = WorkflowFormatDocument::parse(MIXED_LINKS).expect("mixed-link workflow");
        let prompt = graph_to_prompt(&workflow, &descriptors(), "fixture")
            .expect("mixed-link prompt projection");

        assert_eq!(
            prompt.prompt.0[&NodeId("legacy-sink".to_owned())].inputs["value"],
            json!(["source", 0])
        );
        assert_eq!(
            prompt.prompt.0[&NodeId("schema-sink".to_owned())].inputs["value"],
            json!(["source", 0])
        );
    }

    #[test]
    fn schema_one_link_projection_errors_have_stable_field_paths() {
        let invalid_slot = WorkflowFormatDocument::parse(
            br#"{"version":1,"nodes":[],"links":[{"id":1,"origin_id":"source","origin_slot":-1,"target_id":"sink","target_slot":0,"type":"VALUE"}]}"#,
        )
        .expect("invalid-slot workflow");
        assert_eq!(
            graph_to_prompt(&invalid_slot, &descriptors(), "fixture"),
            Err(WorkflowFormatError::PromptProjection {
                path: "$.links[0].origin_slot".to_owned(),
                reason: "origin slot must be an unsigned integer".to_owned(),
            })
        );

        let duplicate = WorkflowFormatDocument::parse(
            br#"{"version":1,"nodes":[],"links":[[1,"source",0,"sink",0,"VALUE"],{"id":1,"origin_id":"source","origin_slot":0,"target_id":"sink","target_slot":0,"type":"VALUE"}]}"#,
        )
        .expect("duplicate-link workflow");
        assert_eq!(
            graph_to_prompt(&duplicate, &descriptors(), "fixture"),
            Err(WorkflowFormatError::PromptProjection {
                path: "$.links[1].id".to_owned(),
                reason: "link ID is duplicated".to_owned(),
            })
        );
    }

    #[test]
    fn graph_projection_resolves_bypass_slots_and_explicit_virtual_nodes() {
        let bypass = WorkflowFormatDocument::parse(
            br#"{"version":0.4,"nodes":[{"id":"source","type":"Source","inputs":[],"outputs":[{"name":"value","type":"IMAGE","links":[1]}],"widgets_values":[]},{"id":"pass","type":"Pass","mode":4,"inputs":[{"name":"in","type":"IMAGE","link":1}],"outputs":[{"name":"out","type":"IMAGE","links":[2]}],"widgets_values":[]},{"id":"sink","type":"Sink","inputs":[{"name":"value","type":"IMAGE","link":2}],"outputs":[],"widgets_values":[]}],"links":[[1,"source",0,"pass",0,"IMAGE"],[2,"pass",0,"sink",0,"IMAGE"]]}"#,
        )
        .expect("bypass workflow");
        let prompt =
            graph_to_prompt(&bypass, &descriptors(), "fixture").expect("bypass prompt projection");
        assert!(!prompt.prompt.0.contains_key(&NodeId("pass".to_owned())));
        assert_eq!(
            prompt.prompt.0[&NodeId("sink".to_owned())].inputs["value"],
            json!(["source", 0])
        );

        let virtual_workflow = WorkflowFormatDocument::parse(
            br#"{"version":0.4,"nodes":[{"id":"virtual","type":"Virtual","properties":{"virtualNode":true},"virtual_prompt":{"inner":{"class_type":"Source","inputs":{"literal":7}}}}],"links":[]}"#,
        )
        .expect("virtual workflow");
        let prompt = graph_to_prompt(&virtual_workflow, &descriptors(), "fixture")
            .expect("virtual prompt projection");
        assert_eq!(
            prompt.prompt.0[&NodeId("inner".to_owned())].inputs["literal"],
            7
        );
        assert!(!prompt.prompt.0.contains_key(&NodeId("virtual".to_owned())));
    }

    #[test]
    fn controlled_values_are_bounded_and_partial_runs_do_not_advance() {
        assert_eq!(
            next_controlled_value(
                ControlledValue::Integer(5),
                ControlMode::Increment,
                Some((0, 5)),
                &[],
                4,
                false,
            ),
            ControlledValue::Integer(5)
        );
        assert_eq!(
            next_controlled_value(
                ControlledValue::Combo("b".to_owned()),
                ControlMode::Increment,
                None,
                &["a".to_owned(), "b".to_owned()],
                0,
                false,
            ),
            ControlledValue::Combo("a".to_owned())
        );
        assert_eq!(
            next_controlled_value(
                ControlledValue::Integer(2),
                ControlMode::Randomize,
                None,
                &[],
                99,
                true,
            ),
            ControlledValue::Integer(2)
        );
    }

    #[test]
    fn a1111_conversion_is_editable_and_never_executes_on_import() {
        let document = convert_a1111_parameters(
            "a cat\nNegative prompt: blur\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 9, Size: 512x768, Model: sd.safetensors",
        )
        .expect("A1111 conversion");
        assert_eq!(document.value()["version"], 0.4);
        assert_eq!(document.value()["nodes"].as_array().map(Vec::len), Some(7));
        assert_eq!(document.value()["nodes"][4]["widgets_values"][0], 9);

        let extended = convert_a1111_parameters(
            "a cat <lora:detail:0.8:0.6>\nNegative prompt: blur\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 9, Size: 512x768, Hires upscale: 2, Hires steps: 8, Denoising strength: 0.5, Clip skip: 2",
        )
        .expect("extended A1111 conversion");
        let node_types = extended.value()["nodes"]
            .as_array()
            .expect("nodes")
            .iter()
            .filter_map(|node| node["type"].as_str())
            .collect::<BTreeSet<_>>();
        assert!(node_types.contains("LoraLoader"));
        assert!(node_types.contains("CLIPSetLastLayer"));
        assert!(node_types.contains("LatentUpscaleBy"));
        assert_eq!(extended.value()["nodes"][1]["widgets_values"][0], "a cat");
    }

    #[test]
    fn graph_prompt_projection_uses_separate_native_prompt_widget_values() {
        let workflow = serde_json::to_vec(&json!({
            "version": 0.4,
            "nodes": [{
                "id": "source",
                "type": "Source",
                "inputs": [],
                "outputs": [],
                "widgets_values": ["workflow-second", "workflow-value"],
                "zed:native-widgets": {
                    "version": 1,
                    "widgets": [
                        {
                            "identifier": "second",
                            "kind": {"kind": "text", "multiline": false},
                            "value": "workflow-second",
                            "prompt_value": "prompt-second",
                            "validation": "valid",
                            "converted_to_input": false,
                            "visible": true
                        },
                        {
                            "identifier": "literal",
                            "kind": {"kind": "text", "multiline": false},
                            "value": "workflow-value",
                            "prompt_value": "prompt-value",
                            "validation": "valid",
                            "converted_to_input": false,
                            "visible": true
                        }
                    ]
                }
            }],
            "links": []
        }))
        .expect("serialize native widget workflow");
        let workflow = WorkflowFormatDocument::parse(&workflow).expect("parse workflow");
        let prompt = graph_to_prompt(&workflow, &descriptors(), "test").expect("project prompt");
        assert_eq!(
            prompt.prompt.0[&NodeId("source".to_owned())].inputs["literal"],
            "prompt-value"
        );
        assert_eq!(
            prompt.prompt.0[&NodeId("source".to_owned())].inputs["second"],
            "prompt-second"
        );

        let mut malformed = workflow.value().clone();
        malformed["nodes"][0]["zed:native-widgets"]["widgets"][0]
            .as_object_mut()
            .expect("native widget object")
            .remove("prompt_value");
        let malformed = WorkflowFormatDocument::parse(
            &serde_json::to_vec(&malformed).expect("serialize malformed workflow"),
        )
        .expect("parse malformed workflow");
        assert!(matches!(
            graph_to_prompt(&malformed, &descriptors(), "test"),
            Err(WorkflowFormatError::PromptProjection { path, .. }) if path.ends_with(".prompt_value")
        ));

        let mut duplicated = workflow.value().clone();
        duplicated["nodes"][0]["zed:native-widgets"]["widgets"][1]["identifier"] =
            Value::String("second".to_owned());
        let duplicated = WorkflowFormatDocument::parse(
            &serde_json::to_vec(&duplicated).expect("serialize duplicated widget workflow"),
        )
        .expect("parse duplicated widget workflow");
        assert!(matches!(
            graph_to_prompt(&duplicated, &descriptors(), "test"),
            Err(WorkflowFormatError::PromptProjection { path, .. }) if path.ends_with(".identifier")
        ));

        let mut unknown = workflow.value().clone();
        unknown["nodes"][0]["zed:native-widgets"]["widgets"][0]["identifier"] =
            Value::String("socket_only".to_owned());
        let unknown = WorkflowFormatDocument::parse(
            &serde_json::to_vec(&unknown).expect("serialize unknown widget workflow"),
        )
        .expect("parse unknown widget workflow");
        assert!(matches!(
            graph_to_prompt(&unknown, &descriptors(), "test"),
            Err(WorkflowFormatError::PromptProjection { path, .. }) if path.ends_with(".identifier")
        ));
    }

    #[test]
    fn locators_and_model_names_reject_ambient_paths() {
        let locator = SavedResultLocator {
            filename: "result.png".to_owned(),
            subfolder: "images/day".to_owned(),
            namespace: FileNamespace::Output,
            animated: Some(false),
            unknown: BTreeMap::from([("future".to_owned(), Value::Bool(true))]),
        };
        let key = locator
            .artifact_key("output")
            .expect("canonical locator key");
        assert_eq!(key.relative_path, PathBuf::from("images/day/result.png"));
        assert_eq!(locator.asset_namespace(), crate::AssetNamespace::Output);
        let traversal = SavedResultLocator {
            filename: "../escape.png".to_owned(),
            subfolder: String::new(),
            namespace: FileNamespace::Output,
            animated: None,
            unknown: BTreeMap::new(),
        };
        assert!(traversal.artifact_key("output").is_err());
        assert!(normalize_model_name("C:\\models\\x.safetensors").is_err());
        assert_eq!(
            normalize_model_name("checkpoints\\x.safetensors"),
            Ok("checkpoints/x.safetensors".to_owned())
        );
    }

    #[test]
    fn templates_keep_provenance_requirements_and_new_identity() {
        let JsonImport::Templates(template) = import_json(
            br#"{"templates":[{"workflow":{"version":0.4,"nodes":[{"id":1,"type":"MissingNode"}],"links":[]},"thumbnail":"thumb.png","models":[{"kind":"model","identifier":"x.safetensors","future":true}],"future":"keep"}]}"#,
        )
        .expect("template container")
        else {
            panic!("expected templates");
        };
        let identity = Uuid::from_u128(9);
        let instance = template
            .instantiate(
                0,
                TemplateProvenance::SignedPlugin {
                    plugin: "example.plugin".to_owned(),
                    digest: "sha256:1".to_owned(),
                },
                &BTreeSet::new(),
                identity,
            )
            .expect("template instance");
        assert_eq!(instance.document_identity, identity);
        assert_eq!(instance.missing_node_identifiers, ["MissingNode"]);
        assert_eq!(instance.requirements[0].unknown["future"], true);
        assert_eq!(instance.unknown["future"], "keep");
    }

    #[test]
    fn app_mode_restoration_is_explicit_and_reversible() {
        let workflow = WorkflowFormatDocument::parse(
            br#"{"version":0.4,"nodes":[],"links":[],"future":true}"#,
        )
        .expect("workflow");
        let port = AppModePort {
            identifier: "prompt".to_owned(),
            node_identifier: "7".to_owned(),
            port: "text".to_owned(),
            unknown: BTreeMap::from([("future".to_owned(), Value::Bool(true))]),
        };
        let mut locked = AppModeDocument::new(
            workflow.clone(),
            AppModeConfiguration {
                inputs: vec![port.clone()],
                outputs: Vec::new(),
                allow_restore_editing: false,
                unknown: BTreeMap::new(),
            },
        );
        assert!(!locked.restore_editing());
        assert!(locked.editing_hidden());

        let mut reversible = AppModeDocument::new(
            workflow,
            AppModeConfiguration {
                inputs: vec![port],
                outputs: Vec::new(),
                allow_restore_editing: true,
                unknown: BTreeMap::from([("future".to_owned(), json!(1))]),
            },
        );
        assert!(reversible.restore_editing());
        assert!(!reversible.editing_hidden());
        reversible.hide_editing();
        assert!(reversible.editing_hidden());
        assert_eq!(reversible.configuration().unknown["future"], 1);
        assert_eq!(reversible.workflow().value()["future"], true);
    }

    #[test]
    fn save_journal_is_bounded_base64_and_reload_is_non_destructive() {
        let base = vec![0, 1, 2, 254, 255];
        let mut coordinator =
            WorkflowSaveCoordinator::new("workflow.json", WorkflowStorageProvider::LocalFile, base)
                .expect("coordinator");
        coordinator.edit(vec![9, 8, 7]).expect("local edit");
        coordinator
            .observe_external_change(vec![6, 5, 4])
            .expect("external change");
        let encoded = coordinator.encode().expect("encoded journal");
        let encoded_text = std::str::from_utf8(&encoded).expect("journal UTF-8");
        assert!(encoded_text.contains("AAEC/v8="));
        assert!(!encoded_text.contains("[0,1,2,254,255]"));
        let recovered = WorkflowSaveCoordinator::decode(&encoded).expect("decoded journal");
        assert_eq!(recovered.local_bytes, vec![9, 8, 7]);
        let mut tampered: Value = serde_json::from_slice(&encoded).expect("journal JSON");
        tampered["external"]["revision"] = Value::String("corrupt".to_owned());
        let tampered = serde_json::to_vec(&tampered).expect("tampered journal");
        assert!(matches!(
            WorkflowSaveCoordinator::decode(&tampered),
            Err(WorkflowSaveError::InvalidState(_))
        ));

        coordinator.reload_external().expect("reload external");
        assert_eq!(coordinator.local_bytes, vec![6, 5, 4]);
        assert_eq!(coordinator.authority, WorkflowAuthority::InSync);
        assert!(matches!(
            validate_journal_length(MAX_SAVE_JOURNAL_BYTES + 1),
            Err(WorkflowSaveError::JournalTooLarge { .. })
        ));
    }

    #[test]
    fn save_journal_migrates_schema_one_without_external_missing_state() {
        let coordinator = WorkflowSaveCoordinator::new(
            "workflow.json",
            WorkflowStorageProvider::LocalFile,
            b"schema-one".to_vec(),
        )
        .expect("save coordinator");
        let mut document: Value =
            serde_json::from_slice(&coordinator.encode().expect("schema two journal encoding"))
                .expect("schema two journal JSON");
        document["schema_version"] = json!(1);
        document
            .as_object_mut()
            .expect("journal object")
            .remove("external_missing");

        let migrated = WorkflowSaveCoordinator::decode(
            &serde_json::to_vec(&document).expect("schema one journal encoding"),
        )
        .expect("schema one migration");
        assert_eq!(
            migrated.schema_version(),
            WorkflowSaveCoordinator::SCHEMA_VERSION
        );
        assert!(!migrated.external_missing());
        assert!(!migrated.missing_recreation_approved());
        assert_eq!(migrated.authority(), WorkflowAuthority::InSync);
        let migrated_document: Value =
            serde_json::from_slice(&migrated.encode().expect("migrated journal encoding"))
                .expect("migrated journal JSON");
        assert_eq!(
            migrated_document["schema_version"],
            json!(WorkflowSaveCoordinator::SCHEMA_VERSION)
        );
        assert_eq!(migrated_document["external_missing"], json!(false));
        assert_eq!(
            migrated_document["missing_recreation_approved"],
            json!(false)
        );
    }

    #[test]
    fn external_deletion_requires_resolution_before_recreate() {
        let base = br#"{"version":0.4,"nodes":[],"links":[]}"#.to_vec();
        let local = br#"{"version":0.4,"nodes":[],"links":[],"local":true}"#.to_vec();
        let mut clean = WorkflowSaveCoordinator::new(
            "workflow.json",
            WorkflowStorageProvider::LocalFile,
            base.clone(),
        )
        .expect("clean coordinator");

        clean
            .observe_external_deletion()
            .expect("observe external deletion");
        assert_eq!(clean.authority(), WorkflowAuthority::ExternalMissing);
        assert!(clean.external_missing());
        assert!(clean.external().is_none());
        assert!(clean.comparison().external.is_none());
        assert!(matches!(
            clean.reload_external(),
            Err(WorkflowSaveError::NoExternalVersion)
        ));
        assert!(matches!(
            clean.prepare_save(
                Uuid::from_u128(101),
                clean.base().revision.clone(),
                "workflow.json",
                false,
            ),
            Err(WorkflowSaveError::InvalidState(message))
                if message.contains("keep the local version")
        ));
        let recovered =
            WorkflowSaveCoordinator::decode(&clean.encode().expect("missing journal encoding"))
                .expect("missing journal recovery");
        assert_eq!(recovered.authority(), WorkflowAuthority::ExternalMissing);
        assert!(recovered.external_missing());

        clean
            .observe_external_change(base.clone())
            .expect("external file reappeared");
        assert_eq!(clean.authority(), WorkflowAuthority::InSync);
        assert!(!clean.external_missing());

        let mut dirty =
            WorkflowSaveCoordinator::new("workflow.json", WorkflowStorageProvider::LocalFile, base)
                .expect("dirty coordinator");
        dirty.edit(local.clone()).expect("local edit");
        dirty
            .observe_external_deletion()
            .expect("observe deletion with local edits");
        assert_eq!(dirty.authority(), WorkflowAuthority::Conflict);
        assert!(dirty.external_missing());
        dirty.keep_local().expect("keep local missing workflow");
        assert_eq!(dirty.authority(), WorkflowAuthority::LocalDirty);
        assert!(dirty.external_missing());
        assert!(dirty.missing_recreation_approved());
        assert_eq!(dirty.local_bytes(), local);

        let operation_id = Uuid::from_u128(102);
        let expected_revision = dirty.base().revision.clone();
        let prepared = dirty
            .prepare_save(
                operation_id,
                expected_revision.clone(),
                "workflow.json",
                false,
            )
            .expect("prepare missing workflow recreation");
        assert_eq!(prepared.bytes, local);
        dirty
            .commit_save(
                operation_id,
                expected_revision,
                ContentRevision::from_bytes(&local),
            )
            .expect("commit recreated workflow");
        assert_eq!(dirty.authority(), WorkflowAuthority::InSync);
        assert!(!dirty.external_missing());
        assert!(!dirty.missing_recreation_approved());
        assert_eq!(dirty.base().bytes, local);

        let mut draft = WorkflowSaveCoordinator::new(
            "draft",
            WorkflowStorageProvider::Draft,
            b"draft".to_vec(),
        )
        .expect("draft coordinator");
        assert!(matches!(
            draft.observe_external_deletion(),
            Err(WorkflowSaveError::InvalidState(_))
        ));
    }

    #[test]
    fn save_owner_transitions_reject_bypasses_and_remain_decodable() {
        assert!(matches!(
            WorkflowSaveCoordinator::new(" ", WorkflowStorageProvider::Draft, vec![1]),
            Err(WorkflowSaveError::InvalidState(_))
        ));
        assert!(matches!(
            WorkflowSaveCoordinator::new(
                "workflow",
                WorkflowStorageProvider::Provider {
                    identifier: " ".to_owned()
                },
                vec![1]
            ),
            Err(WorkflowSaveError::InvalidState(_))
        ));

        let base = br#"{"version":0.4,"nodes":[],"links":[]}"#.to_vec();
        let local = br#"{"version":0.4,"nodes":[],"links":[],"local":true}"#.to_vec();
        let mut coordinator =
            WorkflowSaveCoordinator::new("workflow.json", WorkflowStorageProvider::LocalFile, base)
                .expect("local coordinator");
        coordinator.edit(local.clone()).expect("local edit");
        let operation = Uuid::from_u128(91);
        let expected = coordinator.base().revision.clone();
        let first = coordinator
            .prepare_save(operation, expected.clone(), "workflow.json", false)
            .expect("prepare save");
        let replay = coordinator
            .prepare_save(operation, expected.clone(), "workflow.json", false)
            .expect("idempotent prepare");
        assert_eq!(replay, first);
        assert!(matches!(
            coordinator.prepare_save(Uuid::from_u128(92), expected, "other.json", false),
            Err(WorkflowSaveError::InvalidState(_))
        ));
        assert!(matches!(
            coordinator.commit_save(
                operation,
                ContentRevision("externally-changed".to_owned()),
                ContentRevision::from_bytes(&local),
            ),
            Err(WorkflowSaveError::Conflict { .. })
        ));
        assert_eq!(coordinator.authority(), WorkflowAuthority::Interrupted);
        let recovered = WorkflowSaveCoordinator::decode(
            &coordinator.encode().expect("encodable interrupted state"),
        )
        .expect("decode interrupted state");
        assert_eq!(recovered.authority(), WorkflowAuthority::Interrupted);

        let mut detached = WorkflowSaveCoordinator::new(
            "workflow.json",
            WorkflowStorageProvider::LocalFile,
            local.clone(),
        )
        .expect("detached coordinator");
        detached
            .retarget_local_file("renamed.json")
            .expect("retarget after rename");
        assert_eq!(detached.document_identity(), "renamed.json");
        detached
            .detach_local_file_to_draft("draft-identity")
            .expect("detach deleted file");
        assert_eq!(detached.provider(), &WorkflowStorageProvider::Draft);
        assert_eq!(detached.authority(), WorkflowAuthority::LocalDirty);
        assert!(detached.external().is_none());
        assert!(detached.prepared().is_none());
        WorkflowSaveCoordinator::decode(&detached.encode().expect("detached journal"))
            .expect("decode detached journal");

        let revision = ContentRevision::from_bytes(&local);
        let mut switched =
            WorkflowSaveCoordinator::new("draft", WorkflowStorageProvider::Draft, local)
                .expect("committed coordinator");
        assert_eq!(switched.base().revision, revision);
        switched
            .switch_provider_after_committed_save(WorkflowStorageProvider::LocalFile)
            .expect("switch committed provider");
        assert_eq!(switched.provider(), &WorkflowStorageProvider::LocalFile);
    }

    #[test]
    fn val_recovery_002() {
        let base = br#"{"version":0.4,"nodes":[],"links":[]}"#.to_vec();
        let mut coordinator = WorkflowSaveCoordinator::new(
            "workflow.json",
            WorkflowStorageProvider::LocalFile,
            base.clone(),
        )
        .expect("save coordinator");
        let local = br#"{"version":0.4,"nodes":[],"links":[],"local":true}"#.to_vec();
        let external = br#"{"version":0.4,"nodes":[],"links":[],"external":true}"#.to_vec();
        coordinator.edit(local.clone()).expect("local edit");
        coordinator
            .observe_external_change(external.clone())
            .expect("external change");
        assert_eq!(coordinator.authority, WorkflowAuthority::Conflict);
        let dirty_external_conflict = coordinator.authority == WorkflowAuthority::Conflict;
        let comparison = coordinator.comparison();
        assert_eq!(comparison.base, base);
        assert_eq!(comparison.local, local);
        assert_eq!(comparison.external, Some(external.as_slice()));
        let comparison_preserved = comparison.base == base
            && comparison.local == local
            && comparison.external == Some(external.as_slice());
        let conflict = coordinator.prepare_save(
            Uuid::from_u128(1),
            ContentRevision::from_bytes(&external),
            "workflow.json",
            false,
        );
        assert!(matches!(conflict, Err(WorkflowSaveError::Conflict { .. })));
        let no_silent_overwrite = matches!(conflict, Err(WorkflowSaveError::Conflict { .. }))
            && coordinator.local_bytes == local
            && coordinator
                .external
                .as_ref()
                .is_some_and(|version| version.bytes == external);
        coordinator.keep_local().expect("keep local");
        assert_eq!(coordinator.authority, WorkflowAuthority::LocalDirty);
        let kept_local = coordinator.authority == WorkflowAuthority::LocalDirty
            && coordinator.local_bytes == local
            && coordinator.base.bytes == external;

        let mut reload_coordinator = WorkflowSaveCoordinator::new(
            "reload.json",
            WorkflowStorageProvider::LocalFile,
            base.clone(),
        )
        .expect("reload coordinator");
        reload_coordinator
            .observe_external_change(external.clone())
            .expect("external reload candidate");
        assert_eq!(
            reload_coordinator.authority,
            WorkflowAuthority::ExternalNewer
        );
        reload_coordinator
            .reload_external()
            .expect("reload external");
        assert_eq!(reload_coordinator.local_bytes, external);
        assert_eq!(reload_coordinator.authority, WorkflowAuthority::InSync);

        let mut autosave = WorkflowSaveCoordinator::new(
            "autosave.json",
            WorkflowStorageProvider::Draft,
            base.clone(),
        )
        .expect("autosave coordinator");
        autosave.edit(local.clone()).expect("autosave edit");
        let autosave_operation = Uuid::from_u128(4);
        let autosave_expected = autosave.base.revision.clone();
        autosave
            .prepare_save(
                autosave_operation,
                autosave_expected.clone(),
                "autosave.json",
                false,
            )
            .expect("autosave prepare");
        let autosave_revision = ContentRevision::from_bytes(&local);
        autosave
            .commit_save(autosave_operation, autosave_expected, autosave_revision)
            .expect("autosave commit");
        assert_eq!(autosave.authority, WorkflowAuthority::InSync);
        assert_eq!(autosave.base.bytes, local);

        let mut autosave_race = WorkflowSaveCoordinator::new(
            "autosave-race.json",
            WorkflowStorageProvider::Draft,
            base.clone(),
        )
        .expect("autosave race coordinator");
        autosave_race
            .edit(local.clone())
            .expect("autosave race edit");
        let race_operation = Uuid::from_u128(5);
        let race_expected = autosave_race.base.revision.clone();
        autosave_race
            .prepare_save(
                race_operation,
                race_expected.clone(),
                "autosave-race.json",
                false,
            )
            .expect("autosave race prepare");
        autosave_race
            .observe_external_change(external.clone())
            .expect("external change during autosave");
        assert_eq!(autosave_race.authority, WorkflowAuthority::Conflict);
        assert!(matches!(
            autosave_race.commit_save(
                race_operation,
                race_expected,
                ContentRevision::from_bytes(&local),
            ),
            Err(WorkflowSaveError::UnknownOperation(operation)) if operation == race_operation
        ));
        assert_eq!(autosave_race.local_bytes, local);
        assert_eq!(
            autosave_race
                .external
                .as_ref()
                .map(|version| &version.bytes),
            Some(&external)
        );

        let operation = Uuid::from_u128(2);
        let expected = coordinator.base.revision.clone();
        coordinator
            .prepare_save(operation, expected.clone(), "workflow-copy.json", true)
            .expect("save copy preparation");
        let encoded = coordinator.encode().expect("journal encoding");
        let journal: Value = serde_json::from_slice(&encoded).expect("journal JSON");
        let journal_uses_base64 = journal["base"]["bytes"].is_string()
            && journal["local_bytes"].is_string()
            && journal["prepared"]["bytes"].is_string();
        let recovered = WorkflowSaveCoordinator::decode(&encoded).expect("journal recovery");
        assert_eq!(recovered.authority, WorkflowAuthority::Interrupted);
        assert_eq!(recovered.local_bytes, local);
        let restart_interrupted_prepare = recovered.authority == WorkflowAuthority::Interrupted;
        let restart_preserved_local = recovered.local_bytes == local;

        let mut committed = recovered;
        committed
            .prepare_save(operation, expected.clone(), "workflow-copy.json", true)
            .expect("retry save copy");
        let revision = ContentRevision::from_bytes(&committed.local_bytes);
        committed
            .commit_save(operation, expected, revision)
            .expect("save copy commit");
        assert_eq!(committed.authority, WorkflowAuthority::InSync);
        assert_eq!(committed.document_identity, "workflow-copy.json");
        let saved_copy = committed.authority == WorkflowAuthority::InSync
            && committed.document_identity == "workflow-copy.json"
            && committed.base.bytes == local;

        let mut owner = WorkflowSaveCoordinator::new(
            "owner.json",
            WorkflowStorageProvider::LocalFile,
            base.clone(),
        )
        .expect("owner coordinator");
        owner.edit(local.clone()).expect("owner local edit");
        let owner_operation = Uuid::from_u128(6);
        let owner_expected = owner.base().revision.clone();
        let owner_prepared = owner
            .prepare_save(owner_operation, owner_expected.clone(), "owner.json", false)
            .expect("owner prepare");
        let owner_idempotent_prepare = owner
            .prepare_save(owner_operation, owner_expected.clone(), "owner.json", false)
            .is_ok_and(|replay| replay == owner_prepared);
        let owner_rejects_competing_prepare = matches!(
            owner.prepare_save(Uuid::from_u128(7), owner_expected, "other.json", false,),
            Err(WorkflowSaveError::InvalidState(_))
        );
        let owner_conflict_is_recoverable = matches!(
            owner.commit_save(
                owner_operation,
                ContentRevision("changed".to_owned()),
                ContentRevision::from_bytes(&local),
            ),
            Err(WorkflowSaveError::Conflict { .. })
        ) && owner.authority()
            == WorkflowAuthority::Interrupted
            && owner
                .encode()
                .and_then(|bytes| WorkflowSaveCoordinator::decode(&bytes).map(|_| bytes))
                .is_ok();
        let mut detached = WorkflowSaveCoordinator::new(
            "detached.json",
            WorkflowStorageProvider::LocalFile,
            local.clone(),
        )
        .expect("detached coordinator");
        detached
            .detach_local_file_to_draft("draft-identity")
            .expect("detach to draft");
        let owner_detach_is_checked = detached.provider() == &WorkflowStorageProvider::Draft
            && detached.authority() == WorkflowAuthority::LocalDirty
            && detached.prepared().is_none()
            && detached.external().is_none()
            && detached
                .encode()
                .and_then(|bytes| WorkflowSaveCoordinator::decode(&bytes).map(|_| bytes))
                .is_ok();

        let cases = json!({
            "dirty_external_conflict": dirty_external_conflict,
            "compare": comparison_preserved,
            "no_silent_overwrite": no_silent_overwrite,
            "keep": kept_local,
            "reload": reload_coordinator.local_bytes == external,
            "save_copy": saved_copy,
            "autosave_commit": autosave.base.bytes == local,
            "autosave_external_race_conflicts": autosave_race.authority == WorkflowAuthority::Conflict
                && autosave_race.local_bytes == local
                && autosave_race.external.as_ref().map(|version| &version.bytes) == Some(&external),
            "restart_interrupts_prepare": restart_interrupted_prepare,
            "local_bytes_preserved": restart_preserved_local,
            "journal_base64": journal_uses_base64,
            "owner_idempotent_prepare": owner_idempotent_prepare,
            "owner_rejects_competing_prepare": owner_rejects_competing_prepare,
            "owner_conflict_is_recoverable": owner_conflict_is_recoverable,
            "owner_detach_is_checked": owner_detach_is_checked,
        });
        assert!(
            cases
                .as_object()
                .is_some_and(|cases| cases.values().all(|value| value == &Value::Bool(true))),
            "{cases}"
        );
        let artifact = json!({
            "validation": "VAL-RECOVERY-002",
            "scope": "workflow-save-coordinator-stage",
            "fixture_sha256": {
                "base": format!("{:x}", Sha256::digest(&base)),
                "local": format!("{:x}", Sha256::digest(&local)),
                "external": format!("{:x}", Sha256::digest(&external)),
            },
            "environment": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "backend": "native-rust"},
            "cases": cases,
            "skipped": [],
        });
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("target")
            });
        let directory = target.join("sim-parity");
        fs::create_dir_all(&directory).expect("artifact directory");
        fs::write(
            directory.join("val-recovery-002.json"),
            serde_json::to_vec_pretty(&artifact).expect("artifact JSON"),
        )
        .expect("artifact write");
    }

    #[test]
    fn val_domain_002() {
        let workflow = WorkflowFormatDocument::parse(LEGACY).expect("legacy workflow");
        let projected =
            graph_to_prompt(&workflow, &descriptors(), "fixture").expect("prompt projection");
        let schema_one = WorkflowFormatDocument::parse(SCHEMA_ONE_RECURSIVE)
            .expect("valid recursive schema one fixture");
        let invalid_schema_one = WorkflowFormatDocument::parse(INVALID_SCHEMA_ONE_RECURSIVE)
            .expect("invalid recursive schema one fixture remains inspectable");
        let invalid_paths = invalid_schema_one
            .validation_issues()
            .iter()
            .map(|issue| issue.path.as_str())
            .collect::<BTreeSet<_>>();
        let mixed_links = WorkflowFormatDocument::parse(MIXED_LINKS)
            .expect("mixed legacy and object link fixture");
        let mixed_projection = graph_to_prompt(&mixed_links, &descriptors(), "fixture")
            .expect("mixed link prompt projection");
        let mixed_link_types =
            project_links(mixed_links.value().get("links")).expect("mixed link data types");
        let embedded_svg = br#"<svg><metadata><![CDATA[{"templates":{"templates":[]},"workflow":{"version":0.4,"nodes":[],"links":[],"chosen":true},"prompt":{"1":{"class_type":"A","inputs":{}}}}]]></metadata></svg>"#;
        let embedded_metadata = MetadataDocument::parse(
            embedded_svg,
            Some("fixture.svg"),
            None,
            sim_media::MetadataLimits::default(),
        )
        .expect("embedded fixture");
        let embedded = import_embedded_metadata(&embedded_metadata).expect("embedded import");
        let embedded_priority = matches!(
            &embedded.primary,
            EmbeddedPrimary::Workflow(workflow) if workflow.value()["chosen"] == true
        ) && !embedded.executes_on_import;
        let reroute_legacy = include_str!(
            "../../sim_test_support/fixtures/frontend_reroute/legacy/single_connected.json"
        );
        let reroute_native: Value = serde_json::from_str(include_str!(
            "../../sim_test_support/fixtures/frontend_reroute/native/single_connected.json"
        ))
        .expect("native reroute fixture");
        let reroute_document =
            WorkflowFormatDocument::parse(reroute_legacy).expect("legacy reroute fixture");
        let reroute_migrated = crate::workflow_migrations::apply_workflow_migrations(
            &reroute_document,
            &[crate::workflow_migrations::WorkflowMigrationId::LegacyRerouteNative],
        )
        .expect("reroute fixture migration");
        let a1111 = convert_a1111_parameters(
            "cat\nNegative prompt: blur\nSteps: 4, Sampler: Euler, CFG scale: 7, Seed: 1, Size: 64x64",
        )
        .expect("A1111 fixture");
        let invalid_model_source = br#"{"version":1,"state":{"lastGroupId":0,"lastNodeId":0,"lastLinkId":0,"lastRerouteId":0},"nodes":[],"links":[],"models":[{"name":"model","url":"not a url","directory":"checkpoints"}]}"#;
        let invalid_model = WorkflowFormatDocument::parse(invalid_model_source)
            .expect("inspectable invalid model fixture");
        let native_reroute_source = br#"{"version":0.4,"nodes":[{"id":1,"type":"Reroute"}],"links":[],"extra":{"reroutes":[{"id":1,"pos":[0,0]}]}}"#;
        let native_reroute =
            WorkflowFormatDocument::parse(native_reroute_source).expect("native reroute fixture");
        let native_reroute_is_not_candidate =
            crate::workflow_migrations::detect_workflow_migrations(&native_reroute)
                .iter()
                .all(|candidate| {
                    candidate.identifier
                        != crate::workflow_migrations::WorkflowMigrationId::LegacyRerouteNative
                });
        let primitive_boolean_source = br#"{"version":0.4,"nodes":[{"id":1,"type":"PrimitiveNode","widgets_values":[true]}],"links":[]}"#;
        let primitive_boolean = WorkflowFormatDocument::parse(primitive_boolean_source)
            .expect("boolean primitive fixture");
        let primitive_boolean_preserved =
            crate::workflow_migrations::detect_workflow_migrations(&primitive_boolean)
                .iter()
                .all(|candidate| {
                    candidate.identifier
                        != crate::workflow_migrations::WorkflowMigrationId::KSamplerWidgetValues
                });
        let renderer_source = br#"{"version":9,"nodes":[{"id":1,"type":"A","pos":{"0":100,"1":100},"size":[120,60]}],"links":[],"groups":[{"title":"G","bounding":[220,220,120,60]}],"extra":{"workflowRendererVersion":"Vue","reroutes":[{"id":1,"pos":{"0":340,"1":340}}]}}"#;
        let renderer =
            WorkflowFormatDocument::parse(renderer_source).expect("renderer compatibility fixture");
        let renderer = crate::workflow_migrations::apply_workflow_migrations(
            &renderer,
            &[crate::workflow_migrations::WorkflowMigrationId::RendererLayoutScale],
        )
        .expect("renderer compatibility migration");
        let draft_v1 = crate::workflow_migrations::DraftStoreV1 {
            drafts: BTreeMap::from([(
                String::new(),
                crate::workflow_migrations::DraftV1 {
                    data: "{}".to_owned(),
                    updated_at: 1,
                    name: "empty-path".to_owned(),
                    is_temporary: true,
                },
            )]),
            order: vec![String::new()],
            open_paths: vec!["workflow.json".to_owned()],
            active_index: 9,
            unknown: BTreeMap::new(),
        };
        let draft_v2 =
            crate::workflow_migrations::migrate_draft_store_v1_to_v2(&draft_v1, Some("client"));
        let nonfinite_token_is_source_derived =
            workflow.non_finite_tokens().first().and_then(|token| {
                workflow
                    .original_bytes()
                    .get(token.byte_offset..token.byte_offset.checked_add(token.source_length)?)
            }) == Some(b"NaN".as_slice());
        let recursive_paths = [
            "$.state.lastGroupId",
            "$.state.lastNodeId",
            "$.state.lastLinkId",
            "$.state.lastRerouteId",
            "$.links[0].id",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].id",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].revision",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].state",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].inputs[0].id",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].inputs[0].name",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].inputs[0].type",
            "$.definitions.subgraphs[0].definitions.subgraphs[0].inputs[0].linkIds",
        ];
        let cases = json!({
            "schema_04_exact_bytes": workflow.serialized_bytes().expect("bytes") == LEGACY,
            "schema_1_exact_bytes": schema_one.serialized_bytes().expect("schema one bytes") == SCHEMA_ONE_RECURSIVE,
            "schema_1_strict_valid": schema_one.validation_issues().is_empty(),
            "schema_1_unknown_fields": schema_one.value()["futureTop"] == "keep"
                && schema_one.value()["definitions"]["subgraphs"][0]["definitions"]["subgraphs"][0]["future"]["keep"] == 1,
            "schema_1_recursive_invalid_paths": recursive_paths.iter().all(|path| invalid_paths.contains(path)),
            "schema_1_invalid_original_preserved": invalid_schema_one.serialized_bytes().expect("invalid schema one bytes") == INVALID_SCHEMA_ONE_RECURSIVE,
            "unknown_fields": workflow.value()["nodes"][0]["future"]["x"] == 1,
            "textual_ids": projected.prompt.0.contains_key(&NodeId("001".to_owned())),
            "nonfinite_normalization": workflow.value()["extra"]["future"].is_null()
                && nonfinite_token_is_source_derived,
            "prompt_projection": projected.prompt.0.len() == 2,
            "mixed_array_object_link_projection": mixed_projection.prompt.0[&NodeId("legacy-sink".to_owned())].inputs["value"] == json!(["source", 0])
                && mixed_projection.prompt.0[&NodeId("schema-sink".to_owned())].inputs["value"] == json!(["source", 0]),
            "string_slot_projection": mixed_projection.prompt.0[&NodeId("legacy-sink".to_owned())].inputs["value"][1] == 0,
            "data_type_union_projection": mixed_link_types["n:1"].type_name == "7"
                && mixed_link_types["s:two"].type_name == "[\"VALUE\"]",
            "a1111_conversion": a1111.value()["nodes"].as_array().map(Vec::len) == Some(7)
                && a1111.validation_issues().is_empty(),
            "embedded_priority": embedded_priority,
            "legacy_reroute_source_fixture": reroute_migrated.value() == &reroute_native,
            "native_reroute_suppresses_legacy_migration": native_reroute_is_not_candidate,
            "primitive_boolean_is_not_control_migration": primitive_boolean_preserved,
            "renderer_serialized_geometry": renderer.value()["nodes"][0]["size"] == json!([100, 50])
                && renderer.value()["groups"][0]["bounding"] == json!([200, 200, 100, 50])
                && renderer.value()["extra"]["reroutes"][0]["pos"] == json!([300, 300]),
            "draft_hash_and_session_adapter": draft_v2.order == ["811c9dc5"]
                && draft_v2.session_client_id.as_deref() == Some("client")
                && draft_v2.session_active_index == 0,
            "invalid_model_url_has_stable_path": invalid_model.validation_issues().iter()
                .any(|issue| issue.path == "$.models[0].url"),
            "malformed_visible": WorkflowFormatDocument::parse(b"{").is_err(),
            "oversized_rejected": WorkflowFormatDocument::parse(vec![b' '; MAX_WORKFLOW_BYTES + 1]).is_err(),
            "import_never_executes": !embedded.executes_on_import,
        });
        assert!(
            cases
                .as_object()
                .is_some_and(|cases| cases.values().all(|value| value == &Value::Bool(true)))
        );
        let fixture_digests = json!({
            "legacy": format!("{:x}", Sha256::digest(LEGACY)),
            "schema_one_recursive": format!("{:x}", Sha256::digest(SCHEMA_ONE_RECURSIVE)),
            "schema_one_recursive_invalid": format!("{:x}", Sha256::digest(INVALID_SCHEMA_ONE_RECURSIVE)),
            "mixed_links": format!("{:x}", Sha256::digest(MIXED_LINKS)),
            "embedded_svg": format!("{:x}", Sha256::digest(embedded_svg)),
            "legacy_reroute": format!("{:x}", Sha256::digest(reroute_legacy.as_bytes())),
            "invalid_model": format!("{:x}", Sha256::digest(invalid_model_source)),
            "native_reroute": format!("{:x}", Sha256::digest(native_reroute_source)),
            "primitive_boolean": format!("{:x}", Sha256::digest(primitive_boolean_source)),
            "renderer": format!("{:x}", Sha256::digest(renderer_source)),
        });
        let artifact = json!({
            "validation": "VAL-DOMAIN-002",
            "scope": "workflow-and-embedded-metadata-stage",
            "fixture_sha256": fixture_digests,
            "environment": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "backend": "native-rust"},
            "cases": cases,
            "skipped": [],
        });
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("target")
            });
        let directory = target.join("sim-parity");
        fs::create_dir_all(&directory).expect("artifact directory");
        fs::write(
            directory.join("val-domain-002.json"),
            serde_json::to_vec_pretty(&artifact).expect("artifact JSON"),
        )
        .expect("artifact write");
    }

    #[test]
    fn deterministic_lossless_identifier_and_unknown_field_matrix() {
        for index in 0..256u16 {
            let identifier = format!("{index:04}");
            let source = format!(
                "{{\"version\":0.4,\"nodes\":[{{\"id\":\"{identifier}\",\"type\":\"A\",\"future_{index}\":{{\"value\":{index}}}}}],\"links\":[],\"unknown\":\"{identifier}\"}}"
            );
            let document = WorkflowFormatDocument::parse(source.as_bytes())
                .expect("deterministic workflow fixture");
            assert_eq!(
                document.serialized_bytes().expect("lossless serialization"),
                source.as_bytes()
            );
            assert_eq!(document.value()["nodes"][0]["id"], identifier);
            assert_eq!(document.value()["unknown"], identifier);
        }
    }
}
