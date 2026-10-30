use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use thiserror::Error;
use uuid::Uuid;

use crate::{JsonImport, WorkflowFormat, WorkflowFormatDocument, WorkflowFormatError, import_json};

pub const GRAPH_DOCUMENT_SCHEMA_VERSION: u16 = 1;
pub const GRAPH_CLIPBOARD_SCHEMA_VERSION: u16 = 1;
pub const MAX_GRAPH_HISTORY_ENTRIES: usize = 256;
pub const MAX_GRAPH_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_GRAPH_CLIPBOARD_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES: usize = 16 * 1024 * 1024;
pub const BLUEPRINT_DESCRIPTION_FIELD: &str = "BlueprintDescription";
pub const BLUEPRINT_SEARCH_ALIASES_FIELD: &str = "BlueprintSearchAliases";
const MIN_VIEWPORT_SCALE: f32 = 0.1;
const MAX_VIEWPORT_SCALE: f32 = 8.0;
const MAX_GRAPH_DEPTH: usize = 64;
pub(crate) const NATIVE_WIDGETS_FIELD: &str = "zed:native-widgets";
const NATIVE_WIDGETS_VERSION: u16 = 1;
const EXPOSED_WIDGETS_FIELD: &str = "zed:exposed-widgets";

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct NativeWidgetsEnvelope {
    version: u16,
    widgets: Vec<GraphWidget>,
    #[serde(flatten)]
    unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GraphIdentifier {
    Integer(i64),
    String(String),
}

impl GraphIdentifier {
    pub fn text(&self) -> String {
        match self {
            Self::Integer(value) => value.to_string(),
            Self::String(value) => value.clone(),
        }
    }

    fn from_value(value: &Value) -> Option<Self> {
        if let Some(value) = value.as_i64() {
            Some(Self::Integer(value))
        } else {
            value.as_str().map(|value| Self::String(value.to_owned()))
        }
    }
}

impl From<&str> for GraphIdentifier {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphPoint {
    pub x: f32,
    pub y: f32,
}

impl GraphPoint {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn translated(self, delta: Self) -> Self {
        Self {
            x: self.x + delta.x,
            y: self.y + delta.y,
        }
    }

    fn validate(self) -> Result<(), GraphError> {
        if self.x.is_finite() && self.y.is_finite() {
            Ok(())
        } else {
            Err(GraphError::NonFiniteGeometry)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphSize {
    pub width: f32,
    pub height: f32,
}

impl Default for GraphSize {
    fn default() -> Self {
        Self {
            width: 240.0,
            height: 120.0,
        }
    }
}

impl GraphSize {
    fn validate(self) -> Result<(), GraphError> {
        if self.width.is_finite()
            && self.height.is_finite()
            && self.width > 0.0
            && self.height > 0.0
        {
            Ok(())
        } else {
            Err(GraphError::InvalidSize)
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphRect {
    pub origin: GraphPoint,
    pub size: GraphSize,
}

impl GraphRect {
    pub fn contains(self, point: GraphPoint) -> bool {
        point.x >= self.origin.x
            && point.y >= self.origin.y
            && point.x <= self.origin.x + self.size.width
            && point.y <= self.origin.y + self.size.height
    }

    pub fn intersects(self, other: Self) -> bool {
        self.origin.x <= other.origin.x + other.size.width
            && self.origin.x + self.size.width >= other.origin.x
            && self.origin.y <= other.origin.y + other.size.height
            && self.origin.y + self.size.height >= other.origin.y
    }

    fn validate(self) -> Result<(), GraphError> {
        self.origin.validate()?;
        self.size.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphPortType {
    Any,
    Concrete(String),
    Union(BTreeSet<String>),
    Virtual(BTreeSet<String>),
}

impl GraphPortType {
    pub fn from_name(type_name: &str) -> Self {
        let type_name = type_name.trim();
        if type_name.is_empty() || type_name == "*" || type_name == "ANY" {
            return Self::Any;
        }
        if let Some(values) = type_name
            .strip_prefix("virtual(")
            .and_then(|values| values.strip_suffix(')'))
        {
            return Self::Virtual(
                values
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned)
                    .collect(),
            );
        }
        let values = type_name
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect::<BTreeSet<_>>();
        if values.len() > 1 {
            Self::Union(values)
        } else {
            Self::Concrete(type_name.to_owned())
        }
    }

    pub fn accepts(&self, output: &Self) -> bool {
        if matches!(self, Self::Any) || matches!(output, Self::Any) {
            return true;
        }
        let input_types = self.concrete_types();
        let output_types = output.concrete_types();
        !input_types.is_disjoint(&output_types)
    }

    pub fn display_name(&self) -> String {
        match self {
            Self::Any => "ANY".to_owned(),
            Self::Concrete(value) => value.clone(),
            Self::Union(values) => values.iter().cloned().collect::<Vec<_>>().join(","),
            Self::Virtual(values) => format!(
                "virtual({})",
                values.iter().cloned().collect::<Vec<_>>().join(",")
            ),
        }
    }

    fn concrete_types(&self) -> BTreeSet<String> {
        match self {
            Self::Any => BTreeSet::new(),
            Self::Concrete(value) => BTreeSet::from([value.clone()]),
            Self::Union(values) | Self::Virtual(values) => values.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GraphPort {
    pub name: String,
    pub port_type: GraphPortType,
    pub multiple: bool,
    pub dynamic: bool,
    #[serde(default)]
    pub unknown: BTreeMap<String, Value>,
}

impl GraphPort {
    pub fn new(name: impl Into<String>, port_type: GraphPortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            multiple: false,
            dynamic: false,
            unknown: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum GraphWidgetKind {
    Boolean,
    Integer {
        minimum: i64,
        maximum: i64,
        step: i64,
    },
    Float {
        minimum: f64,
        maximum: f64,
        step: f64,
    },
    Text {
        multiline: bool,
    },
    Combo {
        values: Vec<String>,
        dynamic: bool,
    },
    Preserved {
        schema: Value,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WidgetValidation {
    Valid,
    Invalid(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphWidget {
    pub identifier: String,
    pub kind: GraphWidgetKind,
    pub value: Value,
    pub prompt_value: Value,
    pub validation: WidgetValidation,
    pub converted_to_input: bool,
    pub visible: bool,
    #[serde(default)]
    pub unknown: BTreeMap<String, Value>,
}

impl GraphWidget {
    pub fn preserved(identifier: impl Into<String>, value: Value) -> Self {
        Self {
            identifier: identifier.into(),
            kind: GraphWidgetKind::Preserved {
                schema: value.clone(),
            },
            prompt_value: value.clone(),
            value,
            validation: WidgetValidation::Valid,
            converted_to_input: false,
            visible: true,
            unknown: BTreeMap::new(),
        }
    }

    pub fn normalize(&self, value: Value) -> Result<Value, GraphError> {
        self.validate_schema()?;
        match &self.kind {
            GraphWidgetKind::Boolean => value
                .as_bool()
                .map(Value::Bool)
                .ok_or_else(|| GraphError::InvalidWidgetValue(self.identifier.clone())),
            GraphWidgetKind::Integer {
                minimum,
                maximum,
                step,
            } => {
                let value = value
                    .as_i64()
                    .ok_or_else(|| GraphError::InvalidWidgetValue(self.identifier.clone()))?;
                let step = (*step).max(1);
                let clamped = value.clamp(*minimum, *maximum);
                let rounded = minimum.saturating_add(
                    clamped
                        .saturating_sub(*minimum)
                        .checked_div(step)
                        .unwrap_or_default()
                        .saturating_mul(step),
                );
                Ok(Value::from(rounded.clamp(*minimum, *maximum)))
            }
            GraphWidgetKind::Float {
                minimum,
                maximum,
                step,
            } => {
                let value = value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| GraphError::InvalidWidgetValue(self.identifier.clone()))?;
                let clamped = value.clamp(*minimum, *maximum);
                let rounded = if step.is_finite() && *step > 0.0 {
                    minimum + ((clamped - minimum) / step).round() * step
                } else {
                    clamped
                };
                serde_json::Number::from_f64(rounded.clamp(*minimum, *maximum))
                    .map(Value::Number)
                    .ok_or_else(|| GraphError::InvalidWidgetValue(self.identifier.clone()))
            }
            GraphWidgetKind::Text { .. } => value
                .as_str()
                .map(|value| Value::String(value.to_owned()))
                .ok_or_else(|| GraphError::InvalidWidgetValue(self.identifier.clone())),
            GraphWidgetKind::Combo { values, dynamic } => {
                let value = value
                    .as_str()
                    .ok_or_else(|| GraphError::InvalidWidgetValue(self.identifier.clone()))?;
                if *dynamic || values.iter().any(|candidate| candidate == value) {
                    Ok(Value::String(value.to_owned()))
                } else {
                    Err(GraphError::InvalidWidgetValue(self.identifier.clone()))
                }
            }
            GraphWidgetKind::Preserved { .. } => Ok(value),
        }
    }

    fn validate_schema(&self) -> Result<(), GraphError> {
        let invalid = |reason: &str| GraphError::InvalidWidgetSchema {
            widget: self.identifier.clone(),
            reason: reason.to_owned(),
        };
        match &self.kind {
            GraphWidgetKind::Integer {
                minimum,
                maximum,
                step,
            } => {
                if minimum > maximum {
                    return Err(invalid("minimum is greater than maximum"));
                }
                if *step <= 0 {
                    return Err(invalid("step must be positive"));
                }
            }
            GraphWidgetKind::Float {
                minimum,
                maximum,
                step,
            } => {
                if !minimum.is_finite() || !maximum.is_finite() || !step.is_finite() {
                    return Err(invalid("bounds and step must be finite"));
                }
                if minimum > maximum {
                    return Err(invalid("minimum is greater than maximum"));
                }
                if *step <= 0.0 {
                    return Err(invalid("step must be positive"));
                }
            }
            GraphWidgetKind::Boolean
            | GraphWidgetKind::Text { .. }
            | GraphWidgetKind::Combo { .. }
            | GraphWidgetKind::Preserved { .. } => {}
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphNodeMode {
    #[serde(alias = "active", alias = "Active")]
    Always,
    #[serde(alias = "disabled", alias = "Disabled")]
    OnEvent,
    #[serde(alias = "muted", alias = "Muted")]
    Never,
    OnTrigger,
    #[serde(alias = "bypassed", alias = "Bypassed")]
    Bypass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphVisualShape {
    Default,
    Box,
    #[serde(alias = "Round")]
    Round,
    Card,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphPaletteColor {
    Red,
    Brown,
    Green,
    Blue,
    PaleBlue,
    Cyan,
    Purple,
    Yellow,
    Black,
}

impl GraphPaletteColor {
    pub const ALL: [Self; 9] = [
        Self::Red,
        Self::Brown,
        Self::Green,
        Self::Blue,
        Self::PaleBlue,
        Self::Cyan,
        Self::Purple,
        Self::Yellow,
        Self::Black,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Red => "Red",
            Self::Brown => "Brown",
            Self::Green => "Green",
            Self::Blue => "Blue",
            Self::PaleBlue => "Pale blue",
            Self::Cyan => "Cyan",
            Self::Purple => "Purple",
            Self::Yellow => "Yellow",
            Self::Black => "Black",
        }
    }

    pub fn node_header(self) -> &'static str {
        match self {
            Self::Red => "#322",
            Self::Brown => "#332922",
            Self::Green => "#232",
            Self::Blue => "#223",
            Self::PaleBlue => "#2a363b",
            Self::Cyan => "#233",
            Self::Purple => "#323",
            Self::Yellow => "#432",
            Self::Black => "#222",
        }
    }

    pub fn node_background(self) -> &'static str {
        match self {
            Self::Red => "#533",
            Self::Brown => "#593930",
            Self::Green => "#353",
            Self::Blue => "#335",
            Self::PaleBlue => "#3f5159",
            Self::Cyan => "#355",
            Self::Purple => "#535",
            Self::Yellow => "#653",
            Self::Black => "#000",
        }
    }

    pub fn group(self) -> &'static str {
        match self {
            Self::Red => "#A88",
            Self::Brown => "#b06634",
            Self::Green => "#8A8",
            Self::Blue => "#88A",
            Self::PaleBlue => "#3f789e",
            Self::Cyan => "#8AA",
            Self::Purple => "#a1309b",
            Self::Yellow => "#b58b2a",
            Self::Black => "#444",
        }
    }
}

impl GraphVisualShape {
    pub fn source_name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Box => "box",
            Self::Round => "round",
            Self::Card => "card",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphSlotDirection {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub identifier: GraphIdentifier,
    pub type_identifier: String,
    pub title: String,
    pub position: GraphPoint,
    pub size: GraphSize,
    pub inputs: Vec<GraphPort>,
    pub outputs: Vec<GraphPort>,
    pub widgets: Vec<GraphWidget>,
    pub mode: GraphNodeMode,
    pub pinned: bool,
    pub collapsed: bool,
    pub color: Option<String>,
    pub subgraph_definition: Option<GraphIdentifier>,
    #[serde(default)]
    pub quarantine: BTreeMap<String, Value>,
    #[serde(default)]
    pub source_fields: Map<String, Value>,
}

impl GraphNode {
    pub fn new(
        identifier: GraphIdentifier,
        type_identifier: impl Into<String>,
        title: impl Into<String>,
        position: GraphPoint,
    ) -> Self {
        Self {
            identifier,
            type_identifier: type_identifier.into(),
            title: title.into(),
            position,
            size: GraphSize::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            widgets: Vec::new(),
            mode: GraphNodeMode::Always,
            pinned: false,
            collapsed: false,
            color: None,
            subgraph_definition: None,
            quarantine: BTreeMap::new(),
            source_fields: Map::new(),
        }
    }

    pub fn bounds(&self) -> GraphRect {
        GraphRect {
            origin: self.position,
            size: self.size,
        }
    }

    pub fn is_resizable(&self) -> bool {
        !self.pinned && self.operation_flag("resizable") != Some(false)
    }

    pub fn is_collapsible(&self) -> bool {
        !self.pinned
            && self.type_identifier != "Reroute"
            && self.operation_flag("collapsable") != Some(false)
            && self.operation_flag("collapsible") != Some(false)
    }

    pub fn is_clonable(&self) -> bool {
        self.operation_flag("clonable") != Some(false)
    }

    pub fn blocks_deletion(&self) -> bool {
        self.operation_flag("block_delete") == Some(true)
    }

    pub fn is_removable(&self) -> bool {
        self.operation_flag("removable") != Some(false) && !self.blocks_deletion()
    }

    fn operation_flag(&self, field: &str) -> Option<bool> {
        self.source_fields.get(field).and_then(Value::as_bool)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphLink {
    pub identifier: GraphIdentifier,
    pub origin_node: GraphIdentifier,
    pub origin_slot: usize,
    pub target_node: GraphIdentifier,
    pub target_slot: usize,
    pub type_name: String,
    pub parent_reroute: Option<GraphIdentifier>,
    #[serde(default)]
    pub source: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphGroup {
    pub identifier: GraphIdentifier,
    pub title: String,
    pub bounds: GraphRect,
    pub node_ids: BTreeSet<GraphIdentifier>,
    pub collapsed: bool,
    pub pinned: bool,
    pub color: Option<String>,
    #[serde(default)]
    pub source_fields: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphReroute {
    pub identifier: GraphIdentifier,
    pub position: GraphPoint,
    pub parent: Option<GraphIdentifier>,
    pub floating_type: Option<String>,
    #[serde(default)]
    pub source_fields: Map<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SubgraphPort {
    pub identifier: String,
    pub name: String,
    pub port_type: GraphPortType,
    pub internal_node: Option<GraphIdentifier>,
    pub internal_slot: usize,
    #[serde(default)]
    pub source_fields: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubgraphExposedWidget {
    pub identifier: String,
    pub internal_node: GraphIdentifier,
    pub internal_widget: String,
    pub widget: GraphWidget,
    #[serde(default)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubgraphDefinition {
    pub identifier: GraphIdentifier,
    pub name: String,
    pub graph: Box<GraphLevel>,
    pub inputs: Vec<SubgraphPort>,
    pub outputs: Vec<SubgraphPort>,
    pub published: bool,
    pub description: String,
    pub search_aliases: Vec<String>,
    #[serde(default)]
    pub exposed_widgets: Vec<SubgraphExposedWidget>,
    #[serde(default)]
    pub graph_inline: bool,
    #[serde(default)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GraphSelection {
    pub nodes: BTreeSet<GraphIdentifier>,
    pub links: BTreeSet<GraphIdentifier>,
    pub groups: BTreeSet<GraphIdentifier>,
    pub reroutes: BTreeSet<GraphIdentifier>,
}

impl GraphSelection {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
            && self.links.is_empty()
            && self.groups.is_empty()
            && self.reroutes.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphViewport {
    pub offset: GraphPoint,
    pub scale: f32,
    pub minimap_visible: bool,
    pub links_visible: bool,
    pub locked: bool,
}

impl Default for GraphViewport {
    fn default() -> Self {
        Self {
            offset: GraphPoint::ZERO,
            scale: 1.0,
            minimap_visible: false,
            links_visible: true,
            locked: false,
        }
    }
}

impl GraphViewport {
    pub fn screen_to_graph(&self, point: GraphPoint) -> GraphPoint {
        GraphPoint {
            x: (point.x - self.offset.x) / self.scale,
            y: (point.y - self.offset.y) / self.scale,
        }
    }

    pub fn graph_to_screen(&self, point: GraphPoint) -> GraphPoint {
        GraphPoint {
            x: point.x * self.scale + self.offset.x,
            y: point.y * self.scale + self.offset.y,
        }
    }

    fn validate(&self) -> Result<(), GraphError> {
        self.offset.validate()?;
        if self.scale.is_finite() && (MIN_VIEWPORT_SCALE..=MAX_VIEWPORT_SCALE).contains(&self.scale)
        {
            Ok(())
        } else {
            Err(GraphError::InvalidViewportScale(self.scale))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphLevel {
    pub nodes: BTreeMap<GraphIdentifier, GraphNode>,
    pub links: BTreeMap<GraphIdentifier, GraphLink>,
    pub groups: BTreeMap<GraphIdentifier, GraphGroup>,
    pub reroutes: BTreeMap<GraphIdentifier, GraphReroute>,
    pub definitions: BTreeMap<GraphIdentifier, SubgraphDefinition>,
    pub selection: GraphSelection,
    pub viewport: GraphViewport,
    #[serde(default)]
    pub source_fields: Map<String, Value>,
}

impl Default for GraphLevel {
    fn default() -> Self {
        Self {
            nodes: BTreeMap::new(),
            links: BTreeMap::new(),
            groups: BTreeMap::new(),
            reroutes: BTreeMap::new(),
            definitions: BTreeMap::new(),
            selection: GraphSelection::default(),
            viewport: GraphViewport::default(),
            source_fields: Map::new(),
        }
    }
}

impl GraphLevel {
    pub fn resolve_reroute_port_type(
        &self,
        identifier: &GraphIdentifier,
    ) -> Result<GraphPortType, GraphError> {
        resolve_reroute_port_type(self, identifier)
    }

    pub fn validate_subgraph_conversion_selection(&self) -> Result<(), GraphError> {
        conversion_node_identifiers(self).map(drop)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphDocument {
    pub schema_version: u16,
    pub document_identity: Uuid,
    pub profile_identity: Option<Uuid>,
    pub workflow_format: Option<WorkflowFormat>,
    pub root: GraphLevel,
    pub navigation: Vec<GraphIdentifier>,
    pub next_identifier: u64,
    pub diagnostics: Vec<String>,
}

impl Default for GraphDocument {
    fn default() -> Self {
        Self {
            schema_version: GRAPH_DOCUMENT_SCHEMA_VERSION,
            document_identity: Uuid::new_v4(),
            profile_identity: None,
            workflow_format: Some(WorkflowFormat::Schema1),
            root: GraphLevel::default(),
            navigation: Vec::new(),
            next_identifier: 1,
            diagnostics: Vec::new(),
        }
    }
}

impl GraphDocument {
    pub fn from_workflow(document: &WorkflowFormatDocument) -> Result<Self, GraphError> {
        if document.format() == Some(WorkflowFormat::OtherNumeric) {
            return Err(GraphError::InvalidWorkflow {
                path: "$.version".to_owned(),
                reason: "future workflow versions are preserved read-only until an explicit migration is accepted".to_owned(),
            });
        }
        if let Some(issue) = document.validation_issues().first() {
            return Err(GraphError::InvalidWorkflow {
                path: issue.path.clone(),
                reason: issue.reason.clone(),
            });
        }
        let root = parse_level(document.value(), 0)?;
        let document_identity = match document
            .value()
            .get("id")
            .and_then(Value::as_str)
            .and_then(|identifier| Uuid::parse_str(identifier).ok())
        {
            Some(identifier) => identifier,
            None => {
                let canonical_source = serde_json::to_vec(document.value())
                    .map_err(|error| GraphError::Serialization(error.to_string()))?;
                Uuid::new_v5(&Uuid::NAMESPACE_OID, &canonical_source)
            }
        };
        let mut graph = Self {
            schema_version: GRAPH_DOCUMENT_SCHEMA_VERSION,
            document_identity,
            profile_identity: None,
            workflow_format: document.format(),
            root,
            navigation: Vec::new(),
            next_identifier: 1,
            diagnostics: document
                .validation_issues()
                .iter()
                .map(|issue| format!("{}: {}", issue.path, issue.reason))
                .collect(),
        };
        graph.next_identifier = graph.maximum_numeric_identifier().saturating_add(1);
        graph.validate()?;
        Ok(graph)
    }

    pub fn from_workflow_bytes(bytes: &[u8]) -> Result<Self, GraphError> {
        match import_json(bytes)? {
            JsonImport::Workflow(document) => Self::from_workflow(&document),
            JsonImport::ApiPrompt(prompt) => Self::from_api_prompt(&prompt),
            JsonImport::Templates(_) => Err(GraphError::UnsupportedImport("workflow templates")),
        }
    }

    pub fn has_same_persisted_workflow(&self, other: &Self) -> Result<bool, GraphError> {
        let mut first = self.clone();
        let mut second = other.clone();
        normalize_workspace_projection(&mut first);
        normalize_workspace_projection(&mut second);
        Ok(first.to_workflow_value()? == second.to_workflow_value()?)
    }

    fn from_api_prompt(prompt: &crate::ApiPromptDocument) -> Result<Self, GraphError> {
        let submission = prompt.to_submission()?;
        let mut root = GraphLevel::default();
        root.source_fields
            .insert("zed_api_prompt".to_owned(), prompt.value().clone());
        let known_node_identifiers = submission
            .prompt
            .0
            .keys()
            .map(|identifier| GraphIdentifier::String(identifier.0.clone()))
            .collect::<BTreeSet<_>>();
        let mut pending_links = Vec::new();
        for (index, (node_identifier, prompt_node)) in submission.prompt.0.iter().enumerate() {
            let identifier = GraphIdentifier::String(node_identifier.0.clone());
            let title = prompt_node
                .unknown
                .get("_meta")
                .and_then(|meta| meta.get("title"))
                .and_then(Value::as_str)
                .unwrap_or(&prompt_node.class_type)
                .to_owned();
            let mut node = GraphNode::new(
                identifier.clone(),
                prompt_node.class_type.clone(),
                title,
                GraphPoint {
                    x: (index % 4) as f32 * 340.0,
                    y: (index / 4) as f32 * 220.0,
                },
            );
            if let Some(source) = prompt.value().get(&node_identifier.0) {
                node.source_fields
                    .insert("zed_api_prompt_node".to_owned(), source.clone());
            }
            for (name, value) in &prompt_node.inputs {
                let linked = value.as_array().and_then(|values| {
                    let origin = values.first().and_then(GraphIdentifier::from_value)?;
                    let origin_slot = values
                        .get(1)?
                        .as_u64()
                        .and_then(|slot| usize::try_from(slot).ok())?;
                    let origin = GraphIdentifier::String(origin.text());
                    known_node_identifiers
                        .contains(&origin)
                        .then_some((origin, origin_slot))
                });
                let input_slot = node.inputs.len();
                if let Some((origin, origin_slot)) = linked {
                    node.inputs.push(GraphPort::new(name, GraphPortType::Any));
                    pending_links.push((identifier.clone(), input_slot, origin, origin_slot));
                } else {
                    node.widgets
                        .push(GraphWidget::preserved(name, value.clone()));
                }
            }
            root.nodes.insert(identifier, node);
        }
        for (_, _, origin, origin_slot) in &pending_links {
            let origin_node = root
                .nodes
                .get_mut(origin)
                .ok_or_else(|| GraphError::UnknownNode(origin.clone()))?;
            while origin_node.outputs.len() <= *origin_slot {
                let slot = origin_node.outputs.len();
                origin_node
                    .outputs
                    .push(GraphPort::new(format!("output-{slot}"), GraphPortType::Any));
            }
        }
        let canonical_source = serde_json::to_vec(prompt.value())
            .map_err(|error| GraphError::Serialization(error.to_string()))?;
        let api_prompt_namespace = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"zed-sim-api-prompt");
        let mut graph = Self {
            schema_version: GRAPH_DOCUMENT_SCHEMA_VERSION,
            document_identity: Uuid::new_v5(&api_prompt_namespace, &canonical_source),
            profile_identity: None,
            workflow_format: Some(WorkflowFormat::Schema1),
            root,
            navigation: Vec::new(),
            next_identifier: 1,
            diagnostics: vec![
                "Imported API-format prompt into an editable native graph".to_owned(),
            ],
        };
        for (target, target_slot, origin, origin_slot) in pending_links {
            let identifier = graph.allocate_identifier();
            graph.root.links.insert(
                identifier.clone(),
                GraphLink {
                    identifier,
                    origin_node: origin,
                    origin_slot,
                    target_node: target,
                    target_slot,
                    type_name: "ANY".to_owned(),
                    parent_reroute: None,
                    source: Value::Object(Map::new()),
                },
            );
        }
        graph.validate()?;
        Ok(graph)
    }

    pub fn to_workflow_value(&self) -> Result<Value, GraphError> {
        self.validate()?;
        serialize_level(
            &self.root,
            self.workflow_format,
            Some(self.document_identity),
        )
    }

    pub fn to_workflow_bytes(&self) -> Result<Vec<u8>, GraphError> {
        serde_json::to_vec(&self.to_workflow_value()?)
            .map_err(|error| GraphError::Serialization(error.to_string()))
    }

    pub fn export_selected_subgraph_blueprint(
        &self,
        display_name: &str,
    ) -> Result<PublishedSubgraphBlueprint, GraphError> {
        export_selected_subgraph_blueprint(self, display_name)
    }

    pub fn active_graph(&self) -> Result<&GraphLevel, GraphError> {
        graph_at_path(&self.root, &self.navigation)
    }

    pub fn active_graph_mut(&mut self) -> Result<&mut GraphLevel, GraphError> {
        graph_at_path_mut(&mut self.root, &self.navigation)
    }

    pub fn active_subgraph_definition(&self) -> Result<&SubgraphDefinition, GraphError> {
        let (identifier, parent_path) = self
            .navigation
            .split_last()
            .ok_or(GraphError::AtRootGraph)?;
        graph_at_path(&self.root, parent_path)?
            .definitions
            .get(identifier)
            .ok_or_else(|| GraphError::UnknownSubgraph(identifier.clone()))
    }

    pub fn allocate_identifier(&mut self) -> GraphIdentifier {
        let identifier = self.next_available_identifier();
        advance_identifier_counter(self, &identifier);
        identifier
    }

    pub fn next_available_identifier(&self) -> GraphIdentifier {
        let mut next_identifier = self.next_identifier;
        loop {
            let identifier = GraphIdentifier::String(format!("zed-{next_identifier}"));
            if !contains_identifier(&self.root, &identifier) {
                return identifier;
            }
            next_identifier = next_identifier.saturating_add(1);
        }
    }

    pub fn allocate_subgraph_identifier(&mut self) -> GraphIdentifier {
        loop {
            let counter = self.next_identifier;
            self.next_identifier = self.next_identifier.saturating_add(1);
            let identifier = GraphIdentifier::String(
                Uuid::new_v5(
                    &self.document_identity,
                    format!("zed-subgraph-{counter}").as_bytes(),
                )
                .to_string(),
            );
            if !contains_identifier(&self.root, &identifier) {
                return identifier;
            }
        }
    }

    pub fn validate(&self) -> Result<(), GraphError> {
        if self.schema_version != GRAPH_DOCUMENT_SCHEMA_VERSION {
            return Err(GraphError::UnsupportedDocumentSchema(self.schema_version));
        }
        validate_level(&self.root, 0)?;
        graph_at_path(&self.root, &self.navigation)?;
        Ok(())
    }

    fn maximum_numeric_identifier(&self) -> u64 {
        fn visit(level: &GraphLevel, maximum: &mut u64) {
            let identifiers = level
                .nodes
                .keys()
                .chain(level.links.keys())
                .chain(level.groups.keys())
                .chain(level.reroutes.keys())
                .chain(level.definitions.keys());
            for identifier in identifiers {
                match identifier {
                    GraphIdentifier::Integer(value) if *value >= 0 => {
                        *maximum = (*maximum).max(*value as u64);
                    }
                    GraphIdentifier::String(value) => {
                        if let Some(value) = value
                            .strip_prefix("zed-")
                            .and_then(|value| value.parse::<u64>().ok())
                        {
                            *maximum = (*maximum).max(value);
                        }
                    }
                    GraphIdentifier::Integer(_) => {}
                }
            }
            for definition in level.definitions.values() {
                visit(&definition.graph, maximum);
            }
        }
        let mut maximum = 0;
        visit(&self.root, &mut maximum);
        maximum
    }
}

fn normalize_workspace_projection(document: &mut GraphDocument) {
    fn normalize_level(level: &mut GraphLevel) {
        level.selection = GraphSelection::default();
        level.viewport = GraphViewport::default();
        for definition in level.definitions.values_mut() {
            normalize_level(&mut definition.graph);
        }
    }

    document.profile_identity = None;
    document.navigation.clear();
    document.next_identifier = 0;
    document.diagnostics.clear();
    normalize_level(&mut document.root);
}

fn contains_identifier(level: &GraphLevel, identifier: &GraphIdentifier) -> bool {
    level.nodes.contains_key(identifier)
        || level.links.contains_key(identifier)
        || level.groups.contains_key(identifier)
        || level.reroutes.contains_key(identifier)
        || level.definitions.contains_key(identifier)
        || level
            .definitions
            .values()
            .any(|definition| contains_identifier(&definition.graph, identifier))
}

fn graph_at_path<'a>(
    mut level: &'a GraphLevel,
    path: &[GraphIdentifier],
) -> Result<&'a GraphLevel, GraphError> {
    for identifier in path {
        level = level
            .definitions
            .get(identifier)
            .map(|definition| definition.graph.as_ref())
            .ok_or_else(|| GraphError::UnknownSubgraph(identifier.clone()))?;
    }
    Ok(level)
}

fn graph_at_path_mut<'a>(
    level: &'a mut GraphLevel,
    path: &[GraphIdentifier],
) -> Result<&'a mut GraphLevel, GraphError> {
    let Some((head, tail)) = path.split_first() else {
        return Ok(level);
    };
    let definition = level
        .definitions
        .get_mut(head)
        .ok_or_else(|| GraphError::UnknownSubgraph(head.clone()))?;
    graph_at_path_mut(&mut definition.graph, tail)
}

fn advance_identifier_counter(document: &mut GraphDocument, identifier: &GraphIdentifier) {
    let numeric_identifier = match identifier {
        GraphIdentifier::Integer(value) if *value >= 0 => Some(*value as u64),
        GraphIdentifier::String(value) => value
            .strip_prefix("zed-")
            .and_then(|value| value.parse::<u64>().ok()),
        GraphIdentifier::Integer(_) => None,
    };
    if let Some(identifier) = numeric_identifier {
        document.next_identifier = document.next_identifier.max(identifier.saturating_add(1));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NodeCreationSource {
    Library,
    Search,
    LinkRelease,
    Paste,
    Template,
    DragAndDrop,
    Replacement,
    Extension,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NodeToggle {
    Mute,
    Bypass,
    Pin,
    Collapse,
    Disable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GroupToggle {
    Pin,
    Collapse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SelectionMode {
    Replace,
    Add,
    Toggle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LayoutOperation {
    AlignLeft,
    AlignRight,
    AlignTop,
    AlignBottom,
    AlignHorizontalCenters,
    AlignVerticalCenters,
    DistributeHorizontally,
    DistributeVertically,
    ArrangeGrid,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum GraphCommand {
    Batch {
        commands: Vec<GraphCommand>,
    },
    AddNode {
        node: GraphNode,
        source: NodeCreationSource,
    },
    RemoveSelection,
    RemoveItems {
        selection: GraphSelection,
    },
    DuplicateSelection {
        selection: GraphSelection,
        offset: GraphPoint,
    },
    Connect {
        link: GraphLink,
        replace_existing: bool,
    },
    RemoveLink {
        identifier: GraphIdentifier,
    },
    MoveSelection {
        delta: GraphPoint,
        snap: Option<f32>,
    },
    ResizeNode {
        identifier: GraphIdentifier,
        size: GraphSize,
    },
    SetWidget {
        node: GraphIdentifier,
        widget: String,
        value: Value,
    },
    ConvertWidgetToInput {
        node: GraphIdentifier,
        widget: String,
        converted: bool,
    },
    ToggleNodes {
        identifiers: BTreeSet<GraphIdentifier>,
        toggle: NodeToggle,
    },
    ToggleGroups {
        identifiers: BTreeSet<GraphIdentifier>,
        toggle: GroupToggle,
    },
    RenameNode {
        identifier: GraphIdentifier,
        title: String,
    },
    SetNodeColor {
        identifier: GraphIdentifier,
        color: Option<String>,
    },
    SetNodePalette {
        identifiers: BTreeSet<GraphIdentifier>,
        color: Option<GraphPaletteColor>,
    },
    SetNodeShape {
        identifiers: BTreeSet<GraphIdentifier>,
        shape: GraphVisualShape,
    },
    SetNodeMode {
        identifiers: BTreeSet<GraphIdentifier>,
        mode: GraphNodeMode,
    },
    SetNodeAdvancedVisibility {
        identifiers: BTreeSet<GraphIdentifier>,
        visible: bool,
    },
    SetNodeProperties {
        identifier: GraphIdentifier,
        properties: Map<String, Value>,
    },
    RenameGroup {
        identifier: GraphIdentifier,
        title: String,
    },
    SetGroupColor {
        identifier: GraphIdentifier,
        color: Option<String>,
    },
    SetGroupFontSize {
        identifier: GraphIdentifier,
        font_size: f32,
    },
    AddNodesToGroup {
        identifier: GraphIdentifier,
        nodes: BTreeSet<GraphIdentifier>,
        padding: f32,
    },
    DisconnectSubgraphSlot {
        direction: GraphSlotDirection,
        slot: usize,
    },
    RenameSubgraphSlot {
        direction: GraphSlotDirection,
        slot: usize,
        name: String,
    },
    RemoveSubgraphSlot {
        direction: GraphSlotDirection,
        slot: usize,
    },
    SetRerouteTypeVisibility {
        identifiers: BTreeSet<GraphIdentifier>,
        visible: bool,
    },
    SetSelection {
        selection: GraphSelection,
        mode: SelectionMode,
    },
    SelectInRect {
        bounds: GraphRect,
        mode: SelectionMode,
    },
    SelectAll,
    ClearSelection,
    CreateGroup {
        group: GraphGroup,
    },
    AddSubgraphDefinition {
        definition: SubgraphDefinition,
    },
    Ungroup {
        identifier: GraphIdentifier,
    },
    FitGroup {
        identifier: GraphIdentifier,
        padding: f32,
    },
    AddReroute {
        reroute: GraphReroute,
    },
    RemoveReroute {
        identifier: GraphIdentifier,
    },
    MoveReroute {
        identifier: GraphIdentifier,
        position: GraphPoint,
    },
    ReparentReroute {
        identifier: GraphIdentifier,
        parent: Option<GraphIdentifier>,
    },
    LayoutSelection {
        operation: LayoutOperation,
        spacing: f32,
    },
    SetViewport {
        viewport: GraphViewport,
    },
    PanViewport {
        delta: GraphPoint,
    },
    ZoomViewport {
        factor: f32,
        anchor: GraphPoint,
    },
    FitViewport {
        bounds: GraphRect,
        available: GraphSize,
        padding: f32,
    },
    ConvertSelectionToSubgraph {
        definition_identifier: GraphIdentifier,
        instance_identifier: GraphIdentifier,
        name: String,
    },
    OpenSubgraph {
        definition_identifier: GraphIdentifier,
    },
    ExitSubgraph,
    UnpackSubgraph {
        instance_identifier: GraphIdentifier,
    },
    RemoveSubgraphDefinition {
        definition_identifier: GraphIdentifier,
        remove_instances: bool,
    },
    SetSubgraphWidgetExposure {
        definition_identifier: GraphIdentifier,
        internal_node: GraphIdentifier,
        widget: String,
        exposed: bool,
    },
    SetSubgraphMetadata {
        definition_identifier: GraphIdentifier,
        description: String,
        search_aliases: Vec<String>,
    },
    ReconcileNode {
        identifier: GraphIdentifier,
        inputs: Vec<GraphPort>,
        outputs: Vec<GraphPort>,
        widgets: Vec<GraphWidget>,
        confirm_discard: bool,
    },
}

impl GraphCommand {
    fn allowed_while_canvas_locked(&self) -> bool {
        match self {
            Self::Batch { commands } => commands
                .iter()
                .all(GraphCommand::allowed_while_canvas_locked),
            Self::SetViewport { .. }
            | Self::PanViewport { .. }
            | Self::ZoomViewport { .. }
            | Self::FitViewport { .. }
            | Self::OpenSubgraph { .. }
            | Self::ExitSubgraph => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphHistoryEntry {
    pub command: GraphCommand,
    pub before: GraphDocument,
    pub after: GraphDocument,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphCommandEngine {
    pub schema_version: u16,
    pub document: GraphDocument,
    undo: VecDeque<GraphHistoryEntry>,
    redo: VecDeque<GraphHistoryEntry>,
    history_limit: usize,
}

impl GraphCommandEngine {
    pub fn new(mut document: GraphDocument) -> Result<Self, GraphError> {
        recompute_group_memberships(&mut document.root);
        document.validate()?;
        Ok(Self {
            schema_version: GRAPH_DOCUMENT_SCHEMA_VERSION,
            document,
            undo: VecDeque::new(),
            redo: VecDeque::new(),
            history_limit: MAX_GRAPH_HISTORY_ENTRIES,
        })
    }

    pub fn validate_command(&self, command: &GraphCommand) -> Result<(), GraphError> {
        self.validated_candidate(command).map(drop)
    }

    pub fn apply(&mut self, command: GraphCommand) -> Result<(), GraphError> {
        let candidate = self.validated_candidate(&command)?;
        if candidate == self.document {
            return Ok(());
        }
        let before = self.document.clone();
        let entry = GraphHistoryEntry {
            command,
            before,
            after: candidate.clone(),
        };
        self.document = candidate;
        self.undo.push_back(entry);
        while self.undo.len() > self.history_limit {
            self.undo.pop_front();
        }
        self.redo.clear();
        Ok(())
    }

    fn validated_candidate(&self, command: &GraphCommand) -> Result<GraphDocument, GraphError> {
        if self.document.active_graph()?.viewport.locked && !command.allowed_while_canvas_locked() {
            return Err(GraphError::CanvasLocked);
        }
        let mut candidate = self.document.clone();
        apply_command(&mut candidate, command)?;
        candidate.validate()?;
        Ok(candidate)
    }

    pub fn undo(&mut self) -> bool {
        let Some(entry) = self.undo.pop_back() else {
            return false;
        };
        self.document = entry.before.clone();
        self.redo.push_back(entry);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(entry) = self.redo.pop_back() else {
            return false;
        };
        self.document = entry.after.clone();
        self.undo.push_back(entry);
        true
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn replace_workspace_state(
        &mut self,
        selection: GraphSelection,
        viewport: GraphViewport,
    ) -> Result<(), GraphError> {
        viewport.validate()?;
        let graph = self.document.active_graph_mut()?;
        validate_selection(graph, &selection)?;
        graph.selection = selection;
        graph.viewport = viewport;
        Ok(())
    }

    pub fn bind_profile_identity(&mut self, profile_identity: Uuid) {
        if self.document.profile_identity.is_none() {
            self.document.profile_identity = Some(profile_identity);
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>, GraphError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| GraphError::Serialization(error.to_string()))?;
        if bytes.len() > MAX_GRAPH_SNAPSHOT_BYTES {
            return Err(GraphError::SnapshotTooLarge(bytes.len()));
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, GraphError> {
        if bytes.len() > MAX_GRAPH_SNAPSHOT_BYTES {
            return Err(GraphError::SnapshotTooLarge(bytes.len()));
        }
        validate_json_depth(bytes, MAX_GRAPH_DEPTH)?;
        let mut engine: Self = serde_json::from_slice(bytes)
            .map_err(|error| GraphError::Serialization(error.to_string()))?;
        if engine.schema_version != GRAPH_DOCUMENT_SCHEMA_VERSION {
            return Err(GraphError::UnsupportedDocumentSchema(engine.schema_version));
        }
        if engine.undo.len() > MAX_GRAPH_HISTORY_ENTRIES
            || engine.redo.len() > MAX_GRAPH_HISTORY_ENTRIES
            || engine.history_limit > MAX_GRAPH_HISTORY_ENTRIES
        {
            return Err(GraphError::InvalidHistory);
        }
        normalize_json_identifier_map_keys(&mut engine.document.root)?;
        recompute_group_memberships(&mut engine.document.root);
        for entry in engine.undo.iter_mut().chain(engine.redo.iter_mut()) {
            normalize_json_identifier_map_keys(&mut entry.before.root)?;
            normalize_json_identifier_map_keys(&mut entry.after.root)?;
            recompute_group_memberships(&mut entry.before.root);
            recompute_group_memberships(&mut entry.after.root);
        }
        engine.document.validate()?;
        for entry in engine.undo.iter().chain(engine.redo.iter()) {
            entry.before.validate()?;
            entry.after.validate()?;
            let mut replayed = entry.before.clone();
            apply_command(&mut replayed, &entry.command)?;
            replayed.validate()?;
            if replayed != entry.after {
                return Err(GraphError::InvalidHistory);
            }
        }
        if engine
            .undo
            .iter()
            .zip(engine.undo.iter().skip(1))
            .any(|(previous, next)| previous.after != next.before)
        {
            return Err(GraphError::InvalidHistory);
        }
        if engine
            .redo
            .iter()
            .zip(engine.redo.iter().skip(1))
            .any(|(later, earlier)| earlier.after != later.before)
        {
            return Err(GraphError::InvalidHistory);
        }
        if let Some(next_redo) = engine.redo.back() {
            if !same_engine_document_state(&engine.document, &next_redo.before) {
                return Err(GraphError::InvalidHistory);
            }
        } else if let Some(last_undo) = engine.undo.back()
            && !same_engine_document_state(&engine.document, &last_undo.after)
        {
            return Err(GraphError::InvalidHistory);
        }
        Ok(engine)
    }
}

fn same_engine_document_state(first: &GraphDocument, second: &GraphDocument) -> bool {
    fn normalize_level(level: &mut GraphLevel) {
        level.selection = GraphSelection::default();
        level.viewport = GraphViewport::default();
        for definition in level.definitions.values_mut() {
            normalize_level(&mut definition.graph);
        }
    }

    let mut first = first.clone();
    let mut second = second.clone();
    first.profile_identity = None;
    second.profile_identity = None;
    normalize_level(&mut first.root);
    normalize_level(&mut second.root);
    first == second
}

fn apply_command(document: &mut GraphDocument, command: &GraphCommand) -> Result<(), GraphError> {
    match command {
        GraphCommand::Batch { commands } => {
            for command in commands {
                apply_command(document, command)?;
            }
        }
        GraphCommand::AddNode { node, .. } => {
            node.position.validate()?;
            node.size.validate()?;
            advance_identifier_counter(document, &node.identifier);
            let graph = document.active_graph_mut()?;
            if graph.nodes.contains_key(&node.identifier) {
                return Err(GraphError::DuplicateEntity(node.identifier.clone()));
            }
            graph.nodes.insert(node.identifier.clone(), node.clone());
            graph.selection = GraphSelection {
                nodes: BTreeSet::from([node.identifier.clone()]),
                ..GraphSelection::default()
            };
        }
        GraphCommand::RemoveSelection => remove_selection(document.active_graph_mut()?)?,
        GraphCommand::RemoveItems { selection } => {
            remove_items(document.active_graph_mut()?, selection)?
        }
        GraphCommand::DuplicateSelection { selection, offset } => {
            offset.validate()?;
            validate_selection(document.active_graph()?, selection)?;
            if selection.is_empty() {
                return Err(GraphError::EmptySelection);
            }
            let mut source = document.clone();
            source.active_graph_mut()?.selection = selection.clone();
            let clipboard = GraphClipboard::copy(&source)?;
            let paste = clipboard.paste_command(document, *offset)?;
            apply_command(document, &paste)?;
        }
        GraphCommand::Connect {
            link,
            replace_existing,
        } => {
            advance_identifier_counter(document, &link.identifier);
            connect(
                document.active_graph_mut()?,
                link.clone(),
                *replace_existing,
            )?;
        }
        GraphCommand::RemoveLink { identifier } => {
            let graph = document.active_graph_mut()?;
            graph
                .links
                .remove(identifier)
                .ok_or_else(|| GraphError::UnknownLink(identifier.clone()))?;
            graph.selection.links.remove(identifier);
        }
        GraphCommand::MoveSelection { delta, snap } => {
            delta.validate()?;
            move_selection(document.active_graph_mut()?, *delta, *snap)?;
        }
        GraphCommand::ResizeNode { identifier, size } => {
            size.validate()?;
            let node = document
                .active_graph_mut()?
                .nodes
                .get_mut(identifier)
                .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
            if !node.is_resizable() {
                return Err(GraphError::NodeOperationRestricted {
                    node: identifier.clone(),
                    operation: "resize",
                });
            }
            node.size = *size;
        }
        GraphCommand::SetWidget {
            node,
            widget,
            value,
        } => {
            let node = document
                .active_graph_mut()?
                .nodes
                .get_mut(node)
                .ok_or_else(|| GraphError::UnknownNode(node.clone()))?;
            let widget = node
                .widgets
                .iter_mut()
                .find(|candidate| candidate.identifier == *widget)
                .ok_or_else(|| GraphError::UnknownWidget(widget.clone()))?;
            let normalized = widget.normalize(value.clone())?;
            widget.value = normalized.clone();
            widget.prompt_value = normalized;
            widget.validation = WidgetValidation::Valid;
        }
        GraphCommand::ConvertWidgetToInput {
            node,
            widget,
            converted,
        } => {
            let graph = document.active_graph_mut()?;
            let node_snapshot = graph
                .nodes
                .get(node)
                .cloned()
                .ok_or_else(|| GraphError::UnknownNode(node.clone()))?;
            let widget_snapshot = node_snapshot
                .widgets
                .iter()
                .find(|candidate| candidate.identifier == *widget)
                .cloned()
                .ok_or_else(|| GraphError::UnknownWidget(widget.clone()))?;
            if *converted {
                if !node_snapshot.inputs.iter().any(|port| port.name == *widget) {
                    let mut port = GraphPort::new(
                        widget.clone(),
                        match &widget_snapshot.kind {
                            GraphWidgetKind::Boolean => GraphPortType::from_name("BOOLEAN"),
                            GraphWidgetKind::Integer { .. } => GraphPortType::from_name("INT"),
                            GraphWidgetKind::Float { .. } => GraphPortType::from_name("FLOAT"),
                            GraphWidgetKind::Text { .. } => GraphPortType::from_name("STRING"),
                            GraphWidgetKind::Combo { dynamic, .. } => {
                                let mut port_type = GraphPortType::from_name("COMBO");
                                if *dynamic {
                                    port_type = GraphPortType::Virtual(BTreeSet::from([
                                        "COMBO".to_owned(),
                                        "STRING".to_owned(),
                                    ]));
                                }
                                port_type
                            }
                            GraphWidgetKind::Preserved { .. } => GraphPortType::Any,
                        },
                    );
                    port.dynamic = matches!(
                        widget_snapshot.kind,
                        GraphWidgetKind::Combo { dynamic: true, .. }
                    );
                    graph
                        .nodes
                        .get_mut(node)
                        .ok_or_else(|| GraphError::UnknownNode(node.clone()))?
                        .inputs
                        .push(port);
                }
            } else if let Some(slot) = node_snapshot
                .inputs
                .iter()
                .position(|port| port.name == *widget)
            {
                let connected = graph
                    .links
                    .values()
                    .filter(|link| link.target_node == *node && link.target_slot == slot)
                    .map(|link| link.identifier.clone())
                    .collect::<Vec<_>>();
                if !connected.is_empty() {
                    return Err(GraphError::ReconciliationRequiresConfirmation {
                        links: connected,
                        widgets: Vec::new(),
                    });
                }
                let node = graph
                    .nodes
                    .get_mut(node)
                    .ok_or_else(|| GraphError::UnknownNode(node.clone()))?;
                node.inputs.remove(slot);
                let node_identifier = node.identifier.clone();
                for link in graph.links.values_mut() {
                    if link.target_node == node_identifier && link.target_slot > slot {
                        link.target_slot -= 1;
                    }
                }
            }
            graph
                .nodes
                .get_mut(node)
                .and_then(|node| {
                    node.widgets
                        .iter_mut()
                        .find(|candidate| candidate.identifier == *widget)
                })
                .ok_or_else(|| GraphError::UnknownWidget(widget.clone()))?
                .converted_to_input = *converted;
        }
        GraphCommand::ToggleNodes {
            identifiers,
            toggle,
        } => toggle_nodes(document.active_graph_mut()?, identifiers, *toggle)?,
        GraphCommand::ToggleGroups {
            identifiers,
            toggle,
        } => toggle_groups(document.active_graph_mut()?, identifiers, *toggle)?,
        GraphCommand::RenameNode { identifier, title } => {
            validate_graph_label(title)?;
            document
                .active_graph_mut()?
                .nodes
                .get_mut(identifier)
                .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?
                .title = title.trim().to_owned();
        }
        GraphCommand::SetNodeColor { identifier, color } => {
            document
                .active_graph_mut()?
                .nodes
                .get_mut(identifier)
                .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?
                .color = color.clone();
        }
        GraphCommand::SetNodePalette { identifiers, color } => {
            let graph = document.active_graph_mut()?;
            validate_node_identifiers(graph, identifiers)?;
            for identifier in identifiers {
                let node = graph
                    .nodes
                    .get_mut(identifier)
                    .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
                node.color = color.map(|color| color.node_header().to_owned());
                if let Some(color) = color {
                    node.source_fields.insert(
                        "bgcolor".to_owned(),
                        Value::String(color.node_background().to_owned()),
                    );
                } else {
                    node.source_fields.remove("bgcolor");
                }
            }
        }
        GraphCommand::SetNodeShape { identifiers, shape } => {
            let graph = document.active_graph_mut()?;
            validate_node_identifiers(graph, identifiers)?;
            for identifier in identifiers {
                let node = graph
                    .nodes
                    .get_mut(identifier)
                    .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
                node.source_fields.insert(
                    "shape".to_owned(),
                    Value::String(shape.source_name().to_owned()),
                );
            }
        }
        GraphCommand::SetNodeMode { identifiers, mode } => {
            let graph = document.active_graph_mut()?;
            validate_node_identifiers(graph, identifiers)?;
            for identifier in identifiers {
                graph
                    .nodes
                    .get_mut(identifier)
                    .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?
                    .mode = *mode;
            }
        }
        GraphCommand::SetNodeAdvancedVisibility {
            identifiers,
            visible,
        } => {
            let graph = document.active_graph_mut()?;
            validate_node_identifiers(graph, identifiers)?;
            if let Some(identifier) = identifiers.iter().find(|identifier| {
                graph.nodes.get(*identifier).is_some_and(|node| {
                    !node.widgets.iter().any(|widget| {
                        widget
                            .unknown
                            .get("advanced")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                    })
                })
            }) {
                return Err(GraphError::NodeHasNoAdvancedWidgets(identifier.clone()));
            }
            for identifier in identifiers {
                graph
                    .nodes
                    .get_mut(identifier)
                    .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?
                    .source_fields
                    .insert("show_advanced".to_owned(), Value::Bool(*visible));
            }
        }
        GraphCommand::SetNodeProperties {
            identifier,
            properties,
        } => {
            let encoded = serde_json::to_vec(properties)
                .map_err(|error| GraphError::Serialization(error.to_string()))?;
            if encoded.len() > 64 * 1024 {
                return Err(GraphError::InvalidNodeProperties(encoded.len()));
            }
            document
                .active_graph_mut()?
                .nodes
                .get_mut(identifier)
                .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?
                .source_fields
                .insert("properties".to_owned(), Value::Object(properties.clone()));
        }
        GraphCommand::RenameGroup { identifier, title } => {
            validate_graph_label(title)?;
            document
                .active_graph_mut()?
                .groups
                .get_mut(identifier)
                .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?
                .title = title.trim().to_owned();
        }
        GraphCommand::SetGroupColor { identifier, color } => {
            document
                .active_graph_mut()?
                .groups
                .get_mut(identifier)
                .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?
                .color = color.clone();
        }
        GraphCommand::SetGroupFontSize {
            identifier,
            font_size,
        } => {
            if !font_size.is_finite() || !(6.0..=96.0).contains(font_size) {
                return Err(GraphError::InvalidGroupFontSize(*font_size));
            }
            document
                .active_graph_mut()?
                .groups
                .get_mut(identifier)
                .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?
                .source_fields
                .insert("font_size".to_owned(), Value::from(*font_size));
        }
        GraphCommand::AddNodesToGroup {
            identifier,
            nodes,
            padding,
        } => {
            validate_group_padding(*padding)?;
            let graph = document.active_graph_mut()?;
            validate_node_identifiers(graph, nodes)?;
            let group = graph
                .groups
                .get(identifier)
                .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?;
            if group.pinned {
                return Err(GraphError::PinnedEntity(identifier.clone()));
            }
            let node_ids = group
                .node_ids
                .union(nodes)
                .cloned()
                .collect::<BTreeSet<_>>();
            let bounds = padded_node_bounds(graph, &node_ids, *padding)?;
            graph
                .groups
                .get_mut(identifier)
                .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?
                .bounds = bounds;
        }
        GraphCommand::DisconnectSubgraphSlot { direction, slot } => {
            disconnect_subgraph_slot(document, *direction, *slot)?
        }
        GraphCommand::RenameSubgraphSlot {
            direction,
            slot,
            name,
        } => rename_subgraph_slot(document, *direction, *slot, name)?,
        GraphCommand::RemoveSubgraphSlot { direction, slot } => {
            remove_subgraph_slot(document, *direction, *slot)?
        }
        GraphCommand::SetRerouteTypeVisibility {
            identifiers,
            visible,
        } => {
            let graph = document.active_graph_mut()?;
            if let Some(missing) = identifiers
                .iter()
                .find(|identifier| !graph.reroutes.contains_key(*identifier))
            {
                return Err(GraphError::UnknownReroute(missing.clone()));
            }
            for identifier in identifiers {
                graph
                    .reroutes
                    .get_mut(identifier)
                    .ok_or_else(|| GraphError::UnknownReroute(identifier.clone()))?
                    .source_fields
                    .insert("show_type".to_owned(), Value::Bool(*visible));
            }
        }
        GraphCommand::SetSelection { selection, mode } => {
            set_selection(document.active_graph_mut()?, selection, *mode)?;
        }
        GraphCommand::SelectInRect { bounds, mode } => {
            bounds.validate()?;
            let graph = document.active_graph_mut()?;
            let mut selection = GraphSelection::default();
            selection.nodes.extend(
                graph
                    .nodes
                    .values()
                    .filter(|node| bounds.intersects(node.bounds()))
                    .map(|node| node.identifier.clone()),
            );
            selection.groups.extend(
                graph
                    .groups
                    .values()
                    .filter(|group| bounds.intersects(group.bounds))
                    .map(|group| group.identifier.clone()),
            );
            selection.reroutes.extend(
                graph
                    .reroutes
                    .values()
                    .filter(|reroute| bounds.contains(reroute.position))
                    .map(|reroute| reroute.identifier.clone()),
            );
            selection.links.extend(
                graph
                    .links
                    .values()
                    .filter(|link| link_intersects_rect(graph, link, *bounds))
                    .map(|link| link.identifier.clone()),
            );
            set_selection(graph, &selection, *mode)?;
        }
        GraphCommand::SelectAll => {
            let graph = document.active_graph_mut()?;
            graph.selection = GraphSelection {
                nodes: graph.nodes.keys().cloned().collect(),
                links: graph.links.keys().cloned().collect(),
                groups: graph.groups.keys().cloned().collect(),
                reroutes: graph.reroutes.keys().cloned().collect(),
            };
        }
        GraphCommand::ClearSelection => {
            document.active_graph_mut()?.selection = GraphSelection::default();
        }
        GraphCommand::CreateGroup { group } => {
            group.bounds.validate()?;
            validate_graph_label(&group.title)?;
            advance_identifier_counter(document, &group.identifier);
            let graph = document.active_graph_mut()?;
            if graph.groups.contains_key(&group.identifier) {
                return Err(GraphError::DuplicateEntity(group.identifier.clone()));
            }
            let mut group = group.clone();
            group.title = group.title.trim().to_owned();
            group.node_ids.clear();
            group.source_fields.remove("nodes");
            graph.groups.insert(group.identifier.clone(), group);
        }
        GraphCommand::AddSubgraphDefinition { definition } => {
            advance_identifier_counter(document, &definition.identifier);
            validate_level(&definition.graph, 1)?;
            let graph = document.active_graph_mut()?;
            if graph.definitions.contains_key(&definition.identifier) {
                return Err(GraphError::DuplicateEntity(definition.identifier.clone()));
            }
            graph
                .definitions
                .insert(definition.identifier.clone(), definition.clone());
        }
        GraphCommand::Ungroup { identifier } => {
            document
                .active_graph_mut()?
                .groups
                .remove(identifier)
                .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?;
        }
        GraphCommand::FitGroup {
            identifier,
            padding,
        } => fit_group(document.active_graph_mut()?, identifier, *padding)?,
        GraphCommand::AddReroute { reroute } => {
            reroute.position.validate()?;
            validate_floating_reroute(reroute)?;
            advance_identifier_counter(document, &reroute.identifier);
            let graph = document.active_graph_mut()?;
            if graph.reroutes.contains_key(&reroute.identifier) {
                return Err(GraphError::DuplicateEntity(reroute.identifier.clone()));
            }
            if let Some(parent) = &reroute.parent
                && !graph.reroutes.contains_key(parent)
            {
                return Err(GraphError::UnknownReroute(parent.clone()));
            }
            graph
                .reroutes
                .insert(reroute.identifier.clone(), reroute.clone());
        }
        GraphCommand::RemoveReroute { identifier } => {
            remove_reroutes(
                document.active_graph_mut()?,
                &BTreeSet::from([identifier.clone()]),
            )?;
        }
        GraphCommand::MoveReroute {
            identifier,
            position,
        } => {
            position.validate()?;
            document
                .active_graph_mut()?
                .reroutes
                .get_mut(identifier)
                .ok_or_else(|| GraphError::UnknownReroute(identifier.clone()))?
                .position = *position;
        }
        GraphCommand::ReparentReroute { identifier, parent } => {
            reparent_reroute(document.active_graph_mut()?, identifier, parent.clone())?;
        }
        GraphCommand::LayoutSelection { operation, spacing } => {
            layout_selection(document.active_graph_mut()?, *operation, *spacing)?;
        }
        GraphCommand::SetViewport { viewport } => {
            viewport.validate()?;
            document.active_graph_mut()?.viewport = viewport.clone();
        }
        GraphCommand::PanViewport { delta } => {
            delta.validate()?;
            let viewport = &mut document.active_graph_mut()?.viewport;
            viewport.offset = viewport.offset.translated(*delta);
        }
        GraphCommand::ZoomViewport { factor, anchor } => {
            zoom_viewport(document.active_graph_mut()?, *factor, *anchor)?;
        }
        GraphCommand::FitViewport {
            bounds,
            available,
            padding,
        } => fit_viewport(document.active_graph_mut()?, *bounds, *available, *padding)?,
        GraphCommand::ConvertSelectionToSubgraph {
            definition_identifier,
            instance_identifier,
            name,
        } => {
            validate_graph_label(name)?;
            if definition_identifier == instance_identifier {
                return Err(GraphError::InvalidSubgraphConversion);
            }
            if contains_identifier(&document.root, definition_identifier) {
                return Err(GraphError::DuplicateEntity(definition_identifier.clone()));
            }
            if contains_identifier(&document.root, instance_identifier) {
                return Err(GraphError::DuplicateEntity(instance_identifier.clone()));
            }
            advance_identifier_counter(document, definition_identifier);
            advance_identifier_counter(document, instance_identifier);
            convert_selection_to_subgraph(
                document.active_graph_mut()?,
                definition_identifier.clone(),
                instance_identifier.clone(),
                name.trim().to_owned(),
            )?;
        }
        GraphCommand::OpenSubgraph {
            definition_identifier,
        } => {
            if !document
                .active_graph()?
                .definitions
                .contains_key(definition_identifier)
            {
                return Err(GraphError::UnknownSubgraph(definition_identifier.clone()));
            }
            document.navigation.push(definition_identifier.clone());
        }
        GraphCommand::ExitSubgraph => {
            if document.navigation.pop().is_none() {
                return Err(GraphError::AtRootGraph);
            }
        }
        GraphCommand::UnpackSubgraph {
            instance_identifier,
        } => unpack_subgraph(document, instance_identifier)?,
        GraphCommand::RemoveSubgraphDefinition {
            definition_identifier,
            remove_instances,
        } => remove_subgraph_definition(
            document.active_graph_mut()?,
            definition_identifier,
            *remove_instances,
        )?,
        GraphCommand::SetSubgraphWidgetExposure {
            definition_identifier,
            internal_node,
            widget,
            exposed,
        } => set_subgraph_widget_exposure(
            document.active_graph_mut()?,
            definition_identifier,
            internal_node,
            widget,
            *exposed,
        )?,
        GraphCommand::SetSubgraphMetadata {
            definition_identifier,
            description,
            search_aliases,
        } => {
            let definition = document
                .active_graph_mut()?
                .definitions
                .get_mut(definition_identifier)
                .ok_or_else(|| GraphError::UnknownSubgraph(definition_identifier.clone()))?;
            definition.description = description.clone();
            definition.search_aliases = search_aliases.clone();
        }
        GraphCommand::ReconcileNode {
            identifier,
            inputs,
            outputs,
            widgets,
            confirm_discard,
        } => reconcile_node(
            document.active_graph_mut()?,
            identifier,
            inputs,
            outputs,
            widgets,
            *confirm_discard,
        )?,
    }
    recompute_group_memberships(&mut document.root);
    Ok(())
}

fn remove_selection(graph: &mut GraphLevel) -> Result<(), GraphError> {
    let selection = graph.selection.clone();
    remove_items(graph, &selection)
}

fn remove_items(graph: &mut GraphLevel, selection: &GraphSelection) -> Result<(), GraphError> {
    validate_selection(graph, selection)?;
    if selection.is_empty() {
        return Err(GraphError::EmptySelection);
    }
    validate_nodes_removable(graph, &selection.nodes, "remove")?;
    if let Some(identifier) = selection
        .nodes
        .iter()
        .find(|identifier| graph.nodes.get(*identifier).is_some_and(|node| node.pinned))
    {
        return Err(GraphError::PinnedEntity(identifier.clone()));
    }
    if let Some(identifier) = selection.groups.iter().find(|identifier| {
        graph
            .groups
            .get(*identifier)
            .is_some_and(|group| group.pinned)
    }) {
        return Err(GraphError::PinnedEntity(identifier.clone()));
    }
    let nodes = selection.nodes.clone();
    graph
        .nodes
        .retain(|identifier, _| !nodes.contains(identifier));
    graph.links.retain(|identifier, link| {
        !selection.links.contains(identifier)
            && !nodes.contains(&link.origin_node)
            && !nodes.contains(&link.target_node)
    });
    graph
        .groups
        .retain(|identifier, _| !selection.groups.contains(identifier));
    for group in graph.groups.values_mut() {
        group
            .node_ids
            .retain(|identifier| !nodes.contains(identifier));
    }
    let reroutes = selection.reroutes.clone();
    remove_reroutes(graph, &reroutes)?;
    graph
        .selection
        .nodes
        .retain(|identifier| !nodes.contains(identifier));
    graph
        .selection
        .links
        .retain(|identifier| !selection.links.contains(identifier));
    graph
        .selection
        .groups
        .retain(|identifier| !selection.groups.contains(identifier));
    Ok(())
}

fn validate_node_identifiers(
    graph: &GraphLevel,
    identifiers: &BTreeSet<GraphIdentifier>,
) -> Result<(), GraphError> {
    if identifiers.is_empty() {
        return Err(GraphError::EmptySelection);
    }
    if let Some(missing) = identifiers
        .iter()
        .find(|identifier| !graph.nodes.contains_key(*identifier))
    {
        return Err(GraphError::UnknownNode(missing.clone()));
    }
    Ok(())
}

fn validate_nodes_clonable(
    graph: &GraphLevel,
    identifiers: &BTreeSet<GraphIdentifier>,
) -> Result<(), GraphError> {
    if let Some(identifier) = identifiers.iter().find(|identifier| {
        graph
            .nodes
            .get(*identifier)
            .is_some_and(|node| !node.is_clonable())
    }) {
        return Err(GraphError::NodeOperationRestricted {
            node: identifier.clone(),
            operation: "clone",
        });
    }
    Ok(())
}

fn validate_nodes_removable(
    graph: &GraphLevel,
    identifiers: &BTreeSet<GraphIdentifier>,
    operation: &'static str,
) -> Result<(), GraphError> {
    if let Some(identifier) = identifiers.iter().find(|identifier| {
        graph
            .nodes
            .get(*identifier)
            .is_some_and(|node| !node.is_removable())
    }) {
        return Err(GraphError::NodeOperationRestricted {
            node: identifier.clone(),
            operation,
        });
    }
    Ok(())
}

fn validate_graph_label(label: &str) -> Result<(), GraphError> {
    let label = label.trim();
    if label.is_empty() || label.chars().count() > 256 || label.chars().any(char::is_control) {
        return Err(GraphError::InvalidGraphLabel);
    }
    Ok(())
}

fn active_subgraph_definition_mut(
    document: &mut GraphDocument,
) -> Result<(&GraphIdentifier, &mut SubgraphDefinition), GraphError> {
    let (identifier, parent_path) = document
        .navigation
        .split_last()
        .ok_or(GraphError::AtRootGraph)?;
    let definition = graph_at_path_mut(&mut document.root, parent_path)?
        .definitions
        .get_mut(identifier)
        .ok_or_else(|| GraphError::UnknownSubgraph(identifier.clone()))?;
    Ok((identifier, definition))
}

fn subgraph_port_mut(
    definition: &mut SubgraphDefinition,
    direction: GraphSlotDirection,
    slot: usize,
) -> Result<&mut SubgraphPort, GraphError> {
    let identifier = definition.identifier.clone();
    match direction {
        GraphSlotDirection::Input => definition.inputs.get_mut(slot),
        GraphSlotDirection::Output => definition.outputs.get_mut(slot),
    }
    .ok_or(GraphError::UnknownSubgraphPort {
        definition: identifier,
        direction,
        slot,
    })
}

fn disconnect_subgraph_slot(
    document: &mut GraphDocument,
    direction: GraphSlotDirection,
    slot: usize,
) -> Result<(), GraphError> {
    let (_, definition) = active_subgraph_definition_mut(document)?;
    let port = subgraph_port_mut(definition, direction, slot)?;
    let internal_node = port.internal_node.take();
    port.internal_slot = 0;
    let boundary = subgraph_boundary_identifier(
        &definition.graph,
        match direction {
            GraphSlotDirection::Input => "inputNode",
            GraphSlotDirection::Output => "outputNode",
        },
    );
    let removed_links = definition
        .graph
        .links
        .values()
        .filter(|link| match direction {
            GraphSlotDirection::Input => boundary
                .as_ref()
                .is_some_and(|boundary| link.origin_node == *boundary && link.origin_slot == slot),
            GraphSlotDirection::Output => boundary
                .as_ref()
                .is_some_and(|boundary| link.target_node == *boundary && link.target_slot == slot),
        })
        .map(|link| link.identifier.clone())
        .collect::<BTreeSet<_>>();
    if internal_node.is_none() && removed_links.is_empty() {
        return Err(GraphError::SubgraphSlotHasNoLinks {
            definition: definition.identifier.clone(),
            direction,
            slot,
        });
    }
    definition
        .graph
        .links
        .retain(|identifier, _| !removed_links.contains(identifier));
    definition
        .graph
        .selection
        .links
        .retain(|identifier| !removed_links.contains(identifier));
    Ok(())
}

fn rename_subgraph_slot(
    document: &mut GraphDocument,
    direction: GraphSlotDirection,
    slot: usize,
    name: &str,
) -> Result<(), GraphError> {
    validate_graph_label(name)?;
    let name = name.trim().to_owned();
    let definition_identifier = {
        let (identifier, definition) = active_subgraph_definition_mut(document)?;
        let identifier = identifier.clone();
        let port = subgraph_port_mut(definition, direction, slot)?;
        port.name = name.clone();
        port.source_fields
            .insert("name".to_owned(), Value::String(name.clone()));
        identifier
    };
    sync_subgraph_instance_port_name(
        &mut document.root,
        &definition_identifier,
        direction,
        slot,
        &name,
    )?;
    Ok(())
}

fn remove_subgraph_slot(
    document: &mut GraphDocument,
    direction: GraphSlotDirection,
    slot: usize,
) -> Result<(), GraphError> {
    let definition_identifier = {
        let (identifier, definition) = active_subgraph_definition_mut(document)?;
        subgraph_port_mut(definition, direction, slot)?;
        remove_subgraph_boundary_slot(&mut definition.graph, direction, slot);
        match direction {
            GraphSlotDirection::Input => {
                definition.inputs.remove(slot);
            }
            GraphSlotDirection::Output => {
                definition.outputs.remove(slot);
            }
        }
        identifier.clone()
    };
    remove_subgraph_instance_slot(&mut document.root, &definition_identifier, direction, slot);
    Ok(())
}

fn remove_subgraph_boundary_slot(
    graph: &mut GraphLevel,
    direction: GraphSlotDirection,
    slot: usize,
) {
    let boundary = subgraph_boundary_identifier(
        graph,
        match direction {
            GraphSlotDirection::Input => "inputNode",
            GraphSlotDirection::Output => "outputNode",
        },
    );
    let Some(boundary) = boundary else {
        return;
    };
    let removed = graph
        .links
        .values()
        .filter(|link| match direction {
            GraphSlotDirection::Input => link.origin_node == boundary && link.origin_slot == slot,
            GraphSlotDirection::Output => link.target_node == boundary && link.target_slot == slot,
        })
        .map(|link| link.identifier.clone())
        .collect::<BTreeSet<_>>();
    graph
        .links
        .retain(|identifier, _| !removed.contains(identifier));
    graph
        .selection
        .links
        .retain(|identifier| !removed.contains(identifier));
    for link in graph.links.values_mut() {
        match direction {
            GraphSlotDirection::Input
                if link.origin_node == boundary && link.origin_slot > slot =>
            {
                link.origin_slot -= 1;
            }
            GraphSlotDirection::Output
                if link.target_node == boundary && link.target_slot > slot =>
            {
                link.target_slot -= 1;
            }
            _ => {}
        }
    }
}

fn sync_subgraph_instance_port_name(
    graph: &mut GraphLevel,
    definition_identifier: &GraphIdentifier,
    direction: GraphSlotDirection,
    slot: usize,
    name: &str,
) -> Result<(), GraphError> {
    for node in graph
        .nodes
        .values_mut()
        .filter(|node| node.subgraph_definition.as_ref() == Some(definition_identifier))
    {
        let port = match direction {
            GraphSlotDirection::Input => node.inputs.get_mut(slot),
            GraphSlotDirection::Output => node.outputs.get_mut(slot),
        }
        .ok_or_else(|| GraphError::InvalidSubgraphBoundary {
            instance: node.identifier.clone(),
            slot,
        })?;
        port.name = name.to_owned();
    }
    for definition in graph.definitions.values_mut() {
        sync_subgraph_instance_port_name(
            &mut definition.graph,
            definition_identifier,
            direction,
            slot,
            name,
        )?;
    }
    Ok(())
}

fn remove_subgraph_instance_slot(
    graph: &mut GraphLevel,
    definition_identifier: &GraphIdentifier,
    direction: GraphSlotDirection,
    slot: usize,
) {
    let instances = graph
        .nodes
        .values()
        .filter(|node| node.subgraph_definition.as_ref() == Some(definition_identifier))
        .map(|node| node.identifier.clone())
        .collect::<BTreeSet<_>>();
    let removed = graph
        .links
        .values()
        .filter(|link| match direction {
            GraphSlotDirection::Input => {
                instances.contains(&link.target_node) && link.target_slot == slot
            }
            GraphSlotDirection::Output => {
                instances.contains(&link.origin_node) && link.origin_slot == slot
            }
        })
        .map(|link| link.identifier.clone())
        .collect::<BTreeSet<_>>();
    graph
        .links
        .retain(|identifier, _| !removed.contains(identifier));
    graph
        .selection
        .links
        .retain(|identifier| !removed.contains(identifier));
    for node in graph
        .nodes
        .values_mut()
        .filter(|node| instances.contains(&node.identifier))
    {
        match direction {
            GraphSlotDirection::Input if slot < node.inputs.len() => {
                node.inputs.remove(slot);
            }
            GraphSlotDirection::Output if slot < node.outputs.len() => {
                node.outputs.remove(slot);
            }
            _ => {}
        }
    }
    for link in graph.links.values_mut() {
        match direction {
            GraphSlotDirection::Input
                if instances.contains(&link.target_node) && link.target_slot > slot =>
            {
                link.target_slot -= 1;
            }
            GraphSlotDirection::Output
                if instances.contains(&link.origin_node) && link.origin_slot > slot =>
            {
                link.origin_slot -= 1;
            }
            _ => {}
        }
    }
    for definition in graph.definitions.values_mut() {
        remove_subgraph_instance_slot(
            &mut definition.graph,
            definition_identifier,
            direction,
            slot,
        );
    }
}

fn remove_reroutes(
    graph: &mut GraphLevel,
    identifiers: &BTreeSet<GraphIdentifier>,
) -> Result<(), GraphError> {
    if let Some(identifier) = identifiers
        .iter()
        .find(|identifier| !graph.reroutes.contains_key(*identifier))
    {
        return Err(GraphError::UnknownReroute(identifier.clone()));
    }
    let replacement_parents = identifiers
        .iter()
        .map(|identifier| {
            let mut parent = graph
                .reroutes
                .get(identifier)
                .and_then(|reroute| reroute.parent.clone());
            let mut visited = BTreeSet::from([identifier.clone()]);
            while let Some(parent_identifier) = parent.clone() {
                if !visited.insert(parent_identifier.clone()) {
                    return Err(GraphError::RerouteCycle(identifier.clone()));
                }
                if !identifiers.contains(&parent_identifier) {
                    break;
                }
                parent = graph
                    .reroutes
                    .get(&parent_identifier)
                    .ok_or_else(|| GraphError::UnknownReroute(parent_identifier.clone()))?
                    .parent
                    .clone();
            }
            Ok((identifier.clone(), parent))
        })
        .collect::<Result<BTreeMap<_, _>, GraphError>>()?;
    for reroute in graph.reroutes.values_mut() {
        if let Some(parent) = reroute.parent.as_ref()
            && let Some(replacement) = replacement_parents.get(parent)
        {
            reroute.parent = replacement.clone();
        }
    }
    for link in graph.links.values_mut() {
        if let Some(parent) = link.parent_reroute.as_ref()
            && let Some(replacement) = replacement_parents.get(parent)
        {
            link.parent_reroute = replacement.clone();
        }
    }
    graph
        .reroutes
        .retain(|identifier, _| !identifiers.contains(identifier));
    graph
        .selection
        .reroutes
        .retain(|identifier| !identifiers.contains(identifier));
    Ok(())
}

fn connect(
    graph: &mut GraphLevel,
    mut link: GraphLink,
    replace_existing: bool,
) -> Result<(), GraphError> {
    if graph.links.contains_key(&link.identifier) {
        return Err(GraphError::DuplicateEntity(link.identifier));
    }
    let output = graph
        .nodes
        .get(&link.origin_node)
        .ok_or_else(|| GraphError::UnknownNode(link.origin_node.clone()))?
        .outputs
        .get(link.origin_slot)
        .ok_or_else(|| GraphError::UnknownPort {
            node: link.origin_node.clone(),
            slot: link.origin_slot,
        })?
        .clone();
    let target = graph
        .nodes
        .get(&link.target_node)
        .ok_or_else(|| GraphError::UnknownNode(link.target_node.clone()))?;
    let target_slot = link.target_slot;
    let input = target
        .inputs
        .get(target_slot)
        .cloned()
        .ok_or_else(|| GraphError::UnknownPort {
            node: link.target_node.clone(),
            slot: target_slot,
        })?;
    let expands_dynamic_input = input.dynamic && target_slot + 1 == target.inputs.len();
    if !input.port_type.accepts(&output.port_type) {
        return Err(GraphError::IncompatiblePorts {
            output: output.port_type.display_name(),
            input: input.port_type.display_name(),
        });
    }
    let occupied = graph
        .links
        .values()
        .filter(|candidate| {
            candidate.target_node == link.target_node && candidate.target_slot == target_slot
        })
        .map(|candidate| candidate.identifier.clone())
        .collect::<Vec<_>>();
    if !occupied.is_empty() && !input.multiple {
        if !replace_existing {
            return Err(GraphError::InputOccupied {
                node: link.target_node,
                slot: target_slot,
            });
        }
        for identifier in occupied {
            graph.links.remove(&identifier);
            graph.selection.links.remove(&identifier);
        }
    }
    if expands_dynamic_input {
        let inputs = &mut graph
            .nodes
            .get_mut(&link.target_node)
            .ok_or_else(|| GraphError::UnknownNode(link.target_node.clone()))?
            .inputs;
        let template = inputs.last_mut().ok_or_else(|| GraphError::UnknownPort {
            node: link.target_node.clone(),
            slot: target_slot,
        })?;
        template.dynamic = false;
        let mut next_template = input;
        next_template.dynamic = true;
        inputs.push(next_template);
    }
    link.target_slot = target_slot;
    link.type_name = output.port_type.display_name();
    graph.links.insert(link.identifier.clone(), link);
    Ok(())
}

fn link_intersects_rect(graph: &GraphLevel, link: &GraphLink, bounds: GraphRect) -> bool {
    let Some(origin) = graph.nodes.get(&link.origin_node) else {
        return false;
    };
    let Some(target) = graph.nodes.get(&link.target_node) else {
        return false;
    };
    let mut points = vec![GraphPoint {
        x: origin.position.x + origin.size.width,
        y: origin.position.y + 42.0 + link.origin_slot as f32 * 22.0,
    }];
    let mut reroute = link.parent_reroute.as_ref();
    let mut visited = BTreeSet::new();
    while let Some(identifier) = reroute {
        if !visited.insert(identifier.clone()) {
            return false;
        }
        let Some(current) = graph.reroutes.get(identifier) else {
            return false;
        };
        points.push(current.position);
        reroute = current.parent.as_ref();
    }
    points.push(GraphPoint {
        x: target.position.x,
        y: target.position.y + 42.0 + link.target_slot as f32 * 22.0,
    });
    points
        .windows(2)
        .any(|segment| segment_intersects_rect(segment[0], segment[1], bounds))
}

fn segment_intersects_rect(from: GraphPoint, to: GraphPoint, bounds: GraphRect) -> bool {
    if bounds.contains(from) || bounds.contains(to) {
        return true;
    }
    let left = bounds.origin.x;
    let right = bounds.origin.x + bounds.size.width;
    let top = bounds.origin.y;
    let bottom = bounds.origin.y + bounds.size.height;
    [
        (
            GraphPoint { x: left, y: top },
            GraphPoint { x: right, y: top },
        ),
        (
            GraphPoint { x: right, y: top },
            GraphPoint {
                x: right,
                y: bottom,
            },
        ),
        (
            GraphPoint {
                x: right,
                y: bottom,
            },
            GraphPoint { x: left, y: bottom },
        ),
        (
            GraphPoint { x: left, y: bottom },
            GraphPoint { x: left, y: top },
        ),
    ]
    .into_iter()
    .any(|(edge_start, edge_end)| segments_intersect(from, to, edge_start, edge_end))
}

fn segments_intersect(
    first_start: GraphPoint,
    first_end: GraphPoint,
    second_start: GraphPoint,
    second_end: GraphPoint,
) -> bool {
    const EPSILON: f32 = 0.0001;

    fn cross(first: GraphPoint, second: GraphPoint, third: GraphPoint) -> f32 {
        (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (third.x - first.x)
    }

    fn contains(start: GraphPoint, end: GraphPoint, point: GraphPoint) -> bool {
        const EPSILON: f32 = 0.0001;
        point.x >= start.x.min(end.x) - EPSILON
            && point.x <= start.x.max(end.x) + EPSILON
            && point.y >= start.y.min(end.y) - EPSILON
            && point.y <= start.y.max(end.y) + EPSILON
    }

    let first_side_start = cross(first_start, first_end, second_start);
    let first_side_end = cross(first_start, first_end, second_end);
    let second_side_start = cross(second_start, second_end, first_start);
    let second_side_end = cross(second_start, second_end, first_end);
    if first_side_start.abs() <= EPSILON && contains(first_start, first_end, second_start) {
        return true;
    }
    if first_side_end.abs() <= EPSILON && contains(first_start, first_end, second_end) {
        return true;
    }
    if second_side_start.abs() <= EPSILON && contains(second_start, second_end, first_start) {
        return true;
    }
    if second_side_end.abs() <= EPSILON && contains(second_start, second_end, first_end) {
        return true;
    }
    (first_side_start > EPSILON && first_side_end < -EPSILON
        || first_side_start < -EPSILON && first_side_end > EPSILON)
        && (second_side_start > EPSILON && second_side_end < -EPSILON
            || second_side_start < -EPSILON && second_side_end > EPSILON)
}

fn move_selection(
    graph: &mut GraphLevel,
    delta: GraphPoint,
    snap: Option<f32>,
) -> Result<(), GraphError> {
    let mut nodes_to_move = graph.selection.nodes.clone();
    for identifier in &graph.selection.groups {
        let group = graph
            .groups
            .get(identifier)
            .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?;
        if group.pinned {
            return Err(GraphError::PinnedEntity(identifier.clone()));
        }
        nodes_to_move.extend(group.node_ids.iter().cloned());
    }
    if let Some(identifier) = nodes_to_move
        .iter()
        .find(|identifier| graph.nodes.get(*identifier).is_some_and(|node| node.pinned))
    {
        return Err(GraphError::PinnedEntity(identifier.clone()));
    }
    let snap = snap.filter(|value| value.is_finite() && *value > 0.0);
    for identifier in &nodes_to_move {
        let node = graph
            .nodes
            .get_mut(identifier)
            .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
        node.position = snap_point(node.position.translated(delta), snap);
    }
    for identifier in &graph.selection.groups {
        let group = graph
            .groups
            .get_mut(identifier)
            .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?;
        group.bounds.origin = snap_point(group.bounds.origin.translated(delta), snap);
    }
    for identifier in &graph.selection.reroutes {
        let reroute = graph
            .reroutes
            .get_mut(identifier)
            .ok_or_else(|| GraphError::UnknownReroute(identifier.clone()))?;
        reroute.position = snap_point(reroute.position.translated(delta), snap);
    }
    Ok(())
}

fn snap_point(point: GraphPoint, snap: Option<f32>) -> GraphPoint {
    let Some(snap) = snap else {
        return point;
    };
    GraphPoint {
        x: (point.x / snap).round() * snap,
        y: (point.y / snap).round() * snap,
    }
}

fn toggle_nodes(
    graph: &mut GraphLevel,
    identifiers: &BTreeSet<GraphIdentifier>,
    toggle: NodeToggle,
) -> Result<(), GraphError> {
    validate_node_identifiers(graph, identifiers)?;
    if toggle == NodeToggle::Collapse
        && let Some(identifier) = identifiers.iter().find(|identifier| {
            graph
                .nodes
                .get(*identifier)
                .is_some_and(|node| !node.is_collapsible())
        })
    {
        return Err(GraphError::NodeOperationRestricted {
            node: identifier.clone(),
            operation: "collapse",
        });
    }
    for identifier in identifiers {
        let node = graph
            .nodes
            .get_mut(identifier)
            .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
        match toggle {
            NodeToggle::Mute => {
                node.mode = if node.mode == GraphNodeMode::Never {
                    GraphNodeMode::Always
                } else {
                    GraphNodeMode::Never
                };
            }
            NodeToggle::Bypass => {
                node.mode = if node.mode == GraphNodeMode::Bypass {
                    GraphNodeMode::Always
                } else {
                    GraphNodeMode::Bypass
                };
            }
            NodeToggle::Pin => node.pinned = !node.pinned,
            NodeToggle::Collapse => node.collapsed = !node.collapsed,
            NodeToggle::Disable => {
                node.mode = if node.mode == GraphNodeMode::OnEvent {
                    GraphNodeMode::Always
                } else {
                    GraphNodeMode::OnEvent
                };
            }
        }
    }
    Ok(())
}

fn toggle_groups(
    graph: &mut GraphLevel,
    identifiers: &BTreeSet<GraphIdentifier>,
    toggle: GroupToggle,
) -> Result<(), GraphError> {
    if identifiers.is_empty() {
        return Err(GraphError::EmptySelection);
    }
    if let Some(missing) = identifiers
        .iter()
        .find(|identifier| !graph.groups.contains_key(*identifier))
    {
        return Err(GraphError::UnknownGroup(missing.clone()));
    }
    for identifier in identifiers {
        let group = graph
            .groups
            .get_mut(identifier)
            .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?;
        match toggle {
            GroupToggle::Pin => group.pinned = !group.pinned,
            GroupToggle::Collapse => group.collapsed = !group.collapsed,
        }
    }
    Ok(())
}

fn set_selection(
    graph: &mut GraphLevel,
    selection: &GraphSelection,
    mode: SelectionMode,
) -> Result<(), GraphError> {
    validate_selection(graph, selection)?;
    match mode {
        SelectionMode::Replace => graph.selection = selection.clone(),
        SelectionMode::Add => {
            graph
                .selection
                .nodes
                .extend(selection.nodes.iter().cloned());
            graph
                .selection
                .links
                .extend(selection.links.iter().cloned());
            graph
                .selection
                .groups
                .extend(selection.groups.iter().cloned());
            graph
                .selection
                .reroutes
                .extend(selection.reroutes.iter().cloned());
        }
        SelectionMode::Toggle => {
            toggle_set(&mut graph.selection.nodes, &selection.nodes);
            toggle_set(&mut graph.selection.links, &selection.links);
            toggle_set(&mut graph.selection.groups, &selection.groups);
            toggle_set(&mut graph.selection.reroutes, &selection.reroutes);
        }
    }
    Ok(())
}

fn toggle_set<T: Clone + Ord>(target: &mut BTreeSet<T>, values: &BTreeSet<T>) {
    for value in values {
        if !target.remove(value) {
            target.insert(value.clone());
        }
    }
}

fn validate_selection(graph: &GraphLevel, selection: &GraphSelection) -> Result<(), GraphError> {
    if let Some(identifier) = selection
        .nodes
        .iter()
        .find(|identifier| !graph.nodes.contains_key(*identifier))
    {
        return Err(GraphError::UnknownNode(identifier.clone()));
    }
    if let Some(identifier) = selection
        .links
        .iter()
        .find(|identifier| !graph.links.contains_key(*identifier))
    {
        return Err(GraphError::UnknownLink(identifier.clone()));
    }
    if let Some(identifier) = selection
        .groups
        .iter()
        .find(|identifier| !graph.groups.contains_key(*identifier))
    {
        return Err(GraphError::UnknownGroup(identifier.clone()));
    }
    if let Some(identifier) = selection
        .reroutes
        .iter()
        .find(|identifier| !graph.reroutes.contains_key(*identifier))
    {
        return Err(GraphError::UnknownReroute(identifier.clone()));
    }
    Ok(())
}

fn validate_floating_reroute(reroute: &GraphReroute) -> Result<(), GraphError> {
    if reroute
        .floating_type
        .as_deref()
        .is_some_and(|slot_type| !matches!(slot_type, "input" | "output"))
    {
        return Err(GraphError::Serialization(
            "workflow reroute floating.slotType must be input or output".to_owned(),
        ));
    }
    if reroute
        .source_fields
        .get("floating")
        .and_then(Value::as_object)
        .and_then(|floating| floating.get("type"))
        .is_some_and(|port_type| !port_type.is_string())
    {
        return Err(GraphError::Serialization(
            "workflow reroute floating.type must be a string".to_owned(),
        ));
    }
    Ok(())
}

fn resolve_reroute_port_type(
    graph: &GraphLevel,
    identifier: &GraphIdentifier,
) -> Result<GraphPortType, GraphError> {
    if !graph.reroutes.contains_key(identifier) {
        return Err(GraphError::UnknownReroute(identifier.clone()));
    }

    let mut source_types = Vec::new();
    let mut target_types = Vec::new();
    let mut declared_types = Vec::new();
    for link in graph.links.values() {
        if !reroute_chain_contains(graph, link.parent_reroute.as_ref(), identifier)? {
            continue;
        }
        if let Some(port_type) = graph
            .nodes
            .get(&link.origin_node)
            .and_then(|node| node.outputs.get(link.origin_slot))
            .map(|port| port.port_type.clone())
        {
            source_types.push(port_type);
        }
        if let Some(port_type) = graph
            .nodes
            .get(&link.target_node)
            .and_then(|node| node.inputs.get(link.target_slot))
            .map(|port| port.port_type.clone())
        {
            target_types.push(port_type);
        }
        declared_types.push(GraphPortType::from_name(&link.type_name));
    }

    if let Some(floating_links) = graph
        .source_fields
        .get("floatingLinks")
        .and_then(Value::as_array)
    {
        for floating_link in floating_links {
            let Some(floating_link) = floating_link.as_object() else {
                return Err(GraphError::InvalidFloatingLink(
                    "floating link must be an object".to_owned(),
                ));
            };
            let parent = floating_link
                .get("parentId")
                .or_else(|| floating_link.get("parent_id"))
                .and_then(GraphIdentifier::from_value);
            if !reroute_chain_contains(graph, parent.as_ref(), identifier)? {
                continue;
            }
            if let Some(port_type) = floating_endpoint_port_type(
                graph,
                floating_link,
                "origin_id",
                "origin_node",
                "origin_slot",
                true,
            )? {
                source_types.push(port_type);
            }
            if let Some(port_type) = floating_endpoint_port_type(
                graph,
                floating_link,
                "target_id",
                "target_node",
                "target_slot",
                false,
            )? {
                target_types.push(port_type);
            }
            if let Some(type_name) = floating_link.get("type").and_then(Value::as_str) {
                declared_types.push(GraphPortType::from_name(type_name));
            }
        }
    }

    if let Some(port_type) = reconcile_reroute_types(identifier, source_types)? {
        return Ok(port_type);
    }
    if let Some(port_type) = reconcile_reroute_types(identifier, target_types)? {
        return Ok(port_type);
    }
    Ok(reconcile_reroute_types(identifier, declared_types)?.unwrap_or(GraphPortType::Any))
}

fn reroute_chain_contains(
    graph: &GraphLevel,
    start: Option<&GraphIdentifier>,
    target: &GraphIdentifier,
) -> Result<bool, GraphError> {
    let mut current = start;
    let mut visited = BTreeSet::new();
    while let Some(identifier) = current {
        if identifier == target {
            return Ok(true);
        }
        if !visited.insert(identifier.clone()) {
            return Err(GraphError::RerouteCycle(identifier.clone()));
        }
        current = graph
            .reroutes
            .get(identifier)
            .ok_or_else(|| GraphError::UnknownReroute(identifier.clone()))?
            .parent
            .as_ref();
    }
    Ok(false)
}

fn floating_endpoint_port_type(
    graph: &GraphLevel,
    link: &Map<String, Value>,
    identifier_field: &str,
    identifier_alias: &str,
    slot_field: &str,
    output: bool,
) -> Result<Option<GraphPortType>, GraphError> {
    let Some(identifier) = link
        .get(identifier_field)
        .or_else(|| link.get(identifier_alias))
        .and_then(GraphIdentifier::from_value)
    else {
        return Ok(None);
    };
    let Some(node) = graph.nodes.get(&identifier) else {
        return Ok(None);
    };
    let slot = link
        .get(slot_field)
        .and_then(Value::as_u64)
        .and_then(|slot| usize::try_from(slot).ok())
        .ok_or_else(|| {
            GraphError::InvalidFloatingLink(format!(
                "floating link {slot_field} must identify a real endpoint slot"
            ))
        })?;
    let port = if output {
        node.outputs.get(slot)
    } else {
        node.inputs.get(slot)
    }
    .ok_or_else(|| GraphError::UnknownPort {
        node: identifier,
        slot,
    })?;
    Ok(Some(port.port_type.clone()))
}

fn reconcile_reroute_types(
    identifier: &GraphIdentifier,
    candidates: Vec<GraphPortType>,
) -> Result<Option<GraphPortType>, GraphError> {
    let mut candidates = candidates
        .into_iter()
        .filter(|candidate| *candidate != GraphPortType::Any);
    let Some(resolved) = candidates.next() else {
        return Ok(None);
    };
    if candidates.any(|candidate| candidate != resolved) {
        return Err(GraphError::ConflictingReroutePortTypes(identifier.clone()));
    }
    Ok(Some(resolved))
}

fn fit_group(
    graph: &mut GraphLevel,
    identifier: &GraphIdentifier,
    padding: f32,
) -> Result<(), GraphError> {
    validate_group_padding(padding)?;
    let group = graph
        .groups
        .get(identifier)
        .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?;
    if group.pinned {
        return Err(GraphError::PinnedEntity(identifier.clone()));
    }
    let bounds = padded_node_bounds(graph, &group.node_ids, padding)?;
    let group = graph
        .groups
        .get_mut(identifier)
        .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?;
    group.bounds = bounds;
    Ok(())
}

fn validate_group_padding(padding: f32) -> Result<(), GraphError> {
    if padding.is_finite() && padding >= 0.0 {
        Ok(())
    } else {
        Err(GraphError::InvalidGroupPadding(padding))
    }
}

fn padded_node_bounds(
    graph: &GraphLevel,
    identifiers: &BTreeSet<GraphIdentifier>,
    padding: f32,
) -> Result<GraphRect, GraphError> {
    let bounds = bounds_for_nodes(graph, identifiers).ok_or(GraphError::EmptySelection)?;
    let bounds = GraphRect {
        origin: GraphPoint {
            x: bounds.origin.x - padding,
            y: bounds.origin.y - padding,
        },
        size: GraphSize {
            width: bounds.size.width + padding * 2.0,
            height: bounds.size.height + padding * 2.0,
        },
    };
    bounds.validate()?;
    Ok(bounds)
}

fn bounds_for_nodes(
    graph: &GraphLevel,
    identifiers: &BTreeSet<GraphIdentifier>,
) -> Option<GraphRect> {
    let mut nodes = identifiers
        .iter()
        .filter_map(|identifier| graph.nodes.get(identifier));
    let first = nodes.next()?;
    let mut left = first.position.x;
    let mut top = first.position.y;
    let mut right = first.position.x + first.size.width;
    let mut bottom = first.position.y + first.size.height;
    for node in nodes {
        left = left.min(node.position.x);
        top = top.min(node.position.y);
        right = right.max(node.position.x + node.size.width);
        bottom = bottom.max(node.position.y + node.size.height);
    }
    Some(GraphRect {
        origin: GraphPoint { x: left, y: top },
        size: GraphSize {
            width: right - left,
            height: bottom - top,
        },
    })
}

fn node_center(node: &GraphNode) -> GraphPoint {
    GraphPoint {
        x: node.position.x + node.size.width / 2.0,
        y: node.position.y + node.size.height / 2.0,
    }
}

fn derived_group_membership(level: &GraphLevel, bounds: GraphRect) -> BTreeSet<GraphIdentifier> {
    level
        .nodes
        .values()
        .filter(|node| bounds.contains(node_center(node)))
        .map(|node| node.identifier.clone())
        .collect()
}

fn recompute_group_memberships(level: &mut GraphLevel) {
    let memberships = level
        .groups
        .iter()
        .map(|(identifier, group)| {
            (
                identifier.clone(),
                derived_group_membership(level, group.bounds),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for (identifier, membership) in memberships {
        if let Some(group) = level.groups.get_mut(&identifier) {
            group.node_ids = membership;
        }
    }
    for definition in level.definitions.values_mut() {
        recompute_group_memberships(&mut definition.graph);
    }
}

fn reparent_reroute(
    graph: &mut GraphLevel,
    identifier: &GraphIdentifier,
    parent: Option<GraphIdentifier>,
) -> Result<(), GraphError> {
    if let Some(parent) = &parent {
        if parent == identifier {
            return Err(GraphError::RerouteCycle(identifier.clone()));
        }
        if !graph.reroutes.contains_key(parent) {
            return Err(GraphError::UnknownReroute(parent.clone()));
        }
        let mut current = Some(parent.clone());
        let mut visited = BTreeSet::new();
        while let Some(candidate) = current {
            if !visited.insert(candidate.clone()) || candidate == *identifier {
                return Err(GraphError::RerouteCycle(identifier.clone()));
            }
            current = graph
                .reroutes
                .get(&candidate)
                .and_then(|reroute| reroute.parent.clone());
        }
    }
    graph
        .reroutes
        .get_mut(identifier)
        .ok_or_else(|| GraphError::UnknownReroute(identifier.clone()))?
        .parent = parent;
    Ok(())
}

fn remove_subgraph_definition(
    graph: &mut GraphLevel,
    definition_identifier: &GraphIdentifier,
    remove_instances: bool,
) -> Result<(), GraphError> {
    if !graph.definitions.contains_key(definition_identifier) {
        return Err(GraphError::UnknownSubgraph(definition_identifier.clone()));
    }
    let instances = graph
        .nodes
        .values()
        .filter(|node| node.subgraph_definition.as_ref() == Some(definition_identifier))
        .map(|node| node.identifier.clone())
        .collect::<BTreeSet<_>>();
    let mut nested_instances = BTreeSet::new();
    for (_, definition) in graph
        .definitions
        .iter()
        .filter(|(identifier, _)| *identifier != definition_identifier)
    {
        collect_subgraph_instance_references(
            &definition.graph,
            definition_identifier,
            &mut nested_instances,
        );
    }
    if !nested_instances.is_empty() || (!instances.is_empty() && !remove_instances) {
        return Err(GraphError::SubgraphDefinitionInUse {
            definition: definition_identifier.clone(),
            instances: instances.union(&nested_instances).cloned().collect(),
        });
    }
    if remove_instances {
        validate_nodes_removable(graph, &instances, "remove")?;
        graph
            .nodes
            .retain(|identifier, _| !instances.contains(identifier));
        graph.links.retain(|_, link| {
            !instances.contains(&link.origin_node) && !instances.contains(&link.target_node)
        });
        for group in graph.groups.values_mut() {
            group
                .node_ids
                .retain(|identifier| !instances.contains(identifier));
        }
        graph
            .selection
            .nodes
            .retain(|identifier| !instances.contains(identifier));
        graph
            .selection
            .links
            .retain(|identifier| graph.links.contains_key(identifier));
    }
    graph.definitions.remove(definition_identifier);
    Ok(())
}

fn collect_subgraph_instance_references(
    graph: &GraphLevel,
    definition_identifier: &GraphIdentifier,
    instances: &mut BTreeSet<GraphIdentifier>,
) {
    instances.extend(graph.nodes.values().filter_map(|node| {
        (node.subgraph_definition.as_ref() == Some(definition_identifier))
            .then_some(node.identifier.clone())
    }));
    for definition in graph.definitions.values() {
        collect_subgraph_instance_references(&definition.graph, definition_identifier, instances);
    }
}

fn set_subgraph_widget_exposure(
    graph: &mut GraphLevel,
    definition_identifier: &GraphIdentifier,
    internal_node: &GraphIdentifier,
    widget_identifier: &str,
    exposed: bool,
) -> Result<(), GraphError> {
    let definition = graph
        .definitions
        .get(definition_identifier)
        .ok_or_else(|| GraphError::UnknownSubgraph(definition_identifier.clone()))?;
    let existing_identifier = definition
        .exposed_widgets
        .iter()
        .find(|exposure| {
            exposure.internal_node == *internal_node
                && exposure.internal_widget == widget_identifier
        })
        .map(|exposure| exposure.identifier.clone());
    let exposure_identifier = existing_identifier.unwrap_or_else(|| {
        subgraph_widget_identifier(definition_identifier, internal_node, widget_identifier)
    });
    let source_widget = if exposed {
        let node = definition
            .graph
            .nodes
            .get(internal_node)
            .ok_or_else(|| GraphError::UnknownNode(internal_node.clone()))?;
        if node
            .inputs
            .iter()
            .any(|input| input.name == widget_identifier && input.dynamic)
        {
            return Err(GraphError::UnsupportedDynamicSubgraphInput {
                node: internal_node.clone(),
                port: widget_identifier.to_owned(),
            });
        }
        Some(
            node.widgets
                .iter()
                .find(|widget| widget.identifier == widget_identifier)
                .cloned()
                .ok_or_else(|| GraphError::UnknownWidget(widget_identifier.to_owned()))?,
        )
    } else {
        None
    };
    let definition = graph
        .definitions
        .get_mut(definition_identifier)
        .ok_or_else(|| GraphError::UnknownSubgraph(definition_identifier.clone()))?;
    if let Some(mut source_widget) = source_widget {
        source_widget.identifier = exposure_identifier.clone();
        let exposure = SubgraphExposedWidget {
            identifier: exposure_identifier.clone(),
            internal_node: internal_node.clone(),
            internal_widget: widget_identifier.to_owned(),
            widget: source_widget.clone(),
            unknown: BTreeMap::new(),
        };
        if let Some(existing) = definition
            .exposed_widgets
            .iter_mut()
            .find(|candidate| candidate.identifier == exposure_identifier)
        {
            let unknown = std::mem::take(&mut existing.unknown);
            *existing = SubgraphExposedWidget {
                unknown,
                ..exposure
            };
        } else {
            definition.exposed_widgets.push(exposure);
        }
        for instance in graph
            .nodes
            .values_mut()
            .filter(|node| node.subgraph_definition.as_ref() == Some(definition_identifier))
        {
            if let Some(existing) = instance
                .widgets
                .iter_mut()
                .find(|widget| widget.identifier == exposure_identifier)
            {
                if existing.kind == source_widget.kind {
                    source_widget.value = existing.value.clone();
                    source_widget.prompt_value = existing.prompt_value.clone();
                }
                *existing = source_widget.clone();
            } else {
                instance.widgets.push(source_widget.clone());
            }
        }
    } else {
        definition.exposed_widgets.retain(|exposure| {
            !(exposure.internal_node == *internal_node
                && exposure.internal_widget == widget_identifier)
        });
        for instance in graph
            .nodes
            .values_mut()
            .filter(|node| node.subgraph_definition.as_ref() == Some(definition_identifier))
        {
            instance
                .widgets
                .retain(|widget| widget.identifier != exposure_identifier);
        }
    }
    Ok(())
}

fn subgraph_widget_identifier(
    definition_identifier: &GraphIdentifier,
    internal_node: &GraphIdentifier,
    widget_identifier: &str,
) -> String {
    let namespace = Uuid::new_v5(
        &Uuid::NAMESPACE_OID,
        format!("subgraph:{}", definition_identifier.text()).as_bytes(),
    );
    Uuid::new_v5(
        &namespace,
        format!("widget:{}:{widget_identifier}", internal_node.text()).as_bytes(),
    )
    .to_string()
}

fn layout_selection(
    graph: &mut GraphLevel,
    operation: LayoutOperation,
    spacing: f32,
) -> Result<(), GraphError> {
    if !spacing.is_finite() || spacing < 0.0 {
        return Err(GraphError::NonFiniteGeometry);
    }
    if graph.selection.nodes.is_empty() {
        return Err(GraphError::EmptySelection);
    }
    if let Some(identifier) = graph
        .selection
        .nodes
        .iter()
        .find(|identifier| graph.nodes.get(*identifier).is_some_and(|node| node.pinned))
    {
        return Err(GraphError::PinnedEntity(identifier.clone()));
    }
    let identifiers = graph.selection.nodes.iter().cloned().collect::<Vec<_>>();
    let bounds =
        bounds_for_nodes(graph, &graph.selection.nodes).ok_or(GraphError::EmptySelection)?;
    match operation {
        LayoutOperation::AlignLeft
        | LayoutOperation::AlignRight
        | LayoutOperation::AlignTop
        | LayoutOperation::AlignBottom
        | LayoutOperation::AlignHorizontalCenters
        | LayoutOperation::AlignVerticalCenters => {
            for identifier in &identifiers {
                let node = graph
                    .nodes
                    .get_mut(identifier)
                    .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
                match operation {
                    LayoutOperation::AlignLeft => node.position.x = bounds.origin.x,
                    LayoutOperation::AlignRight => {
                        node.position.x = bounds.origin.x + bounds.size.width - node.size.width
                    }
                    LayoutOperation::AlignTop => node.position.y = bounds.origin.y,
                    LayoutOperation::AlignBottom => {
                        node.position.y = bounds.origin.y + bounds.size.height - node.size.height
                    }
                    LayoutOperation::AlignHorizontalCenters => {
                        node.position.x =
                            bounds.origin.x + (bounds.size.width - node.size.width) / 2.0
                    }
                    LayoutOperation::AlignVerticalCenters => {
                        node.position.y =
                            bounds.origin.y + (bounds.size.height - node.size.height) / 2.0
                    }
                    _ => {}
                }
            }
        }
        LayoutOperation::DistributeHorizontally => {
            distribute_nodes(graph, identifiers, true, spacing)?;
        }
        LayoutOperation::DistributeVertically => {
            distribute_nodes(graph, identifiers, false, spacing)?;
        }
        LayoutOperation::ArrangeGrid => {
            let columns = (identifiers.len() as f64).sqrt().ceil() as usize;
            for (index, identifier) in identifiers.iter().enumerate() {
                let node = graph
                    .nodes
                    .get_mut(identifier)
                    .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
                node.position = GraphPoint {
                    x: bounds.origin.x + (index % columns) as f32 * (node.size.width + spacing),
                    y: bounds.origin.y + (index / columns) as f32 * (node.size.height + spacing),
                };
            }
        }
    }
    Ok(())
}

fn distribute_nodes(
    graph: &mut GraphLevel,
    mut identifiers: Vec<GraphIdentifier>,
    horizontal: bool,
    spacing: f32,
) -> Result<(), GraphError> {
    identifiers.sort_by(|left, right| {
        let left_position = graph
            .nodes
            .get(left)
            .map(|node| {
                if horizontal {
                    node.position.x
                } else {
                    node.position.y
                }
            })
            .unwrap_or_default();
        let right_position = graph
            .nodes
            .get(right)
            .map(|node| {
                if horizontal {
                    node.position.x
                } else {
                    node.position.y
                }
            })
            .unwrap_or_default();
        left_position
            .total_cmp(&right_position)
            .then_with(|| left.cmp(right))
    });
    let mut cursor = identifiers
        .first()
        .and_then(|identifier| graph.nodes.get(identifier))
        .map(|node| {
            if horizontal {
                node.position.x
            } else {
                node.position.y
            }
        })
        .ok_or(GraphError::EmptySelection)?;
    for identifier in identifiers {
        let node = graph
            .nodes
            .get_mut(&identifier)
            .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
        if horizontal {
            node.position.x = cursor;
            cursor += node.size.width + spacing;
        } else {
            node.position.y = cursor;
            cursor += node.size.height + spacing;
        }
    }
    Ok(())
}

fn zoom_viewport(
    graph: &mut GraphLevel,
    factor: f32,
    anchor: GraphPoint,
) -> Result<(), GraphError> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(GraphError::InvalidViewportScale(factor));
    }
    anchor.validate()?;
    let before = graph.viewport.screen_to_graph(anchor);
    graph.viewport.scale =
        (graph.viewport.scale * factor).clamp(MIN_VIEWPORT_SCALE, MAX_VIEWPORT_SCALE);
    graph.viewport.offset = GraphPoint {
        x: anchor.x - before.x * graph.viewport.scale,
        y: anchor.y - before.y * graph.viewport.scale,
    };
    Ok(())
}

fn fit_viewport(
    graph: &mut GraphLevel,
    bounds: GraphRect,
    available: GraphSize,
    padding: f32,
) -> Result<(), GraphError> {
    bounds.validate()?;
    available.validate()?;
    if !padding.is_finite() || padding < 0.0 {
        return Err(GraphError::NonFiniteGeometry);
    }
    let width = (available.width - padding * 2.0).max(1.0);
    let height = (available.height - padding * 2.0).max(1.0);
    let scale = (width / bounds.size.width)
        .min(height / bounds.size.height)
        .clamp(MIN_VIEWPORT_SCALE, MAX_VIEWPORT_SCALE);
    graph.viewport.scale = scale;
    graph.viewport.offset = GraphPoint {
        x: (available.width - bounds.size.width * scale) / 2.0 - bounds.origin.x * scale,
        y: (available.height - bounds.size.height * scale) / 2.0 - bounds.origin.y * scale,
    };
    Ok(())
}

fn conversion_node_identifiers(
    graph: &GraphLevel,
) -> Result<BTreeSet<GraphIdentifier>, GraphError> {
    validate_selection(graph, &graph.selection)?;
    let selected_item_count =
        graph.selection.nodes.len() + graph.selection.groups.len() + graph.selection.reroutes.len();
    if selected_item_count == 0 {
        return Err(GraphError::EmptySelection);
    }
    if selected_item_count == 1
        && let Some(identifier) = graph.selection.nodes.first()
        && graph
            .nodes
            .get(identifier)
            .is_some_and(|node| node.subgraph_definition.is_some())
    {
        return Err(GraphError::InvalidSubgraphConversion);
    }

    let mut identifiers = graph.selection.nodes.clone();
    for group_identifier in &graph.selection.groups {
        let group = graph
            .groups
            .get(group_identifier)
            .ok_or_else(|| GraphError::UnknownGroup(group_identifier.clone()))?;
        identifiers.extend(group.node_ids.iter().cloned());
    }
    if identifiers.is_empty() {
        return Err(GraphError::InvalidSubgraphConversion);
    }
    validate_nodes_removable(graph, &identifiers, "convert to subgraph")?;
    Ok(identifiers)
}

fn convert_selection_to_subgraph(
    graph: &mut GraphLevel,
    definition_identifier: GraphIdentifier,
    instance_identifier: GraphIdentifier,
    name: String,
) -> Result<(), GraphError> {
    let selected = conversion_node_identifiers(graph)?;
    if graph.definitions.contains_key(&definition_identifier)
        || graph.nodes.contains_key(&instance_identifier)
    {
        return Err(GraphError::DuplicateEntity(definition_identifier));
    }
    let bounds = bounds_for_nodes(graph, &selected).ok_or(GraphError::EmptySelection)?;
    let mut inner = GraphLevel {
        viewport: graph.viewport.clone(),
        ..GraphLevel::default()
    };
    for identifier in &selected {
        let mut node = graph
            .nodes
            .remove(identifier)
            .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
        node.position = GraphPoint {
            x: node.position.x - bounds.origin.x,
            y: node.position.y - bounds.origin.y,
        };
        inner.nodes.insert(identifier.clone(), node);
    }
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut instance_inputs = Vec::new();
    let mut instance_outputs = Vec::new();
    let links = graph.links.values().cloned().collect::<Vec<_>>();
    for mut link in links {
        let origin_selected = selected.contains(&link.origin_node);
        let target_selected = selected.contains(&link.target_node);
        if origin_selected && target_selected {
            graph.links.remove(&link.identifier);
            inner.links.insert(link.identifier.clone(), link);
        } else if !origin_selected && target_selected {
            let port_type = inner
                .nodes
                .get(&link.target_node)
                .and_then(|node| node.inputs.get(link.target_slot))
                .map(|port| port.port_type.clone())
                .unwrap_or(GraphPortType::Any);
            let name = format!("input-{}", inputs.len());
            inputs.push(SubgraphPort {
                identifier: subgraph_port_identifier(
                    &definition_identifier,
                    "input",
                    inputs.len(),
                    &name,
                ),
                name: name.clone(),
                port_type: port_type.clone(),
                internal_node: Some(link.target_node.clone()),
                internal_slot: link.target_slot,
                source_fields: Map::new(),
            });
            instance_inputs.push(GraphPort::new(name, port_type));
            link.target_node = instance_identifier.clone();
            link.target_slot = instance_inputs.len() - 1;
            graph.links.insert(link.identifier.clone(), link);
        } else if origin_selected && !target_selected {
            let port_type = inner
                .nodes
                .get(&link.origin_node)
                .and_then(|node| node.outputs.get(link.origin_slot))
                .map(|port| port.port_type.clone())
                .unwrap_or(GraphPortType::Any);
            let name = format!("output-{}", outputs.len());
            outputs.push(SubgraphPort {
                identifier: subgraph_port_identifier(
                    &definition_identifier,
                    "output",
                    outputs.len(),
                    &name,
                ),
                name: name.clone(),
                port_type: port_type.clone(),
                internal_node: Some(link.origin_node.clone()),
                internal_slot: link.origin_slot,
                source_fields: Map::new(),
            });
            instance_outputs.push(GraphPort::new(name, port_type));
            link.origin_node = instance_identifier.clone();
            link.origin_slot = instance_outputs.len() - 1;
            graph.links.insert(link.identifier.clone(), link);
        }
    }
    let mut referenced_reroutes = inner
        .links
        .values()
        .filter_map(|link| link.parent_reroute.clone())
        .chain(graph.selection.reroutes.iter().cloned())
        .collect::<BTreeSet<_>>();
    let mut pending_reroutes = referenced_reroutes.iter().cloned().collect::<Vec<_>>();
    while let Some(identifier) = pending_reroutes.pop() {
        if let Some(parent) = graph
            .reroutes
            .get(&identifier)
            .and_then(|reroute| reroute.parent.clone())
            && referenced_reroutes.insert(parent.clone())
        {
            pending_reroutes.push(parent);
        }
    }
    for identifier in referenced_reroutes {
        if let Some(mut reroute) = graph.reroutes.remove(&identifier) {
            reroute.position = GraphPoint {
                x: reroute.position.x - bounds.origin.x,
                y: reroute.position.y - bounds.origin.y,
            };
            inner.reroutes.insert(identifier, reroute);
        }
    }
    let selected_groups = graph
        .groups
        .iter()
        .filter(|(_, group)| group.node_ids.is_subset(&selected))
        .map(|(identifier, _)| identifier.clone())
        .collect::<Vec<_>>();
    for identifier in selected_groups {
        if let Some(mut group) = graph.groups.remove(&identifier) {
            group.bounds.origin = GraphPoint {
                x: group.bounds.origin.x - bounds.origin.x,
                y: group.bounds.origin.y - bounds.origin.y,
            };
            inner.groups.insert(identifier, group);
        }
    }
    for group in graph.groups.values_mut() {
        group
            .node_ids
            .retain(|identifier| !selected.contains(identifier));
    }
    let referenced_definitions = inner
        .nodes
        .values()
        .filter_map(|node| node.subgraph_definition.clone())
        .collect::<BTreeSet<_>>();
    for identifier in referenced_definitions {
        let definition = graph
            .definitions
            .get(&identifier)
            .cloned()
            .ok_or_else(|| GraphError::UnknownSubgraph(identifier.clone()))?;
        inner.definitions.insert(identifier, definition);
    }
    let definition = SubgraphDefinition {
        identifier: definition_identifier.clone(),
        name: name.clone(),
        graph: Box::new(inner),
        inputs,
        outputs,
        published: false,
        description: String::new(),
        search_aliases: Vec::new(),
        exposed_widgets: Vec::new(),
        graph_inline: true,
        unknown: BTreeMap::new(),
    };
    let mut instance = GraphNode::new(
        instance_identifier.clone(),
        "SimSubgraph",
        name,
        bounds.origin,
    );
    instance.inputs = instance_inputs;
    instance.outputs = instance_outputs;
    instance.subgraph_definition = Some(definition_identifier.clone());
    instance.size = GraphSize {
        width: 260.0,
        height: 100.0 + (instance.inputs.len().max(instance.outputs.len()) as f32 * 24.0),
    };
    graph.definitions.insert(definition_identifier, definition);
    graph.nodes.insert(instance_identifier.clone(), instance);
    graph.selection = GraphSelection {
        nodes: BTreeSet::from([instance_identifier]),
        ..GraphSelection::default()
    };
    Ok(())
}

fn subgraph_port_identifier(
    definition_identifier: &GraphIdentifier,
    direction: &str,
    index: usize,
    name: &str,
) -> String {
    let namespace = match definition_identifier {
        GraphIdentifier::String(identifier) => Uuid::parse_str(identifier)
            .unwrap_or_else(|_| Uuid::new_v5(&Uuid::NAMESPACE_OID, identifier.as_bytes())),
        GraphIdentifier::Integer(identifier) => Uuid::new_v5(
            &Uuid::NAMESPACE_OID,
            format!("integer-subgraph-{identifier}").as_bytes(),
        ),
    };
    Uuid::new_v5(&namespace, format!("{direction}:{index}:{name}").as_bytes()).to_string()
}

fn unpack_subgraph(
    document: &mut GraphDocument,
    instance_identifier: &GraphIdentifier,
) -> Result<(), GraphError> {
    let graph = document.active_graph()?;
    let instance = graph
        .nodes
        .get(instance_identifier)
        .cloned()
        .ok_or_else(|| GraphError::UnknownNode(instance_identifier.clone()))?;
    if !instance.is_removable() {
        return Err(GraphError::NodeOperationRestricted {
            node: instance_identifier.clone(),
            operation: "unpack",
        });
    }
    let definition_identifier = instance
        .subgraph_definition
        .clone()
        .ok_or_else(|| GraphError::NotSubgraphInstance(instance_identifier.clone()))?;
    let definition = graph
        .definitions
        .get(&definition_identifier)
        .cloned()
        .ok_or_else(|| GraphError::UnknownSubgraph(definition_identifier.clone()))?;
    let nested_definitions = definition.graph.definitions.clone();
    let mut node_mapping = BTreeMap::new();
    for identifier in definition.graph.nodes.keys() {
        node_mapping.insert(identifier.clone(), document.allocate_identifier());
    }
    let mut link_mapping = BTreeMap::new();
    for identifier in definition.graph.links.keys() {
        link_mapping.insert(identifier.clone(), document.allocate_identifier());
    }
    let mut group_mapping = BTreeMap::new();
    for identifier in definition.graph.groups.keys() {
        group_mapping.insert(identifier.clone(), document.allocate_identifier());
    }
    let mut reroute_mapping = BTreeMap::new();
    for identifier in definition.graph.reroutes.keys() {
        reroute_mapping.insert(identifier.clone(), document.allocate_identifier());
    }
    let graph = document.active_graph_mut()?;
    let boundary_links = graph
        .links
        .values()
        .filter(|link| {
            link.origin_node == *instance_identifier || link.target_node == *instance_identifier
        })
        .cloned()
        .collect::<Vec<_>>();
    graph.nodes.remove(instance_identifier);
    graph.links.retain(|_, link| {
        link.origin_node != *instance_identifier && link.target_node != *instance_identifier
    });
    let mut unpacked = BTreeSet::new();
    for (identifier, mut node) in definition.graph.nodes {
        let mapped = node_mapping
            .get(&identifier)
            .cloned()
            .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
        node.identifier = mapped.clone();
        node.position = node.position.translated(instance.position);
        graph.nodes.insert(mapped.clone(), node);
        unpacked.insert(mapped);
    }
    for (identifier, mut link) in definition.graph.links {
        link.identifier = link_mapping
            .get(&identifier)
            .cloned()
            .ok_or_else(|| GraphError::UnknownLink(identifier.clone()))?;
        link.origin_node = node_mapping
            .get(&link.origin_node)
            .cloned()
            .ok_or_else(|| GraphError::UnknownNode(link.origin_node.clone()))?;
        link.target_node = node_mapping
            .get(&link.target_node)
            .cloned()
            .ok_or_else(|| GraphError::UnknownNode(link.target_node.clone()))?;
        link.parent_reroute = link
            .parent_reroute
            .as_ref()
            .map(|identifier| {
                reroute_mapping
                    .get(identifier)
                    .cloned()
                    .ok_or_else(|| GraphError::UnknownReroute(identifier.clone()))
            })
            .transpose()?;
        graph.links.insert(link.identifier.clone(), link);
    }
    for (identifier, mut group) in definition.graph.groups {
        group.identifier = group_mapping
            .get(&identifier)
            .cloned()
            .ok_or_else(|| GraphError::UnknownGroup(identifier.clone()))?;
        group.bounds.origin = group.bounds.origin.translated(instance.position);
        group.node_ids = group
            .node_ids
            .iter()
            .map(|identifier| {
                node_mapping
                    .get(identifier)
                    .cloned()
                    .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))
            })
            .collect::<Result<_, _>>()?;
        graph.groups.insert(group.identifier.clone(), group);
    }
    for (identifier, mut reroute) in definition.graph.reroutes {
        reroute.identifier = reroute_mapping
            .get(&identifier)
            .cloned()
            .ok_or_else(|| GraphError::UnknownReroute(identifier.clone()))?;
        reroute.position = reroute.position.translated(instance.position);
        reroute.parent = reroute
            .parent
            .as_ref()
            .map(|parent| {
                reroute_mapping
                    .get(parent)
                    .cloned()
                    .ok_or_else(|| GraphError::UnknownReroute(parent.clone()))
            })
            .transpose()?;
        graph.reroutes.insert(reroute.identifier.clone(), reroute);
    }
    for mut link in boundary_links {
        if link.target_node == *instance_identifier {
            let port = definition.inputs.get(link.target_slot).ok_or_else(|| {
                GraphError::InvalidSubgraphBoundary {
                    instance: instance_identifier.clone(),
                    slot: link.target_slot,
                }
            })?;
            let internal_node =
                port.internal_node
                    .as_ref()
                    .ok_or_else(|| GraphError::InvalidSubgraphBoundary {
                        instance: instance_identifier.clone(),
                        slot: link.target_slot,
                    })?;
            link.target_node = node_mapping
                .get(internal_node)
                .cloned()
                .ok_or_else(|| GraphError::UnknownNode(internal_node.clone()))?;
            link.target_slot = port.internal_slot;
        }
        if link.origin_node == *instance_identifier {
            let port = definition.outputs.get(link.origin_slot).ok_or_else(|| {
                GraphError::InvalidSubgraphBoundary {
                    instance: instance_identifier.clone(),
                    slot: link.origin_slot,
                }
            })?;
            let internal_node =
                port.internal_node
                    .as_ref()
                    .ok_or_else(|| GraphError::InvalidSubgraphBoundary {
                        instance: instance_identifier.clone(),
                        slot: link.origin_slot,
                    })?;
            link.origin_node = node_mapping
                .get(internal_node)
                .cloned()
                .ok_or_else(|| GraphError::UnknownNode(internal_node.clone()))?;
            link.origin_slot = port.internal_slot;
        }
        if let Some(mapped) = link
            .parent_reroute
            .as_ref()
            .and_then(|identifier| reroute_mapping.get(identifier).cloned())
        {
            link.parent_reroute = Some(mapped);
        }
        graph.links.insert(link.identifier.clone(), link);
    }
    for (identifier, nested_definition) in nested_definitions {
        match graph.definitions.get(&identifier) {
            Some(existing) if existing == &nested_definition => {}
            Some(_) => return Err(GraphError::DuplicateEntity(identifier)),
            None => {
                graph.definitions.insert(identifier, nested_definition);
            }
        }
    }
    graph.selection = GraphSelection {
        nodes: unpacked,
        ..GraphSelection::default()
    };
    Ok(())
}

fn reconcile_node(
    graph: &mut GraphLevel,
    identifier: &GraphIdentifier,
    inputs: &[GraphPort],
    outputs: &[GraphPort],
    widgets: &[GraphWidget],
    confirm_discard: bool,
) -> Result<(), GraphError> {
    let node = graph
        .nodes
        .get(identifier)
        .cloned()
        .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
    let input_names = inputs
        .iter()
        .map(|port| port.name.as_str())
        .collect::<BTreeSet<_>>();
    let output_names = outputs
        .iter()
        .map(|port| port.name.as_str())
        .collect::<BTreeSet<_>>();
    let widget_names = widgets
        .iter()
        .map(|widget| widget.identifier.as_str())
        .collect::<BTreeSet<_>>();
    let removed_inputs = node
        .inputs
        .iter()
        .enumerate()
        .filter(|(_, port)| !input_names.contains(port.name.as_str()))
        .map(|(index, _)| index)
        .collect::<BTreeSet<_>>();
    let removed_outputs = node
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, port)| !output_names.contains(port.name.as_str()))
        .map(|(index, _)| index)
        .collect::<BTreeSet<_>>();
    let input_mapping = node
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(old_index, port)| {
            inputs
                .iter()
                .position(|candidate| candidate.name == port.name)
                .map(|new_index| (old_index, new_index))
        })
        .collect::<BTreeMap<_, _>>();
    let output_mapping = node
        .outputs
        .iter()
        .enumerate()
        .filter_map(|(old_index, port)| {
            outputs
                .iter()
                .position(|candidate| candidate.name == port.name)
                .map(|new_index| (old_index, new_index))
        })
        .collect::<BTreeMap<_, _>>();
    for widget in widgets {
        widget.validate_schema()?;
    }
    let affected_links = graph
        .links
        .values()
        .filter(|link| {
            if link.target_node == *identifier && removed_inputs.contains(&link.target_slot) {
                return true;
            }
            if link.origin_node == *identifier && removed_outputs.contains(&link.origin_slot) {
                return true;
            }
            let output = if link.origin_node == *identifier {
                output_mapping
                    .get(&link.origin_slot)
                    .and_then(|slot| outputs.get(*slot))
            } else {
                graph
                    .nodes
                    .get(&link.origin_node)
                    .and_then(|node| node.outputs.get(link.origin_slot))
            };
            let input = if link.target_node == *identifier {
                input_mapping
                    .get(&link.target_slot)
                    .and_then(|slot| inputs.get(*slot))
            } else {
                graph
                    .nodes
                    .get(&link.target_node)
                    .and_then(|node| node.inputs.get(link.target_slot))
            };
            match (output, input) {
                (Some(output), Some(input)) => !input.port_type.accepts(&output.port_type),
                _ => true,
            }
        })
        .map(|link| link.identifier.clone())
        .collect::<BTreeSet<_>>();
    let removed_widgets = node
        .widgets
        .iter()
        .filter(|widget| !widget_names.contains(widget.identifier.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let incompatible_widgets = node
        .widgets
        .iter()
        .filter(|old| {
            widgets
                .iter()
                .find(|candidate| candidate.identifier == old.identifier)
                .is_some_and(|candidate| candidate.normalize(old.value.clone()).is_err())
        })
        .cloned()
        .collect::<Vec<_>>();
    let discarded_widgets = removed_widgets
        .iter()
        .chain(&incompatible_widgets)
        .cloned()
        .collect::<Vec<_>>();
    if (!affected_links.is_empty() || !discarded_widgets.is_empty()) && !confirm_discard {
        return Err(GraphError::ReconciliationRequiresConfirmation {
            links: affected_links.iter().cloned().collect(),
            widgets: discarded_widgets
                .iter()
                .map(|widget| widget.identifier.clone())
                .collect(),
        });
    }
    let mut quarantined_links = Vec::new();
    for link_identifier in affected_links {
        if let Some(link) = graph.links.remove(&link_identifier) {
            graph.selection.links.remove(&link_identifier);
            quarantined_links.push(
                serde_json::to_value(link)
                    .map_err(|error| GraphError::Serialization(error.to_string()))?,
            );
        }
    }
    for link in graph.links.values_mut() {
        if link.target_node == *identifier
            && let Some(new_index) = input_mapping.get(&link.target_slot)
        {
            link.target_slot = *new_index;
        }
        if link.origin_node == *identifier
            && let Some(new_index) = output_mapping.get(&link.origin_slot)
        {
            link.origin_slot = *new_index;
        }
    }
    let node = graph
        .nodes
        .get_mut(identifier)
        .ok_or_else(|| GraphError::UnknownNode(identifier.clone()))?;
    if !discarded_widgets.is_empty() {
        node.quarantine.insert(
            "unmapped_widgets".to_owned(),
            serde_json::to_value(discarded_widgets)
                .map_err(|error| GraphError::Serialization(error.to_string()))?,
        );
    }
    if !quarantined_links.is_empty() {
        node.quarantine
            .insert("unmapped_links".to_owned(), Value::Array(quarantined_links));
    }
    let old_widgets = node
        .widgets
        .iter()
        .map(|widget| (widget.identifier.clone(), widget.clone()))
        .collect::<BTreeMap<_, _>>();
    node.inputs = inputs.to_vec();
    node.outputs = outputs.to_vec();
    node.widgets = widgets
        .iter()
        .cloned()
        .map(|mut widget| {
            if let Some(old) = old_widgets.get(&widget.identifier) {
                if let Ok(normalized) = widget.normalize(old.value.clone()) {
                    widget.value = normalized.clone();
                    widget.prompt_value = normalized;
                    widget.validation = WidgetValidation::Valid;
                } else {
                    widget.validation = WidgetValidation::Invalid(
                        "previous value is incompatible with the refreshed definition".to_owned(),
                    );
                }
                widget.converted_to_input = old.converted_to_input;
            }
            widget
        })
        .collect();
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphBlueprintMetadata {
    pub filename: String,
    pub display_name: String,
    pub suggested_name: String,
    pub description: String,
    pub search_aliases: Vec<String>,
    pub inputs: Vec<SubgraphPort>,
    pub outputs: Vec<SubgraphPort>,
    #[serde(default)]
    pub exposed_widgets: Vec<SubgraphExposedWidget>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublishedSubgraphBlueprint {
    pub metadata: GraphBlueprintMetadata,
    pub workflow_bytes: Vec<u8>,
    pub clipboard: GraphClipboard,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphClipboard {
    pub schema_version: u16,
    pub source_document: Uuid,
    pub nodes: Vec<GraphNode>,
    pub links: Vec<GraphLink>,
    pub groups: Vec<GraphGroup>,
    pub reroutes: Vec<GraphReroute>,
    #[serde(default)]
    pub definitions: Vec<SubgraphDefinition>,
    #[serde(default)]
    pub unknown: BTreeMap<String, Value>,
}

impl GraphClipboard {
    pub fn from_published_subgraph_blueprint(
        filename: &str,
        workflow_bytes: &[u8],
    ) -> Result<(Self, GraphBlueprintMetadata), GraphError> {
        if workflow_bytes.len() > MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES {
            return Err(GraphError::BlueprintTooLarge(workflow_bytes.len()));
        }
        let (clipboard, metadata) = extract_published_subgraph_blueprint(filename, workflow_bytes)?;
        let clipboard = GraphClipboard::decode(&clipboard.encode()?)?;
        Ok((clipboard, metadata))
    }

    pub fn copy(document: &GraphDocument) -> Result<Self, GraphError> {
        let graph = document.active_graph()?;
        let mut selected_node_ids = graph.selection.nodes.clone();
        for group_identifier in &graph.selection.groups {
            let group = graph
                .groups
                .get(group_identifier)
                .ok_or_else(|| GraphError::UnknownGroup(group_identifier.clone()))?;
            selected_node_ids.extend(group.node_ids.iter().cloned());
        }
        validate_nodes_clonable(graph, &selected_node_ids)?;
        let nodes = graph
            .nodes
            .iter()
            .filter(|(identifier, _)| selected_node_ids.contains(*identifier))
            .map(|(_, node)| node.clone())
            .collect::<Vec<_>>();
        let selected_nodes = nodes
            .iter()
            .map(|node| node.identifier.clone())
            .collect::<BTreeSet<_>>();
        let links = graph
            .links
            .values()
            .filter(|link| {
                selected_nodes.contains(&link.origin_node)
                    && selected_nodes.contains(&link.target_node)
            })
            .cloned()
            .collect::<Vec<_>>();
        let groups = graph
            .selection
            .groups
            .iter()
            .filter_map(|identifier| graph.groups.get(identifier).cloned())
            .collect::<Vec<_>>();
        let mut reroute_ids = graph.selection.reroutes.clone();
        reroute_ids.extend(links.iter().filter_map(|link| link.parent_reroute.clone()));
        let mut pending_reroutes = reroute_ids.iter().cloned().collect::<Vec<_>>();
        while let Some(identifier) = pending_reroutes.pop() {
            let reroute = graph
                .reroutes
                .get(&identifier)
                .ok_or_else(|| GraphError::UnknownReroute(identifier.clone()))?;
            if let Some(parent) = &reroute.parent
                && reroute_ids.insert(parent.clone())
            {
                pending_reroutes.push(parent.clone());
            }
        }
        let reroutes = graph
            .reroutes
            .iter()
            .filter(|(identifier, _)| reroute_ids.contains(*identifier))
            .map(|(_, reroute)| reroute.clone())
            .collect::<Vec<_>>();
        let definition_ids = nodes
            .iter()
            .filter_map(|node| node.subgraph_definition.clone())
            .collect::<BTreeSet<_>>();
        let definitions = definition_ids
            .iter()
            .map(|identifier| {
                graph
                    .definitions
                    .get(identifier)
                    .cloned()
                    .ok_or_else(|| GraphError::UnknownSubgraph(identifier.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            schema_version: GRAPH_CLIPBOARD_SCHEMA_VERSION,
            source_document: document.document_identity,
            nodes,
            links,
            groups,
            reroutes,
            definitions,
            unknown: BTreeMap::new(),
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>, GraphError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| GraphError::Serialization(error.to_string()))?;
        if bytes.len() > MAX_GRAPH_CLIPBOARD_BYTES {
            return Err(GraphError::ClipboardTooLarge(bytes.len()));
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, GraphError> {
        if bytes.len() > MAX_GRAPH_CLIPBOARD_BYTES {
            return Err(GraphError::ClipboardTooLarge(bytes.len()));
        }
        validate_json_depth(bytes, MAX_GRAPH_DEPTH)?;
        let mut clipboard: Self = serde_json::from_slice(bytes)
            .map_err(|error| GraphError::InvalidClipboard(error.to_string()))?;
        if clipboard.schema_version != GRAPH_CLIPBOARD_SCHEMA_VERSION {
            return Err(GraphError::UnsupportedClipboardSchema(
                clipboard.schema_version,
            ));
        }
        for definition in &mut clipboard.definitions {
            normalize_json_identifier_map_keys(&mut definition.graph)
                .map_err(|error| GraphError::InvalidClipboard(error.to_string()))?;
        }
        let mut node_ids = BTreeSet::new();
        for node in &clipboard.nodes {
            node.position.validate()?;
            node.size.validate()?;
            if !node.is_clonable() {
                return Err(GraphError::InvalidClipboard(format!(
                    "node {:?} does not permit cloning",
                    node.identifier
                )));
            }
            if !node_ids.insert(node.identifier.clone()) {
                return Err(GraphError::DuplicateEntity(node.identifier.clone()));
            }
        }
        let mut link_ids = BTreeSet::new();
        for link in &clipboard.links {
            if !link_ids.insert(link.identifier.clone()) {
                return Err(GraphError::DuplicateEntity(link.identifier.clone()));
            }
            if !node_ids.contains(&link.origin_node) || !node_ids.contains(&link.target_node) {
                return Err(GraphError::InvalidClipboard(
                    "link refers outside the copied node set".to_owned(),
                ));
            }
        }
        let mut group_ids = BTreeSet::new();
        for group in &clipboard.groups {
            group.bounds.validate()?;
            if !group_ids.insert(group.identifier.clone()) {
                return Err(GraphError::DuplicateEntity(group.identifier.clone()));
            }
            if !group.node_ids.is_subset(&node_ids) {
                return Err(GraphError::InvalidClipboard(
                    "group refers outside the copied node set".to_owned(),
                ));
            }
        }
        let reroute_ids = clipboard
            .reroutes
            .iter()
            .map(|reroute| reroute.identifier.clone())
            .collect::<BTreeSet<_>>();
        if reroute_ids.len() != clipboard.reroutes.len() {
            return Err(GraphError::InvalidClipboard(
                "reroute identifiers are duplicated".to_owned(),
            ));
        }
        for reroute in &clipboard.reroutes {
            reroute.position.validate()?;
            if reroute
                .parent
                .as_ref()
                .is_some_and(|parent| !reroute_ids.contains(parent))
            {
                return Err(GraphError::InvalidClipboard(
                    "reroute parent refers outside the copied set".to_owned(),
                ));
            }
        }
        if clipboard.links.iter().any(|link| {
            link.parent_reroute
                .as_ref()
                .is_some_and(|parent| !reroute_ids.contains(parent))
        }) {
            return Err(GraphError::InvalidClipboard(
                "link reroute refers outside the copied set".to_owned(),
            ));
        }
        let definition_ids = clipboard
            .definitions
            .iter()
            .map(|definition| definition.identifier.clone())
            .collect::<BTreeSet<_>>();
        if definition_ids.len() != clipboard.definitions.len() {
            return Err(GraphError::InvalidClipboard(
                "subgraph definition identifiers are duplicated".to_owned(),
            ));
        }
        for definition in &clipboard.definitions {
            validate_level(&definition.graph, 1)?;
        }
        for node in &clipboard.nodes {
            if let Some(definition) = &node.subgraph_definition
                && !definition_ids.contains(definition)
            {
                return Err(GraphError::InvalidClipboard(
                    "subgraph node refers outside the copied definition set".to_owned(),
                ));
            }
        }
        let level = GraphLevel {
            nodes: clipboard
                .nodes
                .iter()
                .cloned()
                .map(|node| (node.identifier.clone(), node))
                .collect(),
            links: clipboard
                .links
                .iter()
                .cloned()
                .map(|link| (link.identifier.clone(), link))
                .collect(),
            groups: clipboard
                .groups
                .iter()
                .cloned()
                .map(|group| (group.identifier.clone(), group))
                .collect(),
            reroutes: clipboard
                .reroutes
                .iter()
                .cloned()
                .map(|reroute| (reroute.identifier.clone(), reroute))
                .collect(),
            definitions: clipboard
                .definitions
                .iter()
                .cloned()
                .map(|definition| (definition.identifier.clone(), definition))
                .collect(),
            ..GraphLevel::default()
        };
        validate_level(&level, 0)
            .map_err(|error| GraphError::InvalidClipboard(error.to_string()))?;
        Ok(clipboard)
    }

    pub fn paste_command(
        &self,
        document: &GraphDocument,
        offset: GraphPoint,
    ) -> Result<GraphCommand, GraphError> {
        offset.validate()?;
        let mut allocation_document = document.clone();
        let mut node_mapping = BTreeMap::new();
        for node in &self.nodes {
            node_mapping.insert(
                node.identifier.clone(),
                allocation_document.allocate_identifier(),
            );
        }
        let mut group_mapping = BTreeMap::new();
        for group in &self.groups {
            group_mapping.insert(
                group.identifier.clone(),
                allocation_document.allocate_identifier(),
            );
        }
        let mut reroute_mapping = BTreeMap::new();
        for reroute in &self.reroutes {
            reroute_mapping.insert(
                reroute.identifier.clone(),
                allocation_document.allocate_identifier(),
            );
        }
        let mut definition_mapping = BTreeMap::new();
        for definition in &self.definitions {
            definition_mapping.insert(
                definition.identifier.clone(),
                allocation_document.allocate_subgraph_identifier(),
            );
        }
        let mut commands = Vec::new();
        for definition in &self.definitions {
            let mut definition = definition.clone();
            definition.identifier = definition_mapping
                .get(&definition.identifier)
                .cloned()
                .ok_or_else(|| {
                    GraphError::InvalidClipboard("definition mapping missing".to_owned())
                })?;
            for (index, port) in definition.inputs.iter_mut().enumerate() {
                port.identifier =
                    subgraph_port_identifier(&definition.identifier, "input", index, &port.name);
            }
            for (index, port) in definition.outputs.iter_mut().enumerate() {
                port.identifier =
                    subgraph_port_identifier(&definition.identifier, "output", index, &port.name);
            }
            commands.push(GraphCommand::AddSubgraphDefinition { definition });
        }
        for node in &self.nodes {
            let mut node = node.clone();
            node.identifier = node_mapping
                .get(&node.identifier)
                .cloned()
                .ok_or_else(|| GraphError::InvalidClipboard("node mapping missing".to_owned()))?;
            node.subgraph_definition = node
                .subgraph_definition
                .as_ref()
                .map(|identifier| {
                    definition_mapping.get(identifier).cloned().ok_or_else(|| {
                        GraphError::InvalidClipboard("definition mapping missing".to_owned())
                    })
                })
                .transpose()?;
            node.position = node.position.translated(offset);
            commands.push(GraphCommand::AddNode {
                node,
                source: NodeCreationSource::Paste,
            });
        }
        for reroute in &self.reroutes {
            let mut reroute = reroute.clone();
            reroute.identifier = reroute_mapping
                .get(&reroute.identifier)
                .cloned()
                .ok_or_else(|| {
                    GraphError::InvalidClipboard("reroute mapping missing".to_owned())
                })?;
            reroute.position = reroute.position.translated(offset);
            reroute.parent = None;
            commands.push(GraphCommand::AddReroute { reroute });
        }
        for reroute in &self.reroutes {
            if let Some(parent) = &reroute.parent {
                commands.push(GraphCommand::ReparentReroute {
                    identifier: reroute_mapping
                        .get(&reroute.identifier)
                        .cloned()
                        .ok_or_else(|| {
                            GraphError::InvalidClipboard("reroute mapping missing".to_owned())
                        })?,
                    parent: Some(reroute_mapping.get(parent).cloned().ok_or_else(|| {
                        GraphError::InvalidClipboard("reroute parent mapping missing".to_owned())
                    })?),
                });
            }
        }
        for link in &self.links {
            let mut link = link.clone();
            link.identifier = allocation_document.allocate_identifier();
            link.origin_node = node_mapping
                .get(&link.origin_node)
                .cloned()
                .ok_or_else(|| GraphError::InvalidClipboard("origin mapping missing".to_owned()))?;
            link.target_node = node_mapping
                .get(&link.target_node)
                .cloned()
                .ok_or_else(|| GraphError::InvalidClipboard("target mapping missing".to_owned()))?;
            link.parent_reroute = link
                .parent_reroute
                .as_ref()
                .map(|identifier| {
                    reroute_mapping.get(identifier).cloned().ok_or_else(|| {
                        GraphError::InvalidClipboard("link reroute mapping missing".to_owned())
                    })
                })
                .transpose()?;
            commands.push(GraphCommand::Connect {
                link,
                replace_existing: false,
            });
        }
        for group in &self.groups {
            let mut group = group.clone();
            group.identifier = group_mapping
                .get(&group.identifier)
                .cloned()
                .ok_or_else(|| GraphError::InvalidClipboard("group mapping missing".to_owned()))?;
            group.bounds.origin = group.bounds.origin.translated(offset);
            group.node_ids = group
                .node_ids
                .iter()
                .map(|identifier| {
                    node_mapping.get(identifier).cloned().ok_or_else(|| {
                        GraphError::InvalidClipboard("group node mapping missing".to_owned())
                    })
                })
                .collect::<Result<_, _>>()?;
            commands.push(GraphCommand::CreateGroup { group });
        }
        commands.push(GraphCommand::SetSelection {
            selection: GraphSelection {
                nodes: node_mapping.values().cloned().collect(),
                groups: group_mapping.values().cloned().collect(),
                reroutes: reroute_mapping.values().cloned().collect(),
                ..GraphSelection::default()
            },
            mode: SelectionMode::Replace,
        });
        Ok(GraphCommand::Batch { commands })
    }
}

impl PublishedSubgraphBlueprint {
    pub fn decode(filename: &str, workflow_bytes: &[u8]) -> Result<Self, GraphError> {
        let (clipboard, metadata) =
            GraphClipboard::from_published_subgraph_blueprint(filename, workflow_bytes)?;
        Ok(Self {
            metadata,
            workflow_bytes: workflow_bytes.to_vec(),
            clipboard,
        })
    }

    pub fn instantiate_command(
        &self,
        document: &GraphDocument,
        offset: GraphPoint,
    ) -> Result<GraphCommand, GraphError> {
        self.clipboard.paste_command(document, offset)
    }
}

fn export_selected_subgraph_blueprint(
    document: &GraphDocument,
    display_name: &str,
) -> Result<PublishedSubgraphBlueprint, GraphError> {
    let filename = blueprint_filename(display_name)?;
    let graph = document.active_graph()?;
    let selected_count = graph
        .selection
        .nodes
        .len()
        .saturating_add(graph.selection.links.len())
        .saturating_add(graph.selection.groups.len())
        .saturating_add(graph.selection.reroutes.len());
    if selected_count != 1 {
        return Err(GraphError::InvalidBlueprintSelection(selected_count));
    }
    let instance_identifier = graph
        .selection
        .nodes
        .iter()
        .next()
        .ok_or(GraphError::InvalidBlueprintSelection(selected_count))?;
    let mut instance = graph
        .nodes
        .get(instance_identifier)
        .cloned()
        .ok_or_else(|| GraphError::UnknownNode(instance_identifier.clone()))?;
    let suggested_name = instance.title.clone();
    let definition_identifier = instance
        .subgraph_definition
        .clone()
        .ok_or_else(|| GraphError::NotSubgraphInstance(instance_identifier.clone()))?;
    let mut definition = graph
        .definitions
        .get(&definition_identifier)
        .cloned()
        .ok_or_else(|| GraphError::UnknownSubgraph(definition_identifier.clone()))?;

    normalize_definition_blueprint_metadata(&mut definition)?;
    validate_blueprint_instance_ports(&instance, &definition)?;
    instance.title = display_name.to_owned();
    definition.name = display_name.to_owned();
    definition.published = true;

    let identity_source = serde_json::to_vec(&(
        display_name,
        &instance,
        &definition,
        WorkflowFormat::Schema04,
    ))
    .map_err(|error| GraphError::Serialization(error.to_string()))?;
    let document_identity = Uuid::new_v5(&Uuid::NAMESPACE_OID, &identity_source);
    let mut root = GraphLevel::default();
    root.nodes
        .insert(instance.identifier.clone(), instance.clone());
    root.definitions
        .insert(definition.identifier.clone(), definition.clone());
    root.selection.nodes.insert(instance.identifier);
    let mut extra = Map::new();
    if !definition.description.is_empty() {
        extra.insert(
            BLUEPRINT_DESCRIPTION_FIELD.to_owned(),
            Value::String(definition.description.clone()),
        );
    }
    if !definition.search_aliases.is_empty() {
        extra.insert(
            BLUEPRINT_SEARCH_ALIASES_FIELD.to_owned(),
            serde_json::to_value(&definition.search_aliases)
                .map_err(|error| GraphError::Serialization(error.to_string()))?,
        );
    }
    if !extra.is_empty() {
        root.source_fields
            .insert("extra".to_owned(), Value::Object(extra));
    }
    let blueprint_document = GraphDocument {
        schema_version: GRAPH_DOCUMENT_SCHEMA_VERSION,
        document_identity,
        profile_identity: None,
        workflow_format: Some(WorkflowFormat::Schema04),
        root,
        navigation: Vec::new(),
        next_identifier: document.next_identifier,
        diagnostics: Vec::new(),
    };
    let workflow_bytes = blueprint_document.to_workflow_bytes()?;
    // The asset library owns the published-workflow byte limit and applies it before commit.
    // External blueprint decoding remains bounded by `GraphClipboard::from_published_subgraph_blueprint`.
    let (clipboard, mut metadata) =
        extract_published_subgraph_blueprint(&filename, &workflow_bytes)?;
    metadata.suggested_name = suggested_name;
    Ok(PublishedSubgraphBlueprint {
        metadata,
        workflow_bytes,
        clipboard,
    })
}

fn extract_published_subgraph_blueprint(
    filename: &str,
    workflow_bytes: &[u8],
) -> Result<(GraphClipboard, GraphBlueprintMetadata), GraphError> {
    let display_name = blueprint_display_name(filename)?;
    let document = GraphDocument::from_workflow_bytes(workflow_bytes)?;
    let graph = &document.root;
    if graph.nodes.len() != 1
        || !graph.links.is_empty()
        || !graph.groups.is_empty()
        || !graph.reroutes.is_empty()
        || graph.definitions.len() != 1
    {
        return Err(GraphError::InvalidBlueprint(
            "root graph must contain one subgraph instance, its definition, and no links or auxiliary items"
                .to_owned(),
        ));
    }
    let instance = graph.nodes.values().next().ok_or_else(|| {
        GraphError::InvalidBlueprint("root subgraph instance is missing".to_owned())
    })?;
    let definition_identifier = instance.subgraph_definition.as_ref().ok_or_else(|| {
        GraphError::InvalidBlueprint("root node is not a subgraph instance".to_owned())
    })?;
    let definition = graph
        .definitions
        .get(definition_identifier)
        .ok_or_else(|| GraphError::UnknownSubgraph(definition_identifier.clone()))?;
    if instance.title != display_name || definition.name != display_name {
        return Err(GraphError::BlueprintNameMismatch {
            filename: filename.to_owned(),
            instance: instance.title.clone(),
            definition: definition.name.clone(),
        });
    }
    validate_blueprint_instance_ports(instance, definition)?;
    let (description, search_aliases) = blueprint_metadata(graph, definition)?;
    let clipboard = GraphClipboard {
        schema_version: GRAPH_CLIPBOARD_SCHEMA_VERSION,
        source_document: document.document_identity,
        nodes: vec![instance.clone()],
        links: Vec::new(),
        groups: Vec::new(),
        reroutes: Vec::new(),
        definitions: vec![definition.clone()],
        unknown: BTreeMap::new(),
    };
    let metadata = GraphBlueprintMetadata {
        filename: filename.to_owned(),
        suggested_name: display_name.clone(),
        display_name,
        description,
        search_aliases,
        inputs: definition.inputs.clone(),
        outputs: definition.outputs.clone(),
        exposed_widgets: definition.exposed_widgets.clone(),
    };
    Ok((clipboard, metadata))
}

fn blueprint_metadata(
    graph: &GraphLevel,
    definition: &SubgraphDefinition,
) -> Result<(String, Vec<String>), GraphError> {
    let extra = graph.source_fields.get("extra").and_then(Value::as_object);
    let description = match extra.and_then(|extra| extra.get(BLUEPRINT_DESCRIPTION_FIELD)) {
        Some(Value::String(description)) => description.clone(),
        Some(_) => {
            return Err(GraphError::InvalidBlueprint(format!(
                "{BLUEPRINT_DESCRIPTION_FIELD} must be a string"
            )));
        }
        None => definition.description.clone(),
    };
    let search_aliases = match extra.and_then(|extra| extra.get(BLUEPRINT_SEARCH_ALIASES_FIELD)) {
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                    GraphError::InvalidBlueprint(format!(
                        "{BLUEPRINT_SEARCH_ALIASES_FIELD} must contain only strings"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => {
            return Err(GraphError::InvalidBlueprint(format!(
                "{BLUEPRINT_SEARCH_ALIASES_FIELD} must be an array"
            )));
        }
        None => definition.search_aliases.clone(),
    };
    Ok((description, search_aliases))
}

fn normalize_definition_blueprint_metadata(
    definition: &mut SubgraphDefinition,
) -> Result<(), GraphError> {
    let remove_extra = if let Some(Value::Object(extra)) = definition.unknown.get_mut("extra") {
        if let Some(description) = extra.get(BLUEPRINT_DESCRIPTION_FIELD) {
            definition.description = description
                .as_str()
                .ok_or_else(|| {
                    GraphError::InvalidBlueprint(format!(
                        "{BLUEPRINT_DESCRIPTION_FIELD} must be a string"
                    ))
                })?
                .to_owned();
        }
        if let Some(search_aliases) = extra.get(BLUEPRINT_SEARCH_ALIASES_FIELD) {
            definition.search_aliases = search_aliases
                .as_array()
                .ok_or_else(|| {
                    GraphError::InvalidBlueprint(format!(
                        "{BLUEPRINT_SEARCH_ALIASES_FIELD} must be an array"
                    ))
                })?
                .iter()
                .map(|alias| {
                    alias.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                        GraphError::InvalidBlueprint(format!(
                            "{BLUEPRINT_SEARCH_ALIASES_FIELD} must contain only strings"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        extra.remove(BLUEPRINT_DESCRIPTION_FIELD);
        extra.remove(BLUEPRINT_SEARCH_ALIASES_FIELD);
        extra.is_empty()
    } else {
        false
    };
    if remove_extra {
        definition.unknown.remove("extra");
    }
    Ok(())
}

fn validate_blueprint_instance_ports(
    instance: &GraphNode,
    definition: &SubgraphDefinition,
) -> Result<(), GraphError> {
    let inputs_match = instance.inputs.len() == definition.inputs.len()
        && instance
            .inputs
            .iter()
            .zip(&definition.inputs)
            .all(|(instance, definition)| {
                instance.name == definition.name && instance.port_type == definition.port_type
            });
    let outputs_match = instance.outputs.len() == definition.outputs.len()
        && instance
            .outputs
            .iter()
            .zip(&definition.outputs)
            .all(|(instance, definition)| {
                instance.name == definition.name && instance.port_type == definition.port_type
            });
    if !inputs_match || !outputs_match {
        return Err(GraphError::InvalidBlueprint(
            "root instance ports do not match its subgraph definition".to_owned(),
        ));
    }
    Ok(())
}

fn blueprint_filename(display_name: &str) -> Result<String, GraphError> {
    if validate_graph_label(display_name).is_err()
        || display_name.trim() != display_name
        || matches!(display_name, "." | "..")
        || display_name.contains(['/', '\\'])
    {
        return Err(GraphError::InvalidBlueprintName(display_name.to_owned()));
    }
    Ok(format!("{display_name}.json"))
}

fn blueprint_display_name(filename: &str) -> Result<String, GraphError> {
    if filename.trim() != filename
        || filename.contains(['/', '\\'])
        || filename.matches('.').count() == 0
    {
        return Err(GraphError::InvalidBlueprintName(filename.to_owned()));
    }
    let display_name = filename
        .strip_suffix(".json")
        .ok_or_else(|| GraphError::InvalidBlueprintName(filename.to_owned()))?;
    if blueprint_filename(display_name)? != filename {
        return Err(GraphError::InvalidBlueprintName(filename.to_owned()));
    }
    Ok(display_name.to_owned())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CatalogGraphAction {
    CopySelected,
    DeleteSelectedItems,
    FitView,
    Lock,
    MoveSelectedDown,
    MoveSelectedLeft,
    MoveSelectedRight,
    MoveSelectedUp,
    PasteFromClipboard,
    PasteFromClipboardWithConnect,
    ResetView,
    Resize,
    SelectAll,
    ToggleLinkVisibility,
    ToggleLock,
    ToggleMinimap,
    ToggleSelectedBypass,
    ToggleSelectedCollapse,
    ToggleSelectedMute,
    ToggleSelectedPin,
    ToggleSelectedItemsPin,
    Unlock,
    ZoomIn,
    ZoomOut,
    ConvertToSubgraph,
    EditSubgraphWidgets,
    ExitSubgraph,
    FitGroupToContents,
    GroupSelectedNodes,
    ToggleWidgetPromotion,
    UnpackSubgraph,
    PublishSubgraph,
    RefreshNodeDefinitions,
    SetSubgraphDescription,
    SetSubgraphSearchAliases,
    ToggleCanvasInfo,
    ToggleVueNodes,
}

impl CatalogGraphAction {
    pub const ALL: [Self; 37] = [
        Self::CopySelected,
        Self::DeleteSelectedItems,
        Self::FitView,
        Self::Lock,
        Self::MoveSelectedDown,
        Self::MoveSelectedLeft,
        Self::MoveSelectedRight,
        Self::MoveSelectedUp,
        Self::PasteFromClipboard,
        Self::PasteFromClipboardWithConnect,
        Self::ResetView,
        Self::Resize,
        Self::SelectAll,
        Self::ToggleLinkVisibility,
        Self::ToggleLock,
        Self::ToggleMinimap,
        Self::ToggleSelectedBypass,
        Self::ToggleSelectedCollapse,
        Self::ToggleSelectedMute,
        Self::ToggleSelectedPin,
        Self::ToggleSelectedItemsPin,
        Self::Unlock,
        Self::ZoomIn,
        Self::ZoomOut,
        Self::ConvertToSubgraph,
        Self::EditSubgraphWidgets,
        Self::ExitSubgraph,
        Self::FitGroupToContents,
        Self::GroupSelectedNodes,
        Self::ToggleWidgetPromotion,
        Self::UnpackSubgraph,
        Self::PublishSubgraph,
        Self::RefreshNodeDefinitions,
        Self::SetSubgraphDescription,
        Self::SetSubgraphSearchAliases,
        Self::ToggleCanvasInfo,
        Self::ToggleVueNodes,
    ];

    pub fn command_id(self) -> &'static str {
        match self {
            Self::CopySelected => "Sim.Canvas.CopySelected",
            Self::DeleteSelectedItems => "Sim.Canvas.DeleteSelectedItems",
            Self::FitView => "Sim.Canvas.FitView",
            Self::Lock => "Sim.Canvas.Lock",
            Self::MoveSelectedDown => "Sim.Canvas.MoveSelectedNodes.Down",
            Self::MoveSelectedLeft => "Sim.Canvas.MoveSelectedNodes.Left",
            Self::MoveSelectedRight => "Sim.Canvas.MoveSelectedNodes.Right",
            Self::MoveSelectedUp => "Sim.Canvas.MoveSelectedNodes.Up",
            Self::PasteFromClipboard => "Sim.Canvas.PasteFromClipboard",
            Self::PasteFromClipboardWithConnect => "Sim.Canvas.PasteFromClipboardWithConnect",
            Self::ResetView => "Sim.Canvas.ResetView",
            Self::Resize => "Sim.Canvas.Resize",
            Self::SelectAll => "Sim.Canvas.SelectAll",
            Self::ToggleLinkVisibility => "Sim.Canvas.ToggleLinkVisibility",
            Self::ToggleLock => "Sim.Canvas.ToggleLock",
            Self::ToggleMinimap => "Sim.Canvas.ToggleMinimap",
            Self::ToggleSelectedBypass => "Sim.Canvas.ToggleSelectedNodes.Bypass",
            Self::ToggleSelectedCollapse => "Sim.Canvas.ToggleSelectedNodes.Collapse",
            Self::ToggleSelectedMute => "Sim.Canvas.ToggleSelectedNodes.Mute",
            Self::ToggleSelectedPin => "Sim.Canvas.ToggleSelectedNodes.Pin",
            Self::ToggleSelectedItemsPin => "Sim.Canvas.ToggleSelected.Pin",
            Self::Unlock => "Sim.Canvas.Unlock",
            Self::ZoomIn => "Sim.Canvas.ZoomIn",
            Self::ZoomOut => "Sim.Canvas.ZoomOut",
            Self::ConvertToSubgraph => "Sim.Graph.ConvertToSubgraph",
            Self::EditSubgraphWidgets => "Sim.Graph.EditSubgraphWidgets",
            Self::ExitSubgraph => "Sim.Graph.ExitSubgraph",
            Self::FitGroupToContents => "Sim.Graph.FitGroupToContents",
            Self::GroupSelectedNodes => "Sim.Graph.GroupSelectedNodes",
            Self::ToggleWidgetPromotion => "Sim.Graph.ToggleWidgetPromotion",
            Self::UnpackSubgraph => "Sim.Graph.UnpackSubgraph",
            Self::PublishSubgraph => "Sim.PublishSubgraph",
            Self::RefreshNodeDefinitions => "Sim.RefreshNodeDefinitions",
            Self::SetSubgraphDescription => "Sim.Subgraph.SetDescription",
            Self::SetSubgraphSearchAliases => "Sim.Subgraph.SetSearchAliases",
            Self::ToggleCanvasInfo => "Sim.ToggleCanvasInfo",
            Self::ToggleVueNodes => "Experimental.ToggleVueNodes",
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum GraphError {
    #[error(transparent)]
    Workflow(#[from] WorkflowFormatError),
    #[error("graph document schema {0} is unsupported")]
    UnsupportedDocumentSchema(u16),
    #[error("graph clipboard schema {0} is unsupported")]
    UnsupportedClipboardSchema(u16),
    #[error("graph entity {0:?} already exists")]
    DuplicateEntity(GraphIdentifier),
    #[error("graph node {0:?} does not exist")]
    UnknownNode(GraphIdentifier),
    #[error("graph link {0:?} does not exist")]
    UnknownLink(GraphIdentifier),
    #[error("graph group {0:?} does not exist")]
    UnknownGroup(GraphIdentifier),
    #[error("graph group {0:?} has stale derived membership")]
    InvalidGroupMembership(GraphIdentifier),
    #[error("graph reroute {0:?} does not exist")]
    UnknownReroute(GraphIdentifier),
    #[error("reroute {0:?} resolves to conflicting port types")]
    ConflictingReroutePortTypes(GraphIdentifier),
    #[error("floating reroute link is invalid: {0}")]
    InvalidFloatingLink(String),
    #[error("graph subgraph {0:?} does not exist")]
    UnknownSubgraph(GraphIdentifier),
    #[error("subgraph {definition:?} is still referenced by instances {instances:?}")]
    SubgraphDefinitionInUse {
        definition: GraphIdentifier,
        instances: Vec<GraphIdentifier>,
    },
    #[error("graph node {node:?} has no port at slot {slot}")]
    UnknownPort { node: GraphIdentifier, slot: usize },
    #[error("subgraph {definition:?} has no {direction:?} port at slot {slot}")]
    UnknownSubgraphPort {
        definition: GraphIdentifier,
        direction: GraphSlotDirection,
        slot: usize,
    },
    #[error("subgraph {definition:?} {direction:?} slot {slot} has no links")]
    SubgraphSlotHasNoLinks {
        definition: GraphIdentifier,
        direction: GraphSlotDirection,
        slot: usize,
    },
    #[error("graph labels must be nonempty, bounded, and contain no control characters")]
    InvalidGraphLabel,
    #[error("group font size {0} is outside the supported range")]
    InvalidGroupFontSize(f32),
    #[error("group padding {0} must be finite and nonnegative")]
    InvalidGroupPadding(f32),
    #[error("node properties are {0} bytes, exceeding their limit")]
    InvalidNodeProperties(usize),
    #[error("graph node {0:?} has no advanced widgets")]
    NodeHasNoAdvancedWidgets(GraphIdentifier),
    #[error("graph widget `{0}` does not exist")]
    UnknownWidget(String),
    #[error("graph widget `{0}` received an invalid value")]
    InvalidWidgetValue(String),
    #[error("graph widget `{widget}` has an invalid schema: {reason}")]
    InvalidWidgetSchema { widget: String, reason: String },
    #[error("output type `{output}` is incompatible with input type `{input}`")]
    IncompatiblePorts { output: String, input: String },
    #[error("input {node:?}:{slot} already has a connection")]
    InputOccupied { node: GraphIdentifier, slot: usize },
    #[error("graph geometry must be finite")]
    NonFiniteGeometry,
    #[error("graph size must be finite and positive")]
    InvalidSize,
    #[error("viewport scale {0} is outside the supported range")]
    InvalidViewportScale(f32),
    #[error("graph selection is empty")]
    EmptySelection,
    #[error("graph entity {0:?} is pinned")]
    PinnedEntity(GraphIdentifier),
    #[error("graph node {node:?} does not permit {operation}")]
    NodeOperationRestricted {
        node: GraphIdentifier,
        operation: &'static str,
    },
    #[error("graph node {node:?} has a non-boolean `{field}` restriction")]
    InvalidNodeOperationFlag {
        node: GraphIdentifier,
        field: &'static str,
    },
    #[error("graph canvas is locked")]
    CanvasLocked,
    #[error("reroute {0:?} would form a parent cycle")]
    RerouteCycle(GraphIdentifier),
    #[error("dynamic graph input {node:?}:{slot} must be the final input")]
    DynamicInputMustBeLast { node: GraphIdentifier, slot: usize },
    #[error("dynamic subgraph input {node:?}:`{port}` cannot expose a widget")]
    UnsupportedDynamicSubgraphInput { node: GraphIdentifier, port: String },
    #[error("node {0:?} is not a subgraph instance")]
    NotSubgraphInstance(GraphIdentifier),
    #[error("subgraph instance {instance:?} has no exposed port at slot {slot}")]
    InvalidSubgraphBoundary {
        instance: GraphIdentifier,
        slot: usize,
    },
    #[error("graph navigation is already at the root")]
    AtRootGraph,
    #[error("the current selection is not eligible for subgraph conversion")]
    InvalidSubgraphConversion,
    #[error(
        "node definition reconciliation requires confirmation for links {links:?} and widgets {widgets:?}"
    )]
    ReconciliationRequiresConfirmation {
        links: Vec<GraphIdentifier>,
        widgets: Vec<String>,
    },
    #[error("graph snapshot is {0} bytes, exceeding its limit")]
    SnapshotTooLarge(usize),
    #[error("graph clipboard is {0} bytes, exceeding its limit")]
    ClipboardTooLarge(usize),
    #[error("graph clipboard is invalid: {0}")]
    InvalidClipboard(String),
    #[error("published subgraph blueprint is {0} bytes, exceeding its limit")]
    BlueprintTooLarge(usize),
    #[error("a published subgraph blueprint requires exactly one selected item, found {0}")]
    InvalidBlueprintSelection(usize),
    #[error("published subgraph blueprint name `{0}` is invalid")]
    InvalidBlueprintName(String),
    #[error(
        "published subgraph blueprint `{filename}` does not match instance `{instance}` and definition `{definition}`"
    )]
    BlueprintNameMismatch {
        filename: String,
        instance: String,
        definition: String,
    },
    #[error("published subgraph blueprint is invalid: {0}")]
    InvalidBlueprint(String),
    #[error("graph history is invalid or exceeds its bound")]
    InvalidHistory,
    #[error("graph nesting exceeds its bound")]
    TooDeep,
    #[error("workflow validation failed at {path}: {reason}")]
    InvalidWorkflow { path: String, reason: String },
    #[error("{0} import is not an editable workflow graph")]
    UnsupportedImport(&'static str),
    #[error("graph serialization failed: {0}")]
    Serialization(String),
}

fn parse_level(value: &Value, depth: usize) -> Result<GraphLevel, GraphError> {
    if depth > MAX_GRAPH_DEPTH {
        return Err(GraphError::TooDeep);
    }
    let object = value.as_object().ok_or_else(|| {
        GraphError::Serialization("workflow graph root must be an object".to_owned())
    })?;
    let mut level = GraphLevel {
        source_fields: object.clone(),
        ..GraphLevel::default()
    };
    level.viewport = parse_viewport(object)?;
    let nodes = object
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| GraphError::InvalidWorkflow {
            path: "$.nodes".to_owned(),
            reason: "workflow nodes must be an array".to_owned(),
        })?;
    for node in nodes {
        let node = parse_node(node)?;
        if level
            .nodes
            .insert(node.identifier.clone(), node.clone())
            .is_some()
        {
            return Err(GraphError::DuplicateEntity(node.identifier));
        }
    }
    if let Some(links) = optional_array_field(object, "links")? {
        for link in links {
            let link = parse_link(link)?;
            if level
                .links
                .insert(link.identifier.clone(), link.clone())
                .is_some()
            {
                return Err(GraphError::DuplicateEntity(link.identifier));
            }
        }
    }
    if let Some(extensions) = object
        .get("extra")
        .and_then(Value::as_object)
        .and_then(|extra| extra.get("linkExtensions"))
    {
        let extensions = extensions
            .as_array()
            .ok_or_else(|| GraphError::InvalidWorkflow {
                path: "$.extra.linkExtensions".to_owned(),
                reason: "link extensions must be an array".to_owned(),
            })?;
        for (index, extension) in extensions.iter().enumerate() {
            let extension = extension
                .as_object()
                .ok_or_else(|| GraphError::InvalidWorkflow {
                    path: format!("$.extra.linkExtensions[{index}]"),
                    reason: "link extension must be an object".to_owned(),
                })?;
            let identifier = extension
                .get("id")
                .and_then(GraphIdentifier::from_value)
                .ok_or_else(|| GraphError::InvalidWorkflow {
                    path: format!("$.extra.linkExtensions[{index}].id"),
                    reason: "link extension id is invalid".to_owned(),
                })?;
            let parent = extension
                .get("parentId")
                .and_then(GraphIdentifier::from_value)
                .ok_or_else(|| GraphError::InvalidWorkflow {
                    path: format!("$.extra.linkExtensions[{index}].parentId"),
                    reason: "link extension parent is invalid".to_owned(),
                })?;
            level
                .links
                .get_mut(&identifier)
                .ok_or_else(|| GraphError::UnknownLink(identifier.clone()))?
                .parent_reroute = Some(parent);
        }
    }
    if let Some(groups) = optional_array_field(object, "groups")? {
        for (index, group) in groups.iter().enumerate() {
            let group = parse_group(group, index)?;
            if level
                .groups
                .insert(group.identifier.clone(), group.clone())
                .is_some()
            {
                return Err(GraphError::DuplicateEntity(group.identifier));
            }
        }
    }
    let reroutes = if object.contains_key("reroutes") {
        optional_array_field(object, "reroutes")?
    } else {
        object
            .get("extra")
            .and_then(Value::as_object)
            .and_then(|extra| extra.get("reroutes"))
            .map(|reroutes| {
                reroutes
                    .as_array()
                    .ok_or_else(|| GraphError::InvalidWorkflow {
                        path: "$.extra.reroutes".to_owned(),
                        reason: "reroutes must be an array".to_owned(),
                    })
            })
            .transpose()?
    };
    if let Some(reroutes) = reroutes {
        for reroute in reroutes {
            let reroute = parse_reroute(reroute)?;
            if level
                .reroutes
                .insert(reroute.identifier.clone(), reroute.clone())
                .is_some()
            {
                return Err(GraphError::DuplicateEntity(reroute.identifier));
            }
        }
    }
    if let Some(definitions) = object.get("definitions") {
        let definitions = match definitions {
            Value::Array(values) => values.iter().map(|value| (None, value)).collect::<Vec<_>>(),
            Value::Object(values) if values.contains_key("subgraphs") => values
                .get("subgraphs")
                .and_then(Value::as_array)
                .ok_or_else(|| GraphError::InvalidWorkflow {
                    path: "$.definitions.subgraphs".to_owned(),
                    reason: "subgraph definitions must be an array".to_owned(),
                })?
                .iter()
                .map(|value| (None, value))
                .collect::<Vec<_>>(),
            Value::Object(values) => values
                .iter()
                .map(|(identifier, value)| (Some(identifier.as_str()), value))
                .collect::<Vec<_>>(),
            _ => {
                return Err(GraphError::InvalidWorkflow {
                    path: "$.definitions".to_owned(),
                    reason: "definitions must be an array or object".to_owned(),
                });
            }
        };
        for (fallback_identifier, value) in definitions {
            if let Some(definition) = parse_definition(value, fallback_identifier, depth + 1)? {
                if level
                    .definitions
                    .insert(definition.identifier.clone(), definition.clone())
                    .is_some()
                {
                    return Err(GraphError::DuplicateEntity(definition.identifier));
                }
            }
        }
    }
    for node in level.nodes.values_mut() {
        let definition_identifier = GraphIdentifier::String(node.type_identifier.clone());
        if level.definitions.contains_key(&definition_identifier) {
            node.subgraph_definition = Some(definition_identifier);
        }
    }
    recompute_group_memberships(&mut level);
    Ok(level)
}

fn optional_array_field<'a>(
    object: &'a Map<String, Value>,
    key: &str,
) -> Result<Option<&'a Vec<Value>>, GraphError> {
    object
        .get(key)
        .map(|value| {
            value.as_array().ok_or_else(|| GraphError::InvalidWorkflow {
                path: format!("$.{key}"),
                reason: format!("{key} must be an array"),
            })
        })
        .transpose()
}

fn parse_node(value: &Value) -> Result<GraphNode, GraphError> {
    let object = value
        .as_object()
        .ok_or_else(|| GraphError::Serialization("workflow node must be an object".to_owned()))?;
    let identifier = object
        .get("id")
        .and_then(GraphIdentifier::from_value)
        .ok_or_else(|| GraphError::Serialization("workflow node id is missing".to_owned()))?;
    let type_identifier = object
        .get("type")
        .or_else(|| object.get("class_type"))
        .and_then(Value::as_str)
        .unwrap_or("Unknown")
        .to_owned();
    let title = object
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or(&type_identifier)
        .to_owned();
    let position = parse_optional_point(object.get("pos"), "workflow node position")?;
    let size = parse_optional_size(object.get("size"), "workflow node size")?;
    let inputs = parse_ports(object.get("inputs"))?;
    let outputs = parse_ports(object.get("outputs"))?;
    let widget_names = object
        .get("properties")
        .and_then(|properties| properties.get("widget_input_names"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let widgets = parse_widgets(object, &widget_names)?;
    let mode = match object
        .get("mode")
        .and_then(Value::as_i64)
        .unwrap_or_default()
    {
        1 => GraphNodeMode::OnEvent,
        2 => GraphNodeMode::Never,
        3 => GraphNodeMode::OnTrigger,
        4 => GraphNodeMode::Bypass,
        _ => GraphNodeMode::Always,
    };
    Ok(GraphNode {
        identifier,
        type_identifier,
        title,
        position,
        size,
        inputs,
        outputs,
        widgets,
        mode,
        pinned: object
            .get("flags")
            .and_then(|flags| flags.get("pinned"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        collapsed: object
            .get("flags")
            .and_then(|flags| flags.get("collapsed"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        color: object
            .get("color")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        subgraph_definition: object
            .get("subgraph_id")
            .and_then(GraphIdentifier::from_value),
        quarantine: BTreeMap::new(),
        source_fields: object.clone(),
    })
}

fn parse_widgets(
    object: &Map<String, Value>,
    widget_names: &[String],
) -> Result<Vec<GraphWidget>, GraphError> {
    let workflow_values = object.get("widgets_values");
    let Some(native) = object.get(NATIVE_WIDGETS_FIELD) else {
        return match workflow_values {
            Some(Value::Array(values)) => Ok(values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    GraphWidget::preserved(
                        widget_names
                            .get(index)
                            .cloned()
                            .unwrap_or_else(|| format!("widget-{index}")),
                        value.clone(),
                    )
                })
                .collect()),
            Some(Value::Object(values)) => Ok(values
                .iter()
                .map(|(identifier, value)| GraphWidget::preserved(identifier, value.clone()))
                .collect()),
            Some(_) => Err(GraphError::InvalidWorkflow {
                path: "$.nodes[].widgets_values".to_owned(),
                reason: "widget values must be an array or object".to_owned(),
            }),
            None => Ok(Vec::new()),
        };
    };
    let mut envelope: NativeWidgetsEnvelope =
        serde_json::from_value(native.clone()).map_err(|error| GraphError::InvalidWorkflow {
            path: format!("$.nodes[].{NATIVE_WIDGETS_FIELD}"),
            reason: error.to_string(),
        })?;
    if envelope.version != NATIVE_WIDGETS_VERSION {
        return Err(GraphError::InvalidWorkflow {
            path: format!("$.nodes[].{NATIVE_WIDGETS_FIELD}.version"),
            reason: format!("native widget schema {} is unsupported", envelope.version),
        });
    }
    match workflow_values {
        Some(Value::Array(values)) => {
            if values.len() != envelope.widgets.len() {
                return Err(GraphError::InvalidWorkflow {
                    path: "$.nodes[].widgets_values".to_owned(),
                    reason: format!(
                        "widget value count {} does not match native widget count {}",
                        values.len(),
                        envelope.widgets.len()
                    ),
                });
            }
            for (widget, value) in envelope.widgets.iter_mut().zip(values) {
                widget.value = value.clone();
            }
        }
        Some(Value::Object(values)) => {
            if values.len() != envelope.widgets.len() {
                return Err(GraphError::InvalidWorkflow {
                    path: "$.nodes[].widgets_values".to_owned(),
                    reason: "widget value names do not match native widget metadata".to_owned(),
                });
            }
            for widget in &mut envelope.widgets {
                widget.value = values.get(&widget.identifier).cloned().ok_or_else(|| {
                    GraphError::InvalidWorkflow {
                        path: "$.nodes[].widgets_values".to_owned(),
                        reason: format!("widget `{}` has no workflow value", widget.identifier),
                    }
                })?;
            }
        }
        Some(_) => {
            return Err(GraphError::InvalidWorkflow {
                path: "$.nodes[].widgets_values".to_owned(),
                reason: "widget values must be an array or object".to_owned(),
            });
        }
        None if !envelope.widgets.is_empty() => {
            return Err(GraphError::InvalidWorkflow {
                path: "$.nodes[].widgets_values".to_owned(),
                reason: "native widget metadata requires workflow values".to_owned(),
            });
        }
        None => {}
    }
    for widget in &envelope.widgets {
        widget.validate_schema()?;
        if widget.normalize(widget.value.clone())? != widget.value
            || widget.normalize(widget.prompt_value.clone())? != widget.prompt_value
        {
            return Err(GraphError::InvalidWidgetValue(widget.identifier.clone()));
        }
    }
    Ok(envelope.widgets)
}

fn parse_ports(value: Option<&Value>) -> Result<Vec<GraphPort>, GraphError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let ports = value
        .as_array()
        .ok_or_else(|| GraphError::InvalidWorkflow {
            path: "$.nodes[].ports".to_owned(),
            reason: "node ports must be an array".to_owned(),
        })?;
    ports
        .iter()
        .enumerate()
        .map(|(index, port)| {
            let port = port.as_object().ok_or_else(|| {
                GraphError::Serialization(format!("workflow port {index} must be an object"))
            })?;
            let name = port
                .get("name")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("port-{index}"));
            let type_name = port.get("type").and_then(Value::as_str).unwrap_or("ANY");
            let unknown = port
                .iter()
                .filter(|(key, _)| {
                    !matches!(key.as_str(), "name" | "type" | "multiple" | "dynamic")
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            Ok(GraphPort {
                name,
                port_type: GraphPortType::from_name(type_name),
                multiple: port
                    .get("multiple")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                dynamic: port
                    .get("dynamic")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                unknown,
            })
        })
        .collect()
}

fn parse_link(value: &Value) -> Result<GraphLink, GraphError> {
    if let Some(values) = value.as_array() {
        if values.len() < 6 {
            return Err(GraphError::Serialization(
                "workflow link array must contain six entries".to_owned(),
            ));
        }
        return Ok(GraphLink {
            identifier: GraphIdentifier::from_value(&values[0])
                .ok_or_else(|| GraphError::Serialization("link id is invalid".to_owned()))?,
            origin_node: GraphIdentifier::from_value(&values[1])
                .ok_or_else(|| GraphError::Serialization("link origin is invalid".to_owned()))?,
            origin_slot: value_usize(&values[2], "link origin slot")?,
            target_node: GraphIdentifier::from_value(&values[3])
                .ok_or_else(|| GraphError::Serialization("link target is invalid".to_owned()))?,
            target_slot: value_usize(&values[4], "link target slot")?,
            type_name: values[5].as_str().unwrap_or("ANY").to_owned(),
            parent_reroute: values.get(6).and_then(GraphIdentifier::from_value),
            source: value.clone(),
        });
    }
    let object = value.as_object().ok_or_else(|| {
        GraphError::Serialization("workflow link must be an array or object".to_owned())
    })?;
    Ok(GraphLink {
        identifier: object
            .get("id")
            .and_then(GraphIdentifier::from_value)
            .ok_or_else(|| GraphError::Serialization("link id is invalid".to_owned()))?,
        origin_node: object
            .get("origin_id")
            .or_else(|| object.get("origin_node"))
            .and_then(GraphIdentifier::from_value)
            .ok_or_else(|| GraphError::Serialization("link origin is invalid".to_owned()))?,
        origin_slot: object
            .get("origin_slot")
            .map(|value| value_usize(value, "link origin slot"))
            .transpose()?
            .unwrap_or_default(),
        target_node: object
            .get("target_id")
            .or_else(|| object.get("target_node"))
            .and_then(GraphIdentifier::from_value)
            .ok_or_else(|| GraphError::Serialization("link target is invalid".to_owned()))?,
        target_slot: object
            .get("target_slot")
            .map(|value| value_usize(value, "link target slot"))
            .transpose()?
            .unwrap_or_default(),
        type_name: object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("ANY")
            .to_owned(),
        parent_reroute: object
            .get("parentId")
            .or_else(|| object.get("parent_id"))
            .and_then(GraphIdentifier::from_value),
        source: value.clone(),
    })
}

fn parse_group(value: &Value, index: usize) -> Result<GraphGroup, GraphError> {
    let object = value
        .as_object()
        .ok_or_else(|| GraphError::Serialization("workflow group must be an object".to_owned()))?;
    let identifier = object
        .get("id")
        .and_then(GraphIdentifier::from_value)
        .unwrap_or_else(|| GraphIdentifier::String(format!("legacy-group-{index}")));
    let (position, size) = if let Some(bounding) = object.get("bounding") {
        let values = bounding.as_array().ok_or_else(|| {
            GraphError::Serialization("workflow group bounding must be an array".to_owned())
        })?;
        if values.len() < 4 {
            return Err(GraphError::Serialization(
                "workflow group bounding must contain four numbers".to_owned(),
            ));
        }
        let position = GraphPoint {
            x: values[0].as_f64().ok_or_else(|| {
                GraphError::Serialization("workflow group x is invalid".to_owned())
            })? as f32,
            y: values[1].as_f64().ok_or_else(|| {
                GraphError::Serialization("workflow group y is invalid".to_owned())
            })? as f32,
        };
        let size = GraphSize {
            width: values[2].as_f64().ok_or_else(|| {
                GraphError::Serialization("workflow group width is invalid".to_owned())
            })? as f32,
            height: values[3].as_f64().ok_or_else(|| {
                GraphError::Serialization("workflow group height is invalid".to_owned())
            })? as f32,
        };
        position.validate()?;
        size.validate()?;
        (position, size)
    } else {
        (
            parse_optional_point(object.get("pos"), "workflow group position")?,
            parse_optional_size(object.get("size"), "workflow group size")?,
        )
    };
    let mut source_fields = object.clone();
    source_fields.remove("nodes");
    Ok(GraphGroup {
        identifier,
        title: object
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("Group")
            .to_owned(),
        bounds: GraphRect {
            origin: position,
            size,
        },
        node_ids: BTreeSet::new(),
        collapsed: object
            .get("collapsed")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        pinned: object
            .get("pinned")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        color: object
            .get("color")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        source_fields,
    })
}

fn parse_reroute(value: &Value) -> Result<GraphReroute, GraphError> {
    let object = value.as_object().ok_or_else(|| {
        GraphError::Serialization("workflow reroute must be an object".to_owned())
    })?;
    let floating_type = match object.get("floating") {
        None => None,
        Some(floating) => {
            let slot_type = floating
                .as_object()
                .and_then(|floating| floating.get("slotType"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    GraphError::Serialization(
                        "workflow reroute floating.slotType must be input or output".to_owned(),
                    )
                })?;
            if !matches!(slot_type, "input" | "output") {
                return Err(GraphError::Serialization(
                    "workflow reroute floating.slotType must be input or output".to_owned(),
                ));
            }
            Some(slot_type.to_owned())
        }
    };
    Ok(GraphReroute {
        identifier: object
            .get("id")
            .and_then(GraphIdentifier::from_value)
            .ok_or_else(|| GraphError::Serialization("reroute id is invalid".to_owned()))?,
        position: parse_optional_point(object.get("pos"), "workflow reroute position")?,
        parent: object
            .get("parentId")
            .or_else(|| object.get("parent_id"))
            .and_then(GraphIdentifier::from_value),
        floating_type,
        source_fields: object.clone(),
    })
}

fn parse_definition(
    value: &Value,
    fallback_identifier: Option<&str>,
    depth: usize,
) -> Result<Option<SubgraphDefinition>, GraphError> {
    let Some(object) = value.as_object() else {
        return Ok(None);
    };
    let identifier = object
        .get("id")
        .and_then(GraphIdentifier::from_value)
        .or_else(|| fallback_identifier.map(GraphIdentifier::from));
    let Some(identifier) = identifier else {
        return Ok(None);
    };
    let graph_inline = !object.contains_key("graph");
    let graph_value = object.get("graph").unwrap_or(value);
    let graph = parse_level(graph_value, depth)?;
    let input_boundary = object
        .get("inputNode")
        .and_then(|node| node.get("id"))
        .and_then(GraphIdentifier::from_value);
    let output_boundary = object
        .get("outputNode")
        .and_then(|node| node.get("id"))
        .and_then(GraphIdentifier::from_value);
    let inputs = parse_subgraph_ports(
        object.get("inputs"),
        "input",
        &graph,
        input_boundary.as_ref(),
    )?;
    let outputs = parse_subgraph_ports(
        object.get("outputs"),
        "output",
        &graph,
        output_boundary.as_ref(),
    )?;
    let known = BTreeSet::from([
        "id",
        "name",
        "graph",
        "nodes",
        "links",
        "groups",
        "reroutes",
        "definitions",
        "version",
        "inputs",
        "outputs",
        "published",
        "description",
        "search_aliases",
        EXPOSED_WIDGETS_FIELD,
    ]);
    let unknown = object
        .iter()
        .filter(|(key, _)| !known.contains(key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Ok(Some(SubgraphDefinition {
        identifier,
        name: object
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("Subgraph")
            .to_owned(),
        graph: Box::new(graph),
        inputs,
        outputs,
        published: object
            .get("published")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        description: object
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        search_aliases: object
            .get("search_aliases")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        exposed_widgets: object
            .get(EXPOSED_WIDGETS_FIELD)
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| GraphError::InvalidWorkflow {
                path: format!("$.definitions[].{EXPOSED_WIDGETS_FIELD}"),
                reason: error.to_string(),
            })?
            .unwrap_or_default(),
        graph_inline,
        unknown,
    }))
}

fn parse_subgraph_ports(
    value: Option<&Value>,
    kind: &str,
    graph: &GraphLevel,
    boundary_node: Option<&GraphIdentifier>,
) -> Result<Vec<SubgraphPort>, GraphError> {
    let Some(values) = value else {
        return Ok(Vec::new());
    };
    let values = values.as_array().ok_or_else(|| {
        GraphError::Serialization(format!("subgraph {kind} ports must be an array"))
    })?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let object = value.as_object().ok_or_else(|| {
                GraphError::Serialization(format!("subgraph {kind} port {index} must be an object"))
            })?;
            let identifier = object
                .get("identifier")
                .or_else(|| object.get("id"))
                .or_else(|| object.get("name"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .ok_or_else(|| {
                    GraphError::Serialization(format!(
                        "subgraph {kind} port {index} identifier is missing"
                    ))
                })?;
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&identifier)
                .to_owned();
            let port_type = object
                .get("type")
                .and_then(Value::as_str)
                .map(GraphPortType::from_name)
                .or_else(|| {
                    object.get("port_type").and_then(|value| {
                        serde_json::from_value::<GraphPortType>(value.clone()).ok()
                    })
                })
                .unwrap_or(GraphPortType::Any);
            let explicit_internal_node = object
                .get("internal_node")
                .or_else(|| object.get("node_id"))
                .or_else(|| object.get("node"))
                .and_then(GraphIdentifier::from_value);
            let explicit_internal_slot = object
                .get("internal_slot")
                .or_else(|| object.get("slot"))
                .map(|value| value_usize(value, "subgraph port slot"))
                .transpose()?;
            let boundary_endpoint = boundary_node.and_then(|boundary_node| {
                graph.links.values().find_map(|link| {
                    if kind == "input"
                        && link.origin_node == *boundary_node
                        && link.origin_slot == index
                    {
                        Some((link.target_node.clone(), link.target_slot))
                    } else if kind == "output"
                        && link.target_node == *boundary_node
                        && link.target_slot == index
                    {
                        Some((link.origin_node.clone(), link.origin_slot))
                    } else {
                        None
                    }
                })
            });
            let internal_node = explicit_internal_node
                .or_else(|| boundary_endpoint.as_ref().map(|(node, _)| node.clone()));
            let internal_slot = explicit_internal_slot
                .or_else(|| boundary_endpoint.map(|(_, slot)| slot))
                .unwrap_or_default();
            Ok(SubgraphPort {
                identifier,
                name,
                port_type,
                internal_node,
                internal_slot,
                source_fields: object.clone(),
            })
        })
        .collect()
}

fn parse_point(value: Option<&Value>) -> Option<GraphPoint> {
    let values = value?.as_array()?;
    Some(GraphPoint {
        x: values.first()?.as_f64()? as f32,
        y: values.get(1)?.as_f64()? as f32,
    })
}

fn parse_optional_point(value: Option<&Value>, label: &str) -> Result<GraphPoint, GraphError> {
    let Some(value) = value else {
        return Ok(GraphPoint::default());
    };
    let point = parse_point(Some(value))
        .ok_or_else(|| GraphError::Serialization(format!("{label} is invalid")))?;
    point.validate()?;
    Ok(point)
}

fn parse_size(value: Option<&Value>) -> Option<GraphSize> {
    let values = value?.as_array()?;
    Some(GraphSize {
        width: values.first()?.as_f64()? as f32,
        height: values.get(1)?.as_f64()? as f32,
    })
}

fn parse_optional_size(value: Option<&Value>, label: &str) -> Result<GraphSize, GraphError> {
    let Some(value) = value else {
        return Ok(GraphSize::default());
    };
    let size = parse_size(Some(value))
        .ok_or_else(|| GraphError::Serialization(format!("{label} is invalid")))?;
    size.validate()?;
    Ok(size)
}

fn value_usize(value: &Value, label: &str) -> Result<usize, GraphError> {
    let value = value
        .as_u64()
        .ok_or_else(|| GraphError::Serialization(format!("{label} is invalid")))?;
    usize::try_from(value).map_err(|_| GraphError::Serialization(format!("{label} is too large")))
}

fn parse_viewport(object: &Map<String, Value>) -> Result<GraphViewport, GraphError> {
    let value = object
        .get("viewport")
        .or_else(|| object.get("state").and_then(|state| state.get("viewport")))
        .or_else(|| object.get("extra").and_then(|extra| extra.get("ds")));
    let Some(value) = value else {
        return Ok(GraphViewport::default());
    };
    let value = value.as_object().ok_or_else(|| {
        GraphError::Serialization("workflow viewport must be an object".to_owned())
    })?;
    let offset = parse_optional_point(value.get("offset"), "workflow viewport offset")?;
    let scale = value.get("scale").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    let viewport = GraphViewport {
        offset,
        scale,
        minimap_visible: value
            .get("minimap_visible")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        links_visible: value
            .get("links_visible")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        locked: value
            .get("locked")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };
    viewport.validate()?;
    Ok(viewport)
}

fn serialize_viewport(
    object: &mut Map<String, Value>,
    viewport: &GraphViewport,
    format: Option<WorkflowFormat>,
) {
    let value = json!({
        "offset": [viewport.offset.x, viewport.offset.y],
        "scale": viewport.scale,
        "minimap_visible": viewport.minimap_visible,
        "links_visible": viewport.links_visible,
        "locked": viewport.locked,
    });
    if format == Some(WorkflowFormat::Schema04) {
        let extra = object
            .entry("extra")
            .or_insert_with(|| Value::Object(Map::new()));
        if !extra.is_object() {
            *extra = Value::Object(Map::new());
        }
        if let Some(extra) = extra.as_object_mut() {
            extra.insert("ds".to_owned(), value);
        }
    } else {
        let state = object
            .entry("state")
            .or_insert_with(|| Value::Object(Map::new()));
        if !state.is_object() {
            *state = Value::Object(Map::new());
        }
        if let Some(state) = state.as_object_mut() {
            state.insert("viewport".to_owned(), value);
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Schema1LevelIdentifiers {
    links: BTreeMap<GraphIdentifier, i64>,
    groups: BTreeMap<GraphIdentifier, i64>,
    reroutes: BTreeMap<GraphIdentifier, i64>,
    definitions: BTreeMap<GraphIdentifier, Uuid>,
}

impl Schema1LevelIdentifiers {
    fn new(level: &GraphLevel, graph_identity: Uuid) -> Self {
        Self {
            links: schema1_numeric_identifiers(level.links.keys()),
            groups: schema1_numeric_identifiers(level.groups.keys()),
            reroutes: schema1_numeric_identifiers(level.reroutes.keys()),
            definitions: level
                .definitions
                .keys()
                .map(|identifier| {
                    let uuid = match identifier {
                        GraphIdentifier::String(identifier) => Uuid::parse_str(identifier)
                            .unwrap_or_else(|_| {
                                Uuid::new_v5(&graph_identity, identifier.as_bytes())
                            }),
                        GraphIdentifier::Integer(identifier) => Uuid::new_v5(
                            &graph_identity,
                            format!("subgraph-{identifier}").as_bytes(),
                        ),
                    };
                    (identifier.clone(), uuid)
                })
                .collect(),
        }
    }
}

fn schema1_numeric_identifiers<'a>(
    identifiers: impl Iterator<Item = &'a GraphIdentifier>,
) -> BTreeMap<GraphIdentifier, i64> {
    let identifiers = identifiers.cloned().collect::<Vec<_>>();
    let mut used = BTreeSet::new();
    let mut mapping = BTreeMap::new();
    for identifier in &identifiers {
        if let GraphIdentifier::Integer(value) = identifier
            && *value >= 0
            && used.insert(*value)
        {
            mapping.insert(identifier.clone(), *value);
        }
    }
    let mut next = used.iter().next_back().copied().unwrap_or_default();
    for identifier in identifiers {
        if mapping.contains_key(&identifier) {
            continue;
        }
        loop {
            next = next.saturating_add(1);
            if used.insert(next) {
                mapping.insert(identifier, next);
                break;
            }
        }
    }
    mapping
}

fn serialize_level(
    level: &GraphLevel,
    format: Option<WorkflowFormat>,
    graph_identity: Option<Uuid>,
) -> Result<Value, GraphError> {
    let graph_identity = graph_identity.or_else(|| {
        level
            .source_fields
            .get("id")
            .and_then(Value::as_str)
            .and_then(|identifier| Uuid::parse_str(identifier).ok())
    });
    let schema_one_identifiers = graph_identity
        .filter(|_| {
            matches!(
                format,
                Some(WorkflowFormat::Schema1 | WorkflowFormat::Schema04)
            )
        })
        .map(|identity| Schema1LevelIdentifiers::new(level, identity));
    let mut object = level.source_fields.clone();
    serialize_viewport(&mut object, &level.viewport, format);
    object.insert(
        "nodes".to_owned(),
        Value::Array(
            level
                .nodes
                .values()
                .enumerate()
                .map(|(order, node)| {
                    serialize_node(node, level, order, format, schema_one_identifiers.as_ref())
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
    );
    object.insert(
        "links".to_owned(),
        Value::Array(
            level
                .links
                .values()
                .map(|link| serialize_link(link, format, schema_one_identifiers.as_ref()))
                .collect(),
        ),
    );
    object.insert(
        "groups".to_owned(),
        Value::Array(
            level
                .groups
                .values()
                .map(|group| serialize_group(group, format, schema_one_identifiers.as_ref()))
                .collect(),
        ),
    );
    let serialized_reroutes = Value::Array(
        level
            .reroutes
            .values()
            .map(|reroute| serialize_reroute(reroute, format, schema_one_identifiers.as_ref()))
            .collect(),
    );
    if format == Some(WorkflowFormat::Schema04) {
        object.remove("reroutes");
        let extra = object
            .entry("extra")
            .or_insert_with(|| Value::Object(Map::new()));
        if !extra.is_object() {
            *extra = Value::Object(Map::new());
        }
        if let Some(extra) = extra.as_object_mut() {
            extra.insert("reroutes".to_owned(), serialized_reroutes);
            let extensions = level
                .links
                .values()
                .filter_map(|link| {
                    let parent = link.parent_reroute.as_ref()?;
                    let identifiers = schema_one_identifiers.as_ref()?;
                    Some(json!({
                        "id": identifiers.links.get(&link.identifier)?,
                        "parentId": identifiers.reroutes.get(parent)?,
                    }))
                })
                .collect::<Vec<_>>();
            if extensions.is_empty() {
                extra.remove("linkExtensions");
            } else {
                extra.insert("linkExtensions".to_owned(), Value::Array(extensions));
            }
        }
    } else {
        object.insert("reroutes".to_owned(), serialized_reroutes);
    }
    if !level.definitions.is_empty() || level.source_fields.contains_key("definitions") {
        let definitions = level
            .definitions
            .values()
            .map(|definition| {
                let identity = schema_one_identifiers
                    .as_ref()
                    .and_then(|identifiers| identifiers.definitions.get(&definition.identifier))
                    .copied();
                let definition_format = if format == Some(WorkflowFormat::Schema04) {
                    Some(WorkflowFormat::Schema1)
                } else {
                    format
                };
                serialize_definition(definition, definition_format, identity)
            })
            .collect::<Result<Vec<_>, GraphError>>()?;
        let serialized = match level.source_fields.get("definitions") {
            Some(Value::Object(source)) if source.contains_key("subgraphs") => {
                let mut source = source.clone();
                source.insert("subgraphs".to_owned(), Value::Array(definitions));
                Value::Object(source)
            }
            Some(Value::Object(_)) => Value::Object(
                level
                    .definitions
                    .keys()
                    .map(GraphIdentifier::text)
                    .zip(definitions)
                    .collect(),
            ),
            Some(Value::Array(_)) => Value::Array(definitions),
            _ if matches!(
                format,
                Some(WorkflowFormat::Schema1 | WorkflowFormat::Schema04)
            ) =>
            {
                Value::Object(Map::from_iter([(
                    "subgraphs".to_owned(),
                    Value::Array(definitions),
                )]))
            }
            _ => Value::Array(definitions),
        };
        object.insert("definitions".to_owned(), serialized);
    }
    if format == Some(WorkflowFormat::Schema1) {
        let identity = graph_identity.ok_or_else(|| {
            GraphError::Serialization("schema 1 graph identity is missing".to_owned())
        })?;
        object.insert("id".to_owned(), Value::String(identity.to_string()));
        if !object.get("revision").is_some_and(Value::is_number) {
            object.insert("revision".to_owned(), Value::from(0));
        }
        if !object
            .get("config")
            .is_some_and(|value| value.is_object() || value.is_null())
        {
            object.insert("config".to_owned(), Value::Object(Map::new()));
        }
        let identifiers = schema_one_identifiers.as_ref().ok_or_else(|| {
            GraphError::Serialization("schema 1 identifiers are missing".to_owned())
        })?;
        let state = object
            .entry("state")
            .or_insert_with(|| Value::Object(Map::new()));
        if !state.is_object() {
            *state = Value::Object(Map::new());
        }
        if let Some(state) = state.as_object_mut() {
            state.insert(
                "lastNodeId".to_owned(),
                Value::from(maximum_numeric_key(level.nodes.keys())),
            );
            state.insert(
                "lastLinkId".to_owned(),
                Value::from(
                    identifiers
                        .links
                        .values()
                        .copied()
                        .max()
                        .unwrap_or_default(),
                ),
            );
            state.insert(
                "lastGroupId".to_owned(),
                Value::from(
                    identifiers
                        .groups
                        .values()
                        .copied()
                        .max()
                        .unwrap_or_default(),
                ),
            );
            state.insert(
                "lastRerouteId".to_owned(),
                Value::from(
                    identifiers
                        .reroutes
                        .values()
                        .copied()
                        .max()
                        .unwrap_or_default(),
                ),
            );
        }
        object.insert("version".to_owned(), Value::from(1));
    } else {
        let identity = graph_identity.ok_or_else(|| {
            GraphError::Serialization("schema 0.4 graph identity is missing".to_owned())
        })?;
        object.insert("id".to_owned(), Value::String(identity.to_string()));
        object.insert("version".to_owned(), Value::from(0.4));
        if !object.get("revision").is_some_and(Value::is_number) {
            object.insert("revision".to_owned(), Value::from(0));
        }
        object.insert(
            "last_node_id".to_owned(),
            Value::from(maximum_numeric_key(level.nodes.keys())),
        );
        object.insert(
            "last_link_id".to_owned(),
            Value::from(
                schema_one_identifiers
                    .as_ref()
                    .and_then(|identifiers| identifiers.links.values().copied().max())
                    .unwrap_or_default(),
            ),
        );
    }
    Ok(Value::Object(object))
}

fn maximum_numeric_key<'a>(identifiers: impl Iterator<Item = &'a GraphIdentifier>) -> i64 {
    identifiers
        .filter_map(|identifier| match identifier {
            GraphIdentifier::Integer(value) if *value >= 0 => Some(*value),
            GraphIdentifier::String(value) => value.parse::<i64>().ok().filter(|value| *value >= 0),
            GraphIdentifier::Integer(_) => None,
        })
        .max()
        .unwrap_or_default()
}

fn serialize_definition(
    definition: &SubgraphDefinition,
    format: Option<WorkflowFormat>,
    schema_one_identity: Option<Uuid>,
) -> Result<Value, GraphError> {
    let mut object = definition
        .unknown
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Map<_, _>>();
    let mut graph = serialize_level(&definition.graph, format, schema_one_identity)?;
    if format == Some(WorkflowFormat::Schema1) {
        let identity = schema_one_identity.ok_or_else(|| {
            GraphError::Serialization("schema 1 subgraph identity is missing".to_owned())
        })?;
        let identifiers = Schema1LevelIdentifiers::new(&definition.graph, identity);
        let input_boundary = ensure_subgraph_boundary(&mut object, "inputNode", -10, -160.0)?;
        let output_boundary = ensure_subgraph_boundary(&mut object, "outputNode", -20, 160.0)?;
        let graph_object = graph.as_object_mut().ok_or_else(|| {
            GraphError::Serialization("serialized subgraph must be an object".to_owned())
        })?;
        let links = graph_object
            .entry("links")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| {
                GraphError::Serialization("serialized subgraph links must be an array".to_owned())
            })?;
        let mut next_link_identifier = identifiers
            .links
            .values()
            .copied()
            .max()
            .unwrap_or_default();
        let inputs = serialize_schema_one_subgraph_ports(
            &definition.inputs,
            "input",
            &definition.graph,
            &identifiers,
            &input_boundary,
            links,
            &mut next_link_identifier,
        );
        let outputs = serialize_schema_one_subgraph_ports(
            &definition.outputs,
            "output",
            &definition.graph,
            &identifiers,
            &output_boundary,
            links,
            &mut next_link_identifier,
        );
        if let Some(state) = graph_object.get_mut("state").and_then(Value::as_object_mut) {
            state.insert("lastLinkId".to_owned(), Value::from(next_link_identifier));
        }
        for (key, value) in graph_object.iter() {
            object.insert(key.clone(), value.clone());
        }
        object.insert("inputs".to_owned(), Value::Array(inputs));
        object.insert("outputs".to_owned(), Value::Array(outputs));
        if !object.get("revision").is_some_and(Value::is_number) {
            object.insert("revision".to_owned(), Value::from(0));
        }
        object.remove("graph");
    } else {
        if definition.graph_inline {
            let graph = graph.as_object().ok_or_else(|| {
                GraphError::Serialization("serialized subgraph must be an object".to_owned())
            })?;
            for (key, value) in graph {
                object.insert(key.clone(), value.clone());
            }
        } else {
            object.insert("graph".to_owned(), graph);
        }
        object.insert(
            "inputs".to_owned(),
            serde_json::to_value(&definition.inputs)
                .map_err(|error| GraphError::Serialization(error.to_string()))?,
        );
        object.insert(
            "outputs".to_owned(),
            serde_json::to_value(&definition.outputs)
                .map_err(|error| GraphError::Serialization(error.to_string()))?,
        );
    }
    object.insert(
        "id".to_owned(),
        schema_one_identity
            .map_or_else(
                || serde_json::to_value(&definition.identifier),
                |identity| Ok(Value::String(identity.to_string())),
            )
            .map_err(|error| GraphError::Serialization(error.to_string()))?,
    );
    object.insert("name".to_owned(), Value::String(definition.name.clone()));
    object.insert("published".to_owned(), Value::Bool(definition.published));
    object.insert(
        "description".to_owned(),
        Value::String(definition.description.clone()),
    );
    object.insert(
        "search_aliases".to_owned(),
        serde_json::to_value(&definition.search_aliases)
            .map_err(|error| GraphError::Serialization(error.to_string()))?,
    );
    object.insert(
        EXPOSED_WIDGETS_FIELD.to_owned(),
        serde_json::to_value(&definition.exposed_widgets)
            .map_err(|error| GraphError::Serialization(error.to_string()))?,
    );
    Ok(Value::Object(object))
}

fn ensure_subgraph_boundary(
    object: &mut Map<String, Value>,
    field: &str,
    default_identifier: i64,
    default_x: f64,
) -> Result<GraphIdentifier, GraphError> {
    let boundary = object.entry(field).or_insert_with(
        || json!({"id": default_identifier, "bounding": [default_x, 0.0, 120.0, 60.0]}),
    );
    let boundary = boundary
        .as_object_mut()
        .ok_or_else(|| GraphError::Serialization(format!("schema 1 {field} must be an object")))?;
    if !boundary
        .get("bounding")
        .is_some_and(|value| value.as_array().is_some_and(|values| values.len() == 4))
    {
        boundary.insert("bounding".to_owned(), json!([default_x, 0.0, 120.0, 60.0]));
    }
    let identifier = boundary
        .get("id")
        .and_then(GraphIdentifier::from_value)
        .unwrap_or(GraphIdentifier::Integer(default_identifier));
    boundary.insert("id".to_owned(), json!(identifier));
    Ok(identifier)
}

fn serialize_schema_one_subgraph_ports(
    ports: &[SubgraphPort],
    direction: &str,
    graph: &GraphLevel,
    identifiers: &Schema1LevelIdentifiers,
    boundary_node: &GraphIdentifier,
    serialized_links: &mut Vec<Value>,
    next_link_identifier: &mut i64,
) -> Vec<Value> {
    ports
        .iter()
        .enumerate()
        .map(|(index, port)| {
            let mut object = port.source_fields.clone();
            let identifier = Uuid::parse_str(&port.identifier).unwrap_or_else(|_| {
                Uuid::new_v5(
                    &Uuid::NAMESPACE_OID,
                    format!("{direction}:{index}:{}", port.identifier).as_bytes(),
                )
            });
            let mut link_ids = graph
                .links
                .values()
                .filter(|link| {
                    (direction == "input"
                        && link.origin_node == *boundary_node
                        && link.origin_slot == index)
                        || (direction == "output"
                            && link.target_node == *boundary_node
                            && link.target_slot == index)
                })
                .filter_map(|link| identifiers.links.get(&link.identifier).copied())
                .collect::<Vec<_>>();
            if link_ids.is_empty()
                && let Some(internal_node) = &port.internal_node
            {
                *next_link_identifier = next_link_identifier.saturating_add(1);
                let (origin_node, origin_slot, target_node, target_slot) = if direction == "input" {
                    (boundary_node, index, internal_node, port.internal_slot)
                } else {
                    (internal_node, port.internal_slot, boundary_node, index)
                };
                serialized_links.push(json!({
                    "id": *next_link_identifier,
                    "origin_id": origin_node,
                    "origin_slot": origin_slot,
                    "target_id": target_node,
                    "target_slot": target_slot,
                    "type": port.port_type.display_name(),
                }));
                link_ids.push(*next_link_identifier);
            }
            object.insert("id".to_owned(), Value::String(identifier.to_string()));
            object.insert("name".to_owned(), Value::String(port.name.clone()));
            object.insert(
                "type".to_owned(),
                Value::String(port.port_type.display_name()),
            );
            object.insert("linkIds".to_owned(), json!(link_ids));
            Value::Object(object)
        })
        .collect()
}

fn serialize_node(
    node: &GraphNode,
    graph: &GraphLevel,
    order: usize,
    format: Option<WorkflowFormat>,
    schema_one_identifiers: Option<&Schema1LevelIdentifiers>,
) -> Result<Value, GraphError> {
    let mut object = node.source_fields.clone();
    object.insert(
        "id".to_owned(),
        serde_json::to_value(&node.identifier)
            .map_err(|error| GraphError::Serialization(error.to_string()))?,
    );
    let type_identifier = node
        .subgraph_definition
        .as_ref()
        .and_then(|identifier| schema_one_identifiers?.definitions.get(identifier))
        .map(ToString::to_string)
        .unwrap_or_else(|| node.type_identifier.clone());
    object.insert("type".to_owned(), Value::String(type_identifier));
    object.insert("title".to_owned(), Value::String(node.title.clone()));
    object.insert("pos".to_owned(), json!([node.position.x, node.position.y]));
    object.insert(
        "size".to_owned(),
        json!([node.size.width, node.size.height]),
    );
    object.insert(
        "inputs".to_owned(),
        Value::Array(
            node.inputs
                .iter()
                .enumerate()
                .map(|(slot, port)| {
                    let links = graph
                        .links
                        .values()
                        .filter(|link| {
                            link.target_node == node.identifier && link.target_slot == slot
                        })
                        .map(|link| link.identifier.clone())
                        .map(|identifier| {
                            schema_one_identifiers
                                .and_then(|identifiers| identifiers.links.get(&identifier))
                                .map_or(identifier, |identifier| {
                                    GraphIdentifier::Integer(*identifier)
                                })
                        })
                        .collect::<Vec<_>>();
                    let mut value = serialize_port(port);
                    if let Some(object) = value.as_object_mut() {
                        object.insert(
                            "link".to_owned(),
                            links
                                .first()
                                .map_or(Value::Null, |identifier| json!(identifier)),
                        );
                        if port.multiple || object.contains_key("links") {
                            object.insert("links".to_owned(), json!(links));
                        }
                    }
                    value
                })
                .collect(),
        ),
    );
    object.insert(
        "outputs".to_owned(),
        Value::Array(
            node.outputs
                .iter()
                .enumerate()
                .map(|(slot, port)| {
                    let links = graph
                        .links
                        .values()
                        .filter(|link| {
                            link.origin_node == node.identifier && link.origin_slot == slot
                        })
                        .map(|link| link.identifier.clone())
                        .map(|identifier| {
                            schema_one_identifiers
                                .and_then(|identifiers| identifiers.links.get(&identifier))
                                .map_or(identifier, |identifier| {
                                    GraphIdentifier::Integer(*identifier)
                                })
                        })
                        .collect::<Vec<_>>();
                    let mut value = serialize_port(port);
                    if let Some(object) = value.as_object_mut() {
                        object.insert("links".to_owned(), json!(links));
                    }
                    value
                })
                .collect(),
        ),
    );
    object.insert(
        "widgets_values".to_owned(),
        Value::Array(
            node.widgets
                .iter()
                .map(|widget| widget.value.clone())
                .collect(),
        ),
    );
    let envelope_unknown = node
        .source_fields
        .get(NATIVE_WIDGETS_FIELD)
        .cloned()
        .and_then(|value| serde_json::from_value::<NativeWidgetsEnvelope>(value).ok())
        .map(|envelope| envelope.unknown)
        .unwrap_or_default();
    object.insert(
        NATIVE_WIDGETS_FIELD.to_owned(),
        serde_json::to_value(NativeWidgetsEnvelope {
            version: NATIVE_WIDGETS_VERSION,
            widgets: node.widgets.clone(),
            unknown: envelope_unknown,
        })
        .map_err(|error| GraphError::Serialization(error.to_string()))?,
    );
    object.insert(
        "mode".to_owned(),
        Value::from(match node.mode {
            GraphNodeMode::Always => 0,
            GraphNodeMode::OnEvent => 1,
            GraphNodeMode::Never => 2,
            GraphNodeMode::OnTrigger => 3,
            GraphNodeMode::Bypass => 4,
        }),
    );
    object.insert("order".to_owned(), Value::from(order as u64));
    if !object.get("properties").is_some_and(Value::is_object) {
        object.insert("properties".to_owned(), Value::Object(Map::new()));
    }
    let flags = object
        .entry("flags")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(flags) = flags.as_object_mut() {
        flags.insert("pinned".to_owned(), Value::Bool(node.pinned));
        flags.insert("collapsed".to_owned(), Value::Bool(node.collapsed));
    }
    if let Some(color) = &node.color {
        object.insert("color".to_owned(), Value::String(color.clone()));
    } else {
        object.remove("color");
    }
    if let Some(identifier) = &node.subgraph_definition {
        if format == Some(WorkflowFormat::Schema1) {
            object.remove("subgraph_id");
        } else {
            object.insert(
                "subgraph_id".to_owned(),
                serde_json::to_value(identifier)
                    .map_err(|error| GraphError::Serialization(error.to_string()))?,
            );
        }
    }
    if !node.quarantine.is_empty() {
        object.insert(
            "zed_unmapped".to_owned(),
            serde_json::to_value(&node.quarantine)
                .map_err(|error| GraphError::Serialization(error.to_string()))?,
        );
    }
    Ok(Value::Object(object))
}

fn serialize_port(port: &GraphPort) -> Value {
    let mut object = port
        .unknown
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Map<_, _>>();
    object.insert("name".to_owned(), Value::String(port.name.clone()));
    object.insert(
        "type".to_owned(),
        Value::String(port.port_type.display_name()),
    );
    if port.multiple {
        object.insert("multiple".to_owned(), Value::Bool(true));
    }
    if port.dynamic {
        object.insert("dynamic".to_owned(), Value::Bool(true));
    }
    Value::Object(object)
}

fn serialize_link(
    link: &GraphLink,
    format: Option<WorkflowFormat>,
    schema_one_identifiers: Option<&Schema1LevelIdentifiers>,
) -> Value {
    if matches!(format, Some(WorkflowFormat::Schema1)) {
        let mut object = link.source.as_object().cloned().unwrap_or_default();
        let identifier = schema_one_identifiers
            .and_then(|identifiers| identifiers.links.get(&link.identifier))
            .copied()
            .unwrap_or_else(|| match &link.identifier {
                GraphIdentifier::Integer(identifier) => *identifier,
                GraphIdentifier::String(_) => 0,
            });
        object.insert("id".to_owned(), Value::from(identifier));
        object.insert("origin_id".to_owned(), json!(link.origin_node));
        object.insert(
            "origin_slot".to_owned(),
            Value::from(link.origin_slot as u64),
        );
        object.insert("target_id".to_owned(), json!(link.target_node));
        object.insert(
            "target_slot".to_owned(),
            Value::from(link.target_slot as u64),
        );
        object.insert("type".to_owned(), Value::String(link.type_name.clone()));
        if let Some(parent) = &link.parent_reroute {
            if let Some(parent) =
                schema_one_identifiers.and_then(|identifiers| identifiers.reroutes.get(parent))
            {
                object.insert("parentId".to_owned(), Value::from(*parent));
            }
        } else {
            object.remove("parentId");
            object.remove("parent_id");
        }
        Value::Object(object)
    } else {
        let values = vec![
            schema_one_identifiers
                .and_then(|identifiers| identifiers.links.get(&link.identifier))
                .map_or_else(|| json!(link.identifier), |identifier| json!(identifier)),
            json!(link.origin_node),
            Value::from(link.origin_slot as u64),
            json!(link.target_node),
            Value::from(link.target_slot as u64),
            Value::String(link.type_name.clone()),
        ];
        Value::Array(values)
    }
}

fn serialize_group(
    group: &GraphGroup,
    _format: Option<WorkflowFormat>,
    schema_one_identifiers: Option<&Schema1LevelIdentifiers>,
) -> Value {
    let mut object = group.source_fields.clone();
    object.remove("nodes");
    if schema_one_identifiers.is_some() {
        if let Some(identifier) =
            schema_one_identifiers.and_then(|identifiers| identifiers.groups.get(&group.identifier))
        {
            object.insert("id".to_owned(), Value::from(*identifier));
        }
    } else {
        object.insert("id".to_owned(), json!(group.identifier));
    }
    object.insert("title".to_owned(), Value::String(group.title.clone()));
    object.insert(
        "bounding".to_owned(),
        json!([
            group.bounds.origin.x,
            group.bounds.origin.y,
            group.bounds.size.width,
            group.bounds.size.height
        ]),
    );
    object.insert("collapsed".to_owned(), Value::Bool(group.collapsed));
    object.insert("pinned".to_owned(), Value::Bool(group.pinned));
    if let Some(color) = &group.color {
        object.insert("color".to_owned(), Value::String(color.clone()));
    }
    Value::Object(object)
}

fn serialize_reroute(
    reroute: &GraphReroute,
    _format: Option<WorkflowFormat>,
    schema_one_identifiers: Option<&Schema1LevelIdentifiers>,
) -> Value {
    let mut object = reroute.source_fields.clone();
    if schema_one_identifiers.is_some() {
        if let Some(identifier) = schema_one_identifiers
            .and_then(|identifiers| identifiers.reroutes.get(&reroute.identifier))
        {
            object.insert("id".to_owned(), Value::from(*identifier));
        }
    } else {
        object.insert("id".to_owned(), json!(reroute.identifier));
    }
    object.insert(
        "pos".to_owned(),
        json!([reroute.position.x, reroute.position.y]),
    );
    if let Some(parent) = &reroute.parent {
        if schema_one_identifiers.is_some() {
            if let Some(parent) =
                schema_one_identifiers.and_then(|identifiers| identifiers.reroutes.get(parent))
            {
                object.insert("parentId".to_owned(), Value::from(*parent));
            }
            object.remove("parent_id");
        } else {
            object.insert("parent_id".to_owned(), json!(parent));
        }
    } else {
        object.remove("parent_id");
        object.remove("parentId");
    }
    if let Some(slot_type) = &reroute.floating_type {
        let mut floating = object
            .get("floating")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        floating.insert("slotType".to_owned(), Value::String(slot_type.clone()));
        object.insert("floating".to_owned(), Value::Object(floating));
    } else {
        object.remove("floating");
    }
    Value::Object(object)
}

fn subgraph_boundary_identifier(level: &GraphLevel, field: &str) -> Option<GraphIdentifier> {
    level
        .source_fields
        .get(field)
        .and_then(|boundary| boundary.get("id"))
        .and_then(GraphIdentifier::from_value)
}

fn normalize_json_identifier_map_keys(level: &mut GraphLevel) -> Result<(), GraphError> {
    for definition in level.definitions.values_mut() {
        normalize_json_identifier_map_keys(&mut definition.graph)?;
    }
    normalize_json_identifier_map(&mut level.nodes, |node| &node.identifier)?;
    normalize_json_identifier_map(&mut level.links, |link| &link.identifier)?;
    normalize_json_identifier_map(&mut level.groups, |group| &group.identifier)?;
    normalize_json_identifier_map(&mut level.reroutes, |reroute| &reroute.identifier)?;
    normalize_json_identifier_map(&mut level.definitions, |definition| &definition.identifier)
}

fn normalize_json_identifier_map<T>(
    map: &mut BTreeMap<GraphIdentifier, T>,
    identifier: impl Fn(&T) -> &GraphIdentifier,
) -> Result<(), GraphError> {
    let mut normalized = BTreeMap::new();
    for (key, value) in std::mem::take(map) {
        let value_identifier = identifier(&value).clone();
        let key_matches = key == value_identifier
            || matches!(
                (&key, &value_identifier),
                (GraphIdentifier::String(key), GraphIdentifier::Integer(identifier))
                    if key == &identifier.to_string()
            );
        if !key_matches {
            return Err(GraphError::Serialization(format!(
                "entity map key {key:?} does not match entity id {value_identifier:?}"
            )));
        }
        if normalized.insert(value_identifier.clone(), value).is_some() {
            return Err(GraphError::DuplicateEntity(value_identifier));
        }
    }
    *map = normalized;
    Ok(())
}

fn validate_level(level: &GraphLevel, depth: usize) -> Result<(), GraphError> {
    if depth > MAX_GRAPH_DEPTH {
        return Err(GraphError::TooDeep);
    }
    level.viewport.validate()?;
    for (identifier, node) in &level.nodes {
        if identifier != &node.identifier {
            return Err(GraphError::Serialization(format!(
                "node map key {identifier:?} does not match node id {:?}",
                node.identifier
            )));
        }
        node.position.validate()?;
        node.size.validate()?;
        for field in [
            "resizable",
            "collapsable",
            "collapsible",
            "clonable",
            "removable",
            "block_delete",
        ] {
            if node
                .source_fields
                .get(field)
                .is_some_and(|value| !value.is_boolean())
            {
                return Err(GraphError::InvalidNodeOperationFlag {
                    node: identifier.clone(),
                    field,
                });
            }
        }
        for widget in &node.widgets {
            widget.validate_schema()?;
        }
        if let Some((slot, _)) = node
            .inputs
            .iter()
            .enumerate()
            .find(|(slot, port)| port.dynamic && *slot + 1 != node.inputs.len())
        {
            return Err(GraphError::DynamicInputMustBeLast {
                node: identifier.clone(),
                slot,
            });
        }
        if let Some(definition) = &node.subgraph_definition
            && !level.definitions.contains_key(definition)
        {
            return Err(GraphError::UnknownSubgraph(definition.clone()));
        }
    }
    let input_boundary = subgraph_boundary_identifier(level, "inputNode");
    let output_boundary = subgraph_boundary_identifier(level, "outputNode");
    let mut input_connections = BTreeMap::<(GraphIdentifier, usize), usize>::new();
    for (identifier, link) in &level.links {
        if identifier != &link.identifier {
            return Err(GraphError::Serialization(format!(
                "link map key {identifier:?} does not match link id {:?}",
                link.identifier
            )));
        }
        let output_type = if let Some(node) = level.nodes.get(&link.origin_node) {
            node.outputs
                .get(link.origin_slot)
                .ok_or_else(|| GraphError::UnknownPort {
                    node: link.origin_node.clone(),
                    slot: link.origin_slot,
                })?
                .port_type
                .clone()
        } else if input_boundary.as_ref() == Some(&link.origin_node) {
            GraphPortType::from_name(&link.type_name)
        } else {
            return Err(GraphError::UnknownNode(link.origin_node.clone()));
        };
        let (input_type, input_multiple) = if let Some(node) = level.nodes.get(&link.target_node) {
            let input =
                node.inputs
                    .get(link.target_slot)
                    .ok_or_else(|| GraphError::UnknownPort {
                        node: link.target_node.clone(),
                        slot: link.target_slot,
                    })?;
            (input.port_type.clone(), input.multiple)
        } else if output_boundary.as_ref() == Some(&link.target_node) {
            (GraphPortType::from_name(&link.type_name), true)
        } else {
            return Err(GraphError::UnknownNode(link.target_node.clone()));
        };
        if !input_type.accepts(&output_type) {
            return Err(GraphError::IncompatiblePorts {
                output: output_type.display_name(),
                input: input_type.display_name(),
            });
        }
        if let Some(parent) = &link.parent_reroute
            && !level.reroutes.contains_key(parent)
        {
            return Err(GraphError::UnknownReroute(parent.clone()));
        }
        if level.nodes.contains_key(&link.target_node) {
            let count = input_connections
                .entry((link.target_node.clone(), link.target_slot))
                .or_default();
            *count = count.saturating_add(1);
            if *count > 1 && !input_multiple {
                return Err(GraphError::InputOccupied {
                    node: link.target_node.clone(),
                    slot: link.target_slot,
                });
            }
        }
    }
    for (identifier, group) in &level.groups {
        if identifier != &group.identifier {
            return Err(GraphError::Serialization(
                "group map key does not match group id".to_owned(),
            ));
        }
        group.bounds.validate()?;
        let expected_membership = derived_group_membership(level, group.bounds);
        if group.node_ids != expected_membership {
            return Err(GraphError::InvalidGroupMembership(identifier.clone()));
        }
    }
    for (identifier, reroute) in &level.reroutes {
        if identifier != &reroute.identifier {
            return Err(GraphError::Serialization(
                "reroute map key does not match reroute id".to_owned(),
            ));
        }
        reroute.position.validate()?;
        validate_floating_reroute(reroute)?;
        if let Some(parent) = &reroute.parent
            && !level.reroutes.contains_key(parent)
        {
            return Err(GraphError::UnknownReroute(parent.clone()));
        }
        let mut visited = BTreeSet::from([identifier.clone()]);
        let mut parent = reroute.parent.as_ref();
        while let Some(parent_identifier) = parent {
            if !visited.insert(parent_identifier.clone()) {
                return Err(GraphError::RerouteCycle(identifier.clone()));
            }
            parent = level
                .reroutes
                .get(parent_identifier)
                .and_then(|parent| parent.parent.as_ref());
        }
    }
    validate_selection(level, &level.selection)?;
    for (identifier, definition) in &level.definitions {
        if identifier != &definition.identifier {
            return Err(GraphError::Serialization(
                "subgraph map key does not match definition id".to_owned(),
            ));
        }
        validate_level(&definition.graph, depth + 1)?;
        let mut exposed_widget_identifiers = BTreeSet::new();
        for exposure in &definition.exposed_widgets {
            if !exposed_widget_identifiers.insert(exposure.identifier.clone()) {
                return Err(GraphError::DuplicateEntity(GraphIdentifier::from(
                    exposure.identifier.as_str(),
                )));
            }
            let node = definition
                .graph
                .nodes
                .get(&exposure.internal_node)
                .ok_or_else(|| GraphError::UnknownNode(exposure.internal_node.clone()))?;
            if node
                .inputs
                .iter()
                .any(|input| input.name == exposure.internal_widget && input.dynamic)
            {
                return Err(GraphError::UnsupportedDynamicSubgraphInput {
                    node: exposure.internal_node.clone(),
                    port: exposure.internal_widget.clone(),
                });
            }
            let widget = node
                .widgets
                .iter()
                .find(|widget| widget.identifier == exposure.internal_widget)
                .ok_or_else(|| GraphError::UnknownWidget(exposure.internal_widget.clone()))?;
            if exposure.widget.identifier != exposure.identifier
                || exposure.widget.kind != widget.kind
            {
                return Err(GraphError::InvalidWidgetSchema {
                    widget: exposure.identifier.clone(),
                    reason: "exposed widget does not match its internal widget schema".to_owned(),
                });
            }
            exposure.widget.validate_schema()?;
        }
        for port in &definition.inputs {
            let Some(internal_node) = &port.internal_node else {
                continue;
            };
            let node = definition
                .graph
                .nodes
                .get(internal_node)
                .ok_or_else(|| GraphError::UnknownNode(internal_node.clone()))?;
            let internal =
                node.inputs
                    .get(port.internal_slot)
                    .ok_or_else(|| GraphError::UnknownPort {
                        node: internal_node.clone(),
                        slot: port.internal_slot,
                    })?;
            if !internal.port_type.accepts(&port.port_type)
                && !port.port_type.accepts(&internal.port_type)
            {
                return Err(GraphError::IncompatiblePorts {
                    output: port.port_type.display_name(),
                    input: internal.port_type.display_name(),
                });
            }
        }
        for port in &definition.outputs {
            let Some(internal_node) = &port.internal_node else {
                continue;
            };
            let node = definition
                .graph
                .nodes
                .get(internal_node)
                .ok_or_else(|| GraphError::UnknownNode(internal_node.clone()))?;
            let internal =
                node.outputs
                    .get(port.internal_slot)
                    .ok_or_else(|| GraphError::UnknownPort {
                        node: internal_node.clone(),
                        slot: port.internal_slot,
                    })?;
            if !port.port_type.accepts(&internal.port_type)
                && !internal.port_type.accepts(&port.port_type)
            {
                return Err(GraphError::IncompatiblePorts {
                    output: internal.port_type.display_name(),
                    input: port.port_type.display_name(),
                });
            }
        }
    }
    Ok(())
}

fn validate_json_depth(bytes: &[u8], maximum: usize) -> Result<(), GraphError> {
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
                depth = depth.checked_add(1).ok_or(GraphError::TooDeep)?;
                if depth > maximum {
                    return Err(GraphError::TooDeep);
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use sim_nodes::{NodeDescriptor, PortDescriptor};
    use std::{error::Error, fs, path::PathBuf};

    fn node(
        identifier: &str,
        input: Option<GraphPortType>,
        output: Option<GraphPortType>,
    ) -> GraphNode {
        let mut node = GraphNode::new(
            GraphIdentifier::from(identifier),
            "Fixture",
            identifier,
            GraphPoint::ZERO,
        );
        if let Some(input) = input {
            node.inputs.push(GraphPort::new("input", input));
        }
        if let Some(output) = output {
            node.outputs.push(GraphPort::new("output", output));
        }
        node
    }

    fn link(identifier: &str, origin: &str, target: &str) -> GraphLink {
        GraphLink {
            identifier: GraphIdentifier::from(identifier),
            origin_node: GraphIdentifier::from(origin),
            origin_slot: 0,
            target_node: GraphIdentifier::from(target),
            target_slot: 0,
            type_name: String::new(),
            parent_reroute: None,
            source: Value::Null,
        }
    }

    fn fixture_engine() -> Result<GraphCommandEngine, GraphError> {
        let mut document = GraphDocument::default();
        document.document_identity = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_0001);
        document.profile_identity =
            Some(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_0002));
        let mut output = node(
            "source",
            None,
            Some(GraphPortType::Concrete("IMAGE".to_owned())),
        );
        output.position = GraphPoint { x: -300.0, y: 0.0 };
        let mut input = node(
            "target",
            Some(GraphPortType::Union(BTreeSet::from([
                "IMAGE".to_owned(),
                "MASK".to_owned(),
            ]))),
            None,
        );
        input.position = GraphPoint { x: 900.0, y: 0.0 };
        document
            .root
            .nodes
            .insert(output.identifier.clone(), output);
        document.root.nodes.insert(input.identifier.clone(), input);
        GraphCommandEngine::new(document)
    }

    fn simple_definition(identifier: &str) -> SubgraphDefinition {
        let internal_identifier_text = format!("{identifier}-internal");
        let internal_identifier = GraphIdentifier::from(internal_identifier_text.as_str());
        let internal = node(internal_identifier.text().as_str(), None, None);
        SubgraphDefinition {
            identifier: GraphIdentifier::from(identifier),
            name: format!("{identifier} definition"),
            graph: Box::new(GraphLevel {
                nodes: BTreeMap::from([(internal.identifier.clone(), internal)]),
                ..GraphLevel::default()
            }),
            inputs: Vec::new(),
            outputs: Vec::new(),
            published: false,
            description: String::new(),
            search_aliases: Vec::new(),
            exposed_widgets: Vec::new(),
            graph_inline: false,
            unknown: BTreeMap::new(),
        }
    }

    fn integer_widget(identifier: &str, value: i64) -> GraphWidget {
        GraphWidget {
            identifier: identifier.to_owned(),
            kind: GraphWidgetKind::Integer {
                minimum: 1,
                maximum: 100,
                step: 2,
            },
            value: Value::from(value),
            prompt_value: Value::from(value),
            validation: WidgetValidation::Valid,
            converted_to_input: false,
            visible: true,
            unknown: BTreeMap::new(),
        }
    }

    #[test]
    fn numeric_link_snapshot_preserves_typed_ids_and_both_history_stacks()
    -> Result<(), Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        let mut connection = link("numeric", "source", "target");
        connection.identifier = GraphIdentifier::Integer(1);
        engine.apply(GraphCommand::Connect {
            link: connection,
            replace_existing: false,
        })?;
        engine.apply(GraphCommand::RenameNode {
            identifier: GraphIdentifier::from("source"),
            title: "Renamed source".into(),
        })?;
        assert!(engine.undo());
        let encoded = engine.encode()?;
        let mut restored = GraphCommandEngine::decode(&encoded)?;
        assert_eq!(restored, engine);
        assert!(
            restored
                .document
                .root
                .links
                .contains_key(&GraphIdentifier::Integer(1))
        );
        assert!(restored.redo());
        assert_eq!(
            restored.document.root.nodes[&GraphIdentifier::from("source")].title,
            "Renamed source"
        );
        assert!(restored.undo());
        assert!(restored.undo());
        assert!(restored.document.root.links.is_empty());
        assert!(restored.redo());
        assert!(
            restored
                .document
                .root
                .links
                .contains_key(&GraphIdentifier::Integer(1))
        );

        let mut corrupted: Value = serde_json::from_slice(&encoded)?;
        let links = corrupted["document"]["root"]["links"]
            .as_object_mut()
            .ok_or("link map")?;
        let value = links.remove("1").ok_or("numeric link")?;
        links.insert("2".into(), value);
        assert!(matches!(
            GraphCommandEngine::decode(&serde_json::to_vec(&corrupted)?),
            Err(GraphError::Serialization(_))
        ));
        Ok(())
    }

    #[test]
    fn incompatible_link_is_atomic_and_valid_link_undoes() -> Result<(), Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        engine.apply(GraphCommand::Connect {
            link: link("link", "source", "target"),
            replace_existing: false,
        })?;
        assert_eq!(engine.document.root.links.len(), 1);
        assert!(engine.undo());
        assert!(engine.document.root.links.is_empty());
        assert!(engine.redo());
        let before = engine.document.clone();
        let mut bad = node(
            "bad",
            Some(GraphPortType::Concrete("AUDIO".to_owned())),
            None,
        );
        bad.position.x = 10.0;
        engine.apply(GraphCommand::AddNode {
            node: bad,
            source: NodeCreationSource::Library,
        })?;
        let result = engine.apply(GraphCommand::Connect {
            link: link("bad-link", "source", "bad"),
            replace_existing: false,
        });
        assert!(matches!(result, Err(GraphError::IncompatiblePorts { .. })));
        assert_eq!(engine.document.root.links.len(), before.root.links.len());
        Ok(())
    }

    #[test]
    fn widget_group_reroute_and_subgraph_commands_round_trip() -> Result<(), Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        let source_id = GraphIdentifier::from("source");
        engine
            .document
            .root
            .nodes
            .get_mut(&source_id)
            .ok_or("source")?
            .widgets
            .push(GraphWidget {
                identifier: "steps".to_owned(),
                kind: GraphWidgetKind::Integer {
                    minimum: 1,
                    maximum: 100,
                    step: 2,
                },
                value: Value::from(1),
                prompt_value: Value::from(1),
                validation: WidgetValidation::Valid,
                converted_to_input: false,
                visible: true,
                unknown: BTreeMap::new(),
            });
        engine.apply(GraphCommand::SetWidget {
            node: source_id.clone(),
            widget: "steps".to_owned(),
            value: Value::from(18),
        })?;
        assert_eq!(engine.document.root.nodes[&source_id].widgets[0].value, 17);
        engine.apply(GraphCommand::SetSelection {
            selection: GraphSelection {
                nodes: BTreeSet::from([source_id.clone(), GraphIdentifier::from("target")]),
                ..GraphSelection::default()
            },
            mode: SelectionMode::Replace,
        })?;
        let group_id = GraphIdentifier::from("group");
        engine.apply(GraphCommand::CreateGroup {
            group: GraphGroup {
                identifier: group_id,
                title: "Fixture group".to_owned(),
                bounds: GraphRect {
                    origin: GraphPoint::ZERO,
                    size: GraphSize {
                        width: 400.0,
                        height: 300.0,
                    },
                },
                node_ids: engine.document.root.selection.nodes.clone(),
                collapsed: false,
                pinned: false,
                color: None,
                source_fields: Map::new(),
            },
        })?;
        engine.apply(GraphCommand::AddReroute {
            reroute: GraphReroute {
                identifier: GraphIdentifier::from("reroute"),
                position: GraphPoint { x: 50.0, y: 50.0 },
                parent: None,
                floating_type: Some("input".to_owned()),
                source_fields: Map::new(),
            },
        })?;
        engine.apply(GraphCommand::ConvertSelectionToSubgraph {
            definition_identifier: GraphIdentifier::from("definition"),
            instance_identifier: GraphIdentifier::from("instance"),
            name: "Fixture subgraph".to_owned(),
        })?;
        assert!(
            engine
                .document
                .root
                .nodes
                .contains_key(&GraphIdentifier::from("instance"))
        );
        let bytes = engine.encode()?;
        let restored = GraphCommandEngine::decode(&bytes)?;
        assert_eq!(restored.document, engine.document);
        Ok(())
    }

    #[test]
    fn clipboard_rejects_hostile_shape_and_paste_is_one_transaction() -> Result<(), Box<dyn Error>>
    {
        let mut engine = fixture_engine()?;
        engine.apply(GraphCommand::SelectAll)?;
        let clipboard = GraphClipboard::copy(&engine.document)?;
        let bytes = clipboard.encode()?;
        let clipboard = GraphClipboard::decode(&bytes)?;
        let command =
            clipboard.paste_command(&mut engine.document, GraphPoint { x: 20.0, y: 30.0 })?;
        engine.apply(command)?;
        assert_eq!(engine.document.root.nodes.len(), 4);
        assert!(engine.undo());
        assert_eq!(engine.document.root.nodes.len(), 2);
        let deeply_nested = format!(
            "{}0{}",
            "[".repeat(MAX_GRAPH_DEPTH + 1),
            "]".repeat(MAX_GRAPH_DEPTH + 1)
        );
        assert_eq!(
            GraphClipboard::decode(deeply_nested.as_bytes()),
            Err(GraphError::TooDeep)
        );
        Ok(())
    }

    fn exercise_large_graph() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut document = GraphDocument::default();
        document.document_identity = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_0016);
        for index in 0..2_000 {
            let identifier = GraphIdentifier::String(format!("node-{index:04}"));
            let mut node = node(identifier.text().as_str(), None, None);
            node.position = GraphPoint {
                x: (index % 100) as f32 * 10.0,
                y: (index / 100) as f32 * 10.0,
            };
            document.root.nodes.insert(identifier, node);
        }
        let mut engine = GraphCommandEngine::new(document)?;
        engine.apply(GraphCommand::SelectAll)?;
        engine.apply(GraphCommand::LayoutSelection {
            operation: LayoutOperation::ArrangeGrid,
            spacing: 4.0,
        })?;
        let first = engine.document.to_workflow_bytes()?;
        let second = engine.document.to_workflow_bytes()?;
        assert_eq!(first, second);
        Ok(first)
    }

    #[test]
    fn large_graph_selection_layout_and_serialization_are_deterministic()
    -> Result<(), Box<dyn Error>> {
        exercise_large_graph()?;
        Ok(())
    }

    #[test]
    fn catalog_graph_action_ids_are_unique_and_complete() {
        let identifiers = CatalogGraphAction::ALL
            .iter()
            .map(|action| action.command_id())
            .collect::<BTreeSet<_>>();
        assert_eq!(identifiers.len(), CatalogGraphAction::ALL.len());
        assert!(identifiers.contains("Sim.Canvas.PasteFromClipboardWithConnect"));
        assert!(identifiers.contains("Sim.Graph.ConvertToSubgraph"));
    }

    fn exercise_audited_graph_failures() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        let source = GraphIdentifier::from("source");
        let target = GraphIdentifier::from("target");
        engine.apply(GraphCommand::Connect {
            link: link("selected-link", "source", "target"),
            replace_existing: false,
        })?;
        engine.document.root.selection.links =
            BTreeSet::from([GraphIdentifier::from("selected-link")]);
        engine.apply(GraphCommand::Connect {
            link: link("replacement", "source", "target"),
            replace_existing: true,
        })?;
        assert!(engine.document.root.selection.links.is_empty());

        let group = GraphIdentifier::from("moving-group");
        engine.document.root.groups.insert(
            group.clone(),
            GraphGroup {
                identifier: group.clone(),
                title: "Moving group".to_owned(),
                bounds: GraphRect {
                    origin: GraphPoint::ZERO,
                    size: GraphSize {
                        width: 2_000.0,
                        height: 300.0,
                    },
                },
                node_ids: BTreeSet::from([source.clone(), target]),
                collapsed: false,
                pinned: false,
                color: None,
                source_fields: Map::new(),
            },
        );
        engine.document.root.selection = GraphSelection {
            groups: BTreeSet::from([group]),
            ..GraphSelection::default()
        };
        let source_before = engine.document.root.nodes[&source].position;
        engine.apply(GraphCommand::MoveSelection {
            delta: GraphPoint { x: 25.0, y: 10.0 },
            snap: None,
        })?;
        assert_eq!(
            engine.document.root.nodes[&source].position,
            source_before.translated(GraphPoint { x: 25.0, y: 10.0 })
        );

        engine.apply(GraphCommand::SelectInRect {
            bounds: GraphRect {
                origin: GraphPoint { x: 200.0, y: 30.0 },
                size: GraphSize {
                    width: 300.0,
                    height: 80.0,
                },
            },
            mode: SelectionMode::Replace,
        })?;
        assert!(
            engine
                .document
                .root
                .selection
                .links
                .contains(&GraphIdentifier::from("replacement"))
        );

        let mut malformed_widget_node = node("malformed-widget", None, None);
        malformed_widget_node.widgets.push(GraphWidget {
            identifier: "range".to_owned(),
            kind: GraphWidgetKind::Integer {
                minimum: 10,
                maximum: 1,
                step: 1,
            },
            value: Value::from(5),
            prompt_value: Value::from(5),
            validation: WidgetValidation::Valid,
            converted_to_input: false,
            visible: true,
            unknown: BTreeMap::new(),
        });
        assert!(matches!(
            engine.apply(GraphCommand::AddNode {
                node: malformed_widget_node,
                source: NodeCreationSource::Library,
            }),
            Err(GraphError::InvalidWidgetSchema { .. })
        ));
        assert!(
            !engine
                .document
                .root
                .nodes
                .contains_key(&GraphIdentifier::from("malformed-widget"))
        );
        let before_invalid_reroute = engine.document.clone();
        assert!(matches!(
            engine.apply(GraphCommand::AddReroute {
                reroute: GraphReroute {
                    identifier: GraphIdentifier::from("invalid-floating-reroute"),
                    position: GraphPoint::ZERO,
                    parent: None,
                    floating_type: Some("IMAGE".to_owned()),
                    source_fields: Map::new(),
                },
            }),
            Err(GraphError::Serialization(_))
        ));
        assert_eq!(engine.document, before_invalid_reroute);
        Ok(engine.encode()?)
    }

    #[test]
    fn audited_graph_failures_are_atomic_and_schema_safe() -> Result<(), Box<dyn Error>> {
        exercise_audited_graph_failures()?;
        Ok(())
    }

    fn exercise_same_name_type_reconciliation() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        engine.apply(GraphCommand::Connect {
            link: link("typed", "source", "target"),
            replace_existing: false,
        })?;
        let changed_inputs = vec![GraphPort::new(
            "input",
            GraphPortType::Concrete("AUDIO".to_owned()),
        )];
        let before = engine.document.clone();
        assert!(matches!(
            engine.apply(GraphCommand::ReconcileNode {
                identifier: GraphIdentifier::from("target"),
                inputs: changed_inputs.clone(),
                outputs: Vec::new(),
                widgets: Vec::new(),
                confirm_discard: false,
            }),
            Err(GraphError::ReconciliationRequiresConfirmation { .. })
        ));
        assert_eq!(engine.document, before);
        engine.apply(GraphCommand::ReconcileNode {
            identifier: GraphIdentifier::from("target"),
            inputs: changed_inputs,
            outputs: Vec::new(),
            widgets: Vec::new(),
            confirm_discard: true,
        })?;
        assert!(engine.document.root.links.is_empty());
        assert!(
            engine.document.root.nodes[&GraphIdentifier::from("target")]
                .quarantine
                .contains_key("unmapped_links")
        );
        Ok(engine.encode()?)
    }

    #[test]
    fn reconciliation_quarantines_same_name_type_changes() -> Result<(), Box<dyn Error>> {
        exercise_same_name_type_reconciliation()?;
        Ok(())
    }

    fn exercise_api_prompt_import() -> Result<Vec<u8>, Box<dyn Error>> {
        let prompt = br#"{
            "1":{"class_type":"LoadImage","inputs":{"image":"fixture.png"}},
            "2":{"class_type":"PreviewImage","inputs":{"images":["1",0]}}
        }"#;
        let document = GraphDocument::from_workflow_bytes(prompt)?;
        assert_eq!(document.root.nodes.len(), 2);
        assert_eq!(document.root.links.len(), 1);
        assert_eq!(
            document.root.source_fields.get("zed_api_prompt"),
            Some(&serde_json::from_slice::<Value>(prompt)?)
        );
        let serialized = document.to_workflow_value()?;
        assert!(
            serialized["id"]
                .as_str()
                .is_some_and(|identifier| Uuid::parse_str(identifier).is_ok())
        );
        for counter in ["lastGroupId", "lastNodeId", "lastLinkId", "lastRerouteId"] {
            assert!(serialized["state"][counter].is_number());
        }
        assert!(
            serialized["links"]
                .as_array()
                .is_some_and(|links| links.iter().all(Value::is_object))
        );
        let bytes = serde_json::to_vec(&serialized)?;
        let reparsed = WorkflowFormatDocument::parse(&bytes)?;
        assert!(reparsed.validation_issues().is_empty());
        Ok(bytes)
    }

    #[test]
    fn api_prompt_import_and_schema_one_output_are_native_and_valid() -> Result<(), Box<dyn Error>>
    {
        exercise_api_prompt_import()?;
        Ok(())
    }

    fn exercise_recursive_schema_one() -> Result<Vec<u8>, Box<dyn Error>> {
        let source = br#"{
          "id":"8f7bd92e-0d48-542e-945b-079f03ec3d80",
          "revision":2,
          "version":1,
          "state":{"lastGroupId":0,"lastNodeId":0,"lastLinkId":0,"lastRerouteId":0},
          "nodes":[],"links":[],"groups":[],"reroutes":[],
          "definitions":{"subgraphs":[{
            "id":"1cb1644a-b9d7-5f3e-a907-15c5720f8729",
            "revision":3,"version":1,"name":"Boundary fixture",
            "state":{"lastGroupId":0,"lastNodeId":1,"lastLinkId":1,"lastRerouteId":0},
            "inputNode":{"id":-10,"bounding":[0,0,120,60]},
            "outputNode":{"id":-20,"bounding":[500,0,120,60]},
            "inputs":[{"id":"bf3be1d0-d92c-561a-a947-d371d9801587","name":"image","type":"IMAGE","linkIds":[1]}],
            "outputs":[],"widgets":[],
            "nodes":[{"id":1,"type":"PreviewImage","pos":[200,0],"size":[240,120],"flags":{},"order":0,"mode":0,"properties":{},"inputs":[{"name":"images","type":"IMAGE","link":1}],"outputs":[]}],
            "links":[{"id":1,"origin_id":-10,"origin_slot":0,"target_id":1,"target_slot":0,"type":"IMAGE"}],
            "groups":[],"reroutes":[],"config":{},"extra":{"future":"preserved"}
          }]},"config":{},"extra":{}
        }"#;
        let document = GraphDocument::from_workflow_bytes(source)?;
        assert_eq!(
            document.document_identity,
            Uuid::parse_str("8f7bd92e-0d48-542e-945b-079f03ec3d80")?
        );
        let definition = document
            .root
            .definitions
            .get(&GraphIdentifier::from(
                "1cb1644a-b9d7-5f3e-a907-15c5720f8729",
            ))
            .ok_or("subgraph definition")?;
        assert_eq!(
            definition.inputs[0].internal_node,
            Some(GraphIdentifier::Integer(1))
        );
        assert_eq!(definition.inputs[0].internal_slot, 0);
        let bytes = document.to_workflow_bytes()?;
        let reparsed = WorkflowFormatDocument::parse(&bytes)?;
        assert!(reparsed.validation_issues().is_empty());
        assert_eq!(
            reparsed.value()["definitions"]["subgraphs"][0]["extra"]["future"],
            "preserved"
        );
        let restored = GraphDocument::from_workflow(&reparsed)?;
        assert_eq!(restored.to_workflow_bytes()?, bytes);
        Ok(bytes)
    }

    #[test]
    fn recursive_schema_one_subgraph_boundaries_round_trip() -> Result<(), Box<dyn Error>> {
        exercise_recursive_schema_one()?;
        Ok(())
    }

    fn exercise_schema_zero_four_reroutes() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut document = fixture_engine()?.document;
        document.workflow_format = Some(WorkflowFormat::Schema04);
        let reroute_identifier = GraphIdentifier::from("legacy-reroute");
        document.root.reroutes.insert(
            reroute_identifier.clone(),
            GraphReroute {
                identifier: reroute_identifier.clone(),
                position: GraphPoint { x: 400.0, y: 60.0 },
                parent: None,
                floating_type: Some("input".to_owned()),
                source_fields: Map::new(),
            },
        );
        let mut routed_link = link("legacy-link", "source", "target");
        routed_link.parent_reroute = Some(reroute_identifier);
        document
            .root
            .links
            .insert(routed_link.identifier.clone(), routed_link);
        let serialized = document.to_workflow_value()?;
        assert_eq!(serialized["version"], 0.4);
        assert_eq!(serialized["links"][0].as_array().map(Vec::len), Some(6));
        assert!(serialized["links"][0][0].is_number());
        assert!(serialized["extra"]["reroutes"][0]["id"].is_number());
        assert_eq!(
            serialized["extra"]["reroutes"][0]["floating"]["slotType"],
            "input"
        );
        assert!(serialized["extra"]["reroutes"][0].get("type").is_none());
        assert!(serialized["extra"]["linkExtensions"][0]["id"].is_number());
        assert!(serialized["extra"]["linkExtensions"][0]["parentId"].is_number());
        let bytes = serde_json::to_vec(&serialized)?;
        let workflow = WorkflowFormatDocument::parse(&bytes)?;
        assert!(workflow.validation_issues().is_empty());
        let restored = GraphDocument::from_workflow(&workflow)?;
        assert_eq!(restored.root.reroutes.len(), 1);
        assert_eq!(
            restored
                .root
                .reroutes
                .values()
                .next()
                .and_then(|reroute| reroute.floating_type.as_deref()),
            Some("input")
        );
        assert_eq!(
            restored
                .root
                .links
                .values()
                .next()
                .and_then(|link| link.parent_reroute.clone()),
            restored.root.reroutes.keys().next().cloned()
        );
        Ok(bytes)
    }

    #[test]
    fn schema_zero_four_reroutes_use_link_extensions() -> Result<(), Box<dyn Error>> {
        exercise_schema_zero_four_reroutes()?;
        Ok(())
    }

    fn exercise_subgraph_conversion_and_unpack() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        let image = GraphPortType::Concrete("IMAGE".to_owned());
        let mut first = node("middle-a", Some(image.clone()), Some(image.clone()));
        first.position = GraphPoint { x: 100.0, y: 200.0 };
        let mut second = node("middle-b", Some(image.clone()), Some(image));
        second.position = GraphPoint { x: 430.0, y: 260.0 };
        second.subgraph_definition = Some(GraphIdentifier::from("nested-definition"));
        engine
            .document
            .root
            .nodes
            .insert(first.identifier.clone(), first);
        engine
            .document
            .root
            .nodes
            .insert(second.identifier.clone(), second);
        let mut nested_definition = simple_definition("nested-definition");
        nested_definition
            .unknown
            .insert("vendor-extension".to_owned(), json!({"preserved": true}));
        engine.document.root.definitions.insert(
            nested_definition.identifier.clone(),
            nested_definition.clone(),
        );
        engine.document.root.reroutes.insert(
            GraphIdentifier::from("outer-reroute"),
            GraphReroute {
                identifier: GraphIdentifier::from("outer-reroute"),
                position: GraphPoint { x: 20.0, y: 100.0 },
                parent: None,
                floating_type: Some("input".to_owned()),
                source_fields: Map::new(),
            },
        );
        engine.document.root.reroutes.insert(
            GraphIdentifier::from("inner-parent"),
            GraphReroute {
                identifier: GraphIdentifier::from("inner-parent"),
                position: GraphPoint { x: 260.0, y: 320.0 },
                parent: None,
                floating_type: Some("input".to_owned()),
                source_fields: Map::new(),
            },
        );
        engine.document.root.reroutes.insert(
            GraphIdentifier::from("inner-child"),
            GraphReroute {
                identifier: GraphIdentifier::from("inner-child"),
                position: GraphPoint { x: 360.0, y: 320.0 },
                parent: Some(GraphIdentifier::from("inner-parent")),
                floating_type: Some("input".to_owned()),
                source_fields: Map::new(),
            },
        );
        let mut inbound = link("inbound", "source", "middle-a");
        inbound.parent_reroute = Some(GraphIdentifier::from("outer-reroute"));
        engine
            .document
            .root
            .links
            .insert(inbound.identifier.clone(), inbound);
        let mut internal = link("internal", "middle-a", "middle-b");
        internal.parent_reroute = Some(GraphIdentifier::from("inner-child"));
        engine
            .document
            .root
            .links
            .insert(internal.identifier.clone(), internal);
        let outbound = link("outbound", "middle-b", "target");
        engine
            .document
            .root
            .links
            .insert(outbound.identifier.clone(), outbound);
        engine.document.root.groups.insert(
            GraphIdentifier::from("inner-group"),
            GraphGroup {
                identifier: GraphIdentifier::from("inner-group"),
                title: "Selected processing".to_owned(),
                bounds: GraphRect {
                    origin: GraphPoint { x: 80.0, y: 180.0 },
                    size: GraphSize {
                        width: 620.0,
                        height: 240.0,
                    },
                },
                node_ids: BTreeSet::from([
                    GraphIdentifier::from("middle-a"),
                    GraphIdentifier::from("middle-b"),
                ]),
                collapsed: false,
                pinned: false,
                color: Some("#345678".to_owned()),
                source_fields: Map::new(),
            },
        );
        engine.document.root.selection = GraphSelection {
            nodes: BTreeSet::from([
                GraphIdentifier::from("middle-a"),
                GraphIdentifier::from("middle-b"),
            ]),
            reroutes: BTreeSet::from([GraphIdentifier::from("inner-child")]),
            ..GraphSelection::default()
        };
        engine.document.validate()?;

        engine.apply(GraphCommand::ConvertSelectionToSubgraph {
            definition_identifier: GraphIdentifier::from("converted-definition"),
            instance_identifier: GraphIdentifier::from("converted-instance"),
            name: "Converted processing".to_owned(),
        })?;
        let definition = engine
            .document
            .root
            .definitions
            .get(&GraphIdentifier::from("converted-definition"))
            .ok_or("converted definition")?;
        assert_eq!(definition.graph.nodes.len(), 2);
        assert_eq!(definition.graph.links.len(), 1);
        assert_eq!(definition.graph.groups.len(), 1);
        assert_eq!(definition.graph.reroutes.len(), 2);
        assert!(
            definition
                .graph
                .definitions
                .contains_key(&GraphIdentifier::from("nested-definition"))
        );
        assert_eq!(
            definition.graph.nodes[&GraphIdentifier::from("middle-a")].position,
            GraphPoint::ZERO
        );
        assert_eq!(
            definition.graph.groups[&GraphIdentifier::from("inner-group")]
                .bounds
                .origin,
            GraphPoint { x: -20.0, y: -20.0 }
        );
        assert_eq!(
            engine.document.root.links[&GraphIdentifier::from("inbound")].target_node,
            GraphIdentifier::from("converted-instance")
        );
        assert_eq!(
            engine.document.root.links[&GraphIdentifier::from("outbound")].origin_node,
            GraphIdentifier::from("converted-instance")
        );
        assert!(
            engine
                .document
                .root
                .reroutes
                .contains_key(&GraphIdentifier::from("outer-reroute"))
        );
        engine.document.validate()?;

        engine.apply(GraphCommand::UnpackSubgraph {
            instance_identifier: GraphIdentifier::from("converted-instance"),
        })?;
        engine.document.validate()?;
        assert!(
            !engine
                .document
                .root
                .nodes
                .contains_key(&GraphIdentifier::from("converted-instance"))
        );
        let unpacked_titles = engine
            .document
            .root
            .selection
            .nodes
            .iter()
            .filter_map(|identifier| engine.document.root.nodes.get(identifier))
            .map(|node| (node.title.clone(), node.position))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            unpacked_titles.get("middle-a"),
            Some(&GraphPoint { x: 100.0, y: 200.0 })
        );
        assert_eq!(
            unpacked_titles.get("middle-b"),
            Some(&GraphPoint { x: 430.0, y: 260.0 })
        );
        let inbound = &engine.document.root.links[&GraphIdentifier::from("inbound")];
        let outbound = &engine.document.root.links[&GraphIdentifier::from("outbound")];
        assert_ne!(
            inbound.target_node,
            GraphIdentifier::from("converted-instance")
        );
        assert_ne!(
            outbound.origin_node,
            GraphIdentifier::from("converted-instance")
        );
        assert_eq!(
            inbound.parent_reroute,
            Some(GraphIdentifier::from("outer-reroute"))
        );
        assert_eq!(engine.document.root.groups.len(), 1);
        assert_eq!(engine.document.root.reroutes.len(), 3);
        let bytes = engine.document.to_workflow_bytes()?;
        assert_eq!(bytes, engine.document.to_workflow_bytes()?);
        Ok(bytes)
    }

    fn exercise_clipboard_remapping() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        engine.document.next_identifier = 1;
        let mut occupied = node("zed-1", None, None);
        occupied.position.y = 500.0;
        engine
            .document
            .root
            .nodes
            .insert(occupied.identifier.clone(), occupied);
        engine.document.root.definitions.insert(
            GraphIdentifier::from("clipboard-definition"),
            simple_definition("clipboard-definition"),
        );
        engine
            .document
            .root
            .nodes
            .get_mut(&GraphIdentifier::from("target"))
            .ok_or("target")?
            .subgraph_definition = Some(GraphIdentifier::from("clipboard-definition"));
        engine.document.root.links.insert(
            GraphIdentifier::from("copy-link"),
            link("copy-link", "source", "target"),
        );
        engine.document.root.reroutes.insert(
            GraphIdentifier::from("copy-parent"),
            GraphReroute {
                identifier: GraphIdentifier::from("copy-parent"),
                position: GraphPoint { x: 0.0, y: 40.0 },
                parent: None,
                floating_type: Some("input".to_owned()),
                source_fields: Map::new(),
            },
        );
        engine.document.root.reroutes.insert(
            GraphIdentifier::from("copy-child"),
            GraphReroute {
                identifier: GraphIdentifier::from("copy-child"),
                position: GraphPoint { x: 100.0, y: 40.0 },
                parent: Some(GraphIdentifier::from("copy-parent")),
                floating_type: Some("input".to_owned()),
                source_fields: Map::new(),
            },
        );
        engine
            .document
            .root
            .links
            .get_mut(&GraphIdentifier::from("copy-link"))
            .ok_or("copy link")?
            .parent_reroute = Some(GraphIdentifier::from("copy-child"));
        engine.document.root.groups.insert(
            GraphIdentifier::from("copy-group"),
            GraphGroup {
                identifier: GraphIdentifier::from("copy-group"),
                title: "Clipboard group".to_owned(),
                bounds: GraphRect {
                    origin: GraphPoint {
                        x: -340.0,
                        y: -40.0,
                    },
                    size: GraphSize {
                        width: 1_500.0,
                        height: 240.0,
                    },
                },
                node_ids: BTreeSet::from([
                    GraphIdentifier::from("source"),
                    GraphIdentifier::from("target"),
                ]),
                collapsed: false,
                pinned: false,
                color: None,
                source_fields: Map::new(),
            },
        );
        engine.document.root.selection.groups =
            BTreeSet::from([GraphIdentifier::from("copy-group")]);
        engine.document.validate()?;
        let clipboard = GraphClipboard::copy(&engine.document)?;
        assert_eq!(clipboard.nodes.len(), 2);
        assert_eq!(clipboard.links.len(), 1);
        assert_eq!(clipboard.groups.len(), 1);
        assert_eq!(clipboard.reroutes.len(), 2);
        assert_eq!(clipboard.definitions.len(), 1);
        let clipboard_bytes = clipboard.encode()?;
        let clipboard = GraphClipboard::decode(&clipboard_bytes)?;
        let counter_before = engine.document.next_identifier;
        let command = clipboard.paste_command(&engine.document, GraphPoint { x: 25.0, y: 35.0 })?;
        assert_eq!(engine.document.next_identifier, counter_before);
        engine.apply(command)?;
        assert_eq!(engine.document.root.nodes.len(), 5);
        assert_eq!(engine.document.root.selection.nodes.len(), 2);
        assert_eq!(engine.document.root.selection.groups.len(), 1);
        assert_eq!(engine.document.root.selection.reroutes.len(), 2);
        assert!(
            !engine
                .document
                .root
                .selection
                .nodes
                .contains(&GraphIdentifier::from("zed-1"))
        );
        engine.document.validate()?;
        assert!(engine.undo());
        assert_eq!(engine.document.root.nodes.len(), 3);
        assert!(engine.redo());
        engine.document.validate()?;
        Ok(clipboard_bytes)
    }

    fn exercise_reconciliation_and_history() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        let image = GraphPortType::Concrete("IMAGE".to_owned());
        let mut refresh = node("refresh", Some(image.clone()), Some(image.clone()));
        refresh
            .inputs
            .insert(0, GraphPort::new("removed", image.clone()));
        refresh.inputs[1].name = "retained-input".to_owned();
        refresh
            .outputs
            .insert(0, GraphPort::new("removed", image.clone()));
        refresh.outputs[1].name = "retained-output".to_owned();
        refresh.widgets = vec![integer_widget("steps", 18), integer_widget("removed", 7)];
        engine
            .document
            .root
            .nodes
            .insert(refresh.identifier.clone(), refresh);
        let mut inbound = link("refresh-in", "source", "refresh");
        inbound.target_slot = 1;
        engine
            .document
            .root
            .links
            .insert(inbound.identifier.clone(), inbound);
        let mut outbound = link("refresh-out", "refresh", "target");
        outbound.origin_slot = 1;
        engine
            .document
            .root
            .links
            .insert(outbound.identifier.clone(), outbound);
        engine.document.validate()?;
        let before = engine.document.clone();
        let inputs = vec![GraphPort::new("retained-input", image.clone())];
        let outputs = vec![GraphPort::new("retained-output", image)];
        let widgets = vec![GraphWidget {
            kind: GraphWidgetKind::Integer {
                minimum: 10,
                maximum: 20,
                step: 3,
            },
            ..integer_widget("steps", 10)
        }];
        let result = engine.apply(GraphCommand::ReconcileNode {
            identifier: GraphIdentifier::from("refresh"),
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            widgets: widgets.clone(),
            confirm_discard: false,
        });
        assert!(matches!(
            result,
            Err(GraphError::ReconciliationRequiresConfirmation { .. })
        ));
        assert_eq!(engine.document, before);
        engine.apply(GraphCommand::ReconcileNode {
            identifier: GraphIdentifier::from("refresh"),
            inputs,
            outputs,
            widgets,
            confirm_discard: true,
        })?;
        assert_eq!(
            engine.document.root.links[&GraphIdentifier::from("refresh-in")].target_slot,
            0
        );
        assert_eq!(
            engine.document.root.links[&GraphIdentifier::from("refresh-out")].origin_slot,
            0
        );
        let refreshed = &engine.document.root.nodes[&GraphIdentifier::from("refresh")];
        assert_eq!(refreshed.widgets[0].value, Value::from(16));
        assert!(refreshed.quarantine.contains_key("unmapped_widgets"));
        engine.apply(GraphCommand::ConvertWidgetToInput {
            node: GraphIdentifier::from("refresh"),
            widget: "steps".to_owned(),
            converted: true,
        })?;
        assert!(
            engine.document.root.nodes[&GraphIdentifier::from("refresh")]
                .inputs
                .iter()
                .any(|port| port.name == "steps")
        );
        let encoded = engine.encode()?;
        let restored = GraphCommandEngine::decode(&encoded)?;
        assert_eq!(restored.document, engine.document);
        let mut corrupt = engine.clone();
        corrupt
            .undo
            .back_mut()
            .ok_or("history entry")?
            .after
            .root
            .nodes
            .get_mut(&GraphIdentifier::from("refresh"))
            .ok_or("refresh history node")?
            .title = "corrupted".to_owned();
        assert_eq!(
            GraphCommandEngine::decode(&corrupt.encode()?),
            Err(GraphError::InvalidHistory)
        );
        Ok(encoded)
    }

    fn exercise_schema_one_definition_round_trip() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut document = fixture_engine()?.document;
        let definition_identifier = "1cb1644a-b9d7-5f3e-a907-15c5720f8729";
        let mut definition = simple_definition(definition_identifier);
        definition.inputs = Vec::new();
        definition.outputs = Vec::new();
        definition.description = "Native schema-one definition".to_owned();
        definition.search_aliases = vec!["native".to_owned(), "fixture".to_owned()];
        definition
            .unknown
            .insert("vendor-definition".to_owned(), json!({"kept": 7}));
        document
            .root
            .definitions
            .insert(definition.identifier.clone(), definition);
        document.root.source_fields.insert(
            "definitions".to_owned(),
            json!({"subgraphs": [], "vendor-container": {"kept": true}}),
        );
        let bytes = document
            .to_workflow_bytes()
            .map_err(|error| format!("schema-one definition fixture did not serialize: {error}"))?;
        let reparsed = GraphDocument::from_workflow_bytes(&bytes).map_err(|error| {
            format!(
                "schema-one definition fixture did not parse: {error}; {}",
                String::from_utf8_lossy(&bytes)
            )
        })?;
        let serialized = reparsed.to_workflow_value().map_err(|error| {
            format!("schema-one definition fixture did not reserialize: {error}")
        })?;
        let definitions = serialized
            .get("definitions")
            .and_then(Value::as_object)
            .ok_or("definitions object")?;
        assert_eq!(
            definitions.get("vendor-container"),
            Some(&json!({"kept": true}))
        );
        let reparsed_definition = reparsed
            .root
            .definitions
            .get(&GraphIdentifier::from(definition_identifier))
            .ok_or("schema definition")?;
        assert_eq!(
            reparsed_definition.unknown.get("vendor-definition"),
            Some(&json!({"kept": 7}))
        );
        let canonical = reparsed.to_workflow_bytes().map_err(|error| {
            format!("schema-one definition fixture did not serialize canonically: {error}")
        })?;
        let canonical_reparsed =
            GraphDocument::from_workflow_bytes(&canonical).map_err(|error| {
                format!(
                    "canonical schema-one definition fixture did not parse: {error}; {}",
                    String::from_utf8_lossy(&canonical)
                )
            })?;
        let reparsed_canonical = canonical_reparsed.to_workflow_bytes().map_err(|error| {
            format!(
                "canonical schema-one definition fixture did not reserialize: {error}; {}",
                String::from_utf8_lossy(&canonical)
            )
        })?;
        assert_eq!(canonical, reparsed_canonical);
        Ok(canonical)
    }

    fn exercise_disabled_prompt_projection() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        engine.apply(GraphCommand::Connect {
            link: link("disabled-link", "source", "target"),
            replace_existing: false,
        })?;
        engine.apply(GraphCommand::ToggleNodes {
            identifiers: BTreeSet::from([GraphIdentifier::from("source")]),
            toggle: NodeToggle::Disable,
        })?;
        assert_eq!(
            engine.document.root.nodes[&GraphIdentifier::from("source")].mode,
            GraphNodeMode::OnEvent
        );
        let disabled_bytes = engine.document.to_workflow_bytes()?;
        let restored = GraphDocument::from_workflow_bytes(&disabled_bytes)?;
        assert_eq!(
            restored.root.nodes[&GraphIdentifier::from("source")].mode,
            GraphNodeMode::OnEvent
        );
        let disabled_workflow = WorkflowFormatDocument::parse(&disabled_bytes)?;
        let disabled_prompt = crate::graph_to_prompt(&disabled_workflow, &BTreeMap::new(), "test")?;
        assert!(
            !disabled_prompt
                .prompt
                .0
                .contains_key(&sim_types::NodeId("source".to_owned()))
        );
        assert!(
            disabled_prompt
                .prompt
                .0
                .contains_key(&sim_types::NodeId("target".to_owned()))
        );
        assert!(engine.undo());
        let active_bytes = engine.document.to_workflow_bytes()?;
        let active_workflow = WorkflowFormatDocument::parse(&active_bytes)?;
        let active_prompt = crate::graph_to_prompt(&active_workflow, &BTreeMap::new(), "test")?;
        assert!(
            active_prompt
                .prompt
                .0
                .contains_key(&sim_types::NodeId("source".to_owned()))
        );
        let mut evidence = serde_json::to_vec(&disabled_prompt)?;
        evidence.extend_from_slice(&serde_json::to_vec(&active_prompt)?);
        evidence.extend_from_slice(&disabled_bytes);
        Ok(evidence)
    }

    fn exercise_imported_definition_removal() -> Result<Vec<u8>, Box<dyn Error>> {
        let schema_one_bytes = exercise_schema_one_definition_round_trip()?;
        let schema_one_document =
            GraphDocument::from_workflow_bytes(&schema_one_bytes).map_err(|error| {
                format!(
                    "schema-one definition fixture did not reopen: {error}; {}",
                    String::from_utf8_lossy(&schema_one_bytes)
                )
            })?;
        let mut schema_one = GraphCommandEngine::new(schema_one_document)
            .map_err(|error| format!("schema-one definition engine was invalid: {error}"))?;
        let schema_one_identifier = GraphIdentifier::from("1cb1644a-b9d7-5f3e-a907-15c5720f8729");
        schema_one
            .apply(GraphCommand::RemoveSubgraphDefinition {
                definition_identifier: schema_one_identifier,
                remove_instances: false,
            })
            .map_err(|error| format!("schema-one definition removal failed: {error}"))?;
        let schema_one_removed = schema_one
            .document
            .to_workflow_bytes()
            .map_err(|error| format!("schema-one definition removal did not serialize: {error}"))?;
        let schema_one_reopened =
            GraphDocument::from_workflow_bytes(&schema_one_removed).map_err(|error| {
                format!(
                    "schema-one definition removal did not reopen: {error}; {}",
                    String::from_utf8_lossy(&schema_one_removed)
                )
            })?;
        assert!(schema_one_reopened.root.definitions.is_empty());
        let schema_one_value: Value = serde_json::from_slice(&schema_one_removed)?;
        assert_eq!(
            schema_one_value["definitions"]["vendor-container"],
            json!({"kept": true})
        );
        assert_eq!(
            schema_one_value["definitions"]["subgraphs"],
            Value::Array(Vec::new())
        );

        let mut schema_zero_four_document = fixture_engine()?.document;
        schema_zero_four_document.workflow_format = Some(WorkflowFormat::Schema04);
        let schema_zero_four_identifier = GraphIdentifier::from("legacy-definition");
        schema_zero_four_document.root.definitions.insert(
            schema_zero_four_identifier,
            simple_definition("legacy-definition"),
        );
        let schema_zero_four_bytes = schema_zero_four_document
            .to_workflow_bytes()
            .map_err(|error| format!("schema-zero-four definition fixture failed: {error}"))?;
        let schema_zero_four_document = GraphDocument::from_workflow_bytes(&schema_zero_four_bytes)
            .map_err(|error| {
                format!("schema-zero-four definition fixture did not parse: {error}")
            })?;
        let schema_zero_four_identifier = schema_zero_four_document
            .root
            .definitions
            .keys()
            .next()
            .cloned()
            .ok_or("schema-zero-four definition fixture lost its definition")?;
        let mut schema_zero_four = GraphCommandEngine::new(schema_zero_four_document)
            .map_err(|error| format!("schema-zero-four definition engine was invalid: {error}"))?;
        schema_zero_four
            .apply(GraphCommand::RemoveSubgraphDefinition {
                definition_identifier: schema_zero_four_identifier,
                remove_instances: false,
            })
            .map_err(|error| format!("schema-zero-four definition removal failed: {error}"))?;
        let schema_zero_four_removed =
            schema_zero_four
                .document
                .to_workflow_bytes()
                .map_err(|error| {
                    format!("schema-zero-four definition removal did not serialize: {error}")
                })?;
        let schema_zero_four_reopened =
            GraphDocument::from_workflow_bytes(&schema_zero_four_removed).map_err(|error| {
                format!(
                    "schema-zero-four definition removal did not reopen: {error}; {}",
                    String::from_utf8_lossy(&schema_zero_four_removed)
                )
            })?;
        assert!(schema_zero_four_reopened.root.definitions.is_empty());
        let schema_zero_four_value: Value = serde_json::from_slice(&schema_zero_four_removed)?;
        assert_eq!(
            schema_zero_four_value["definitions"]["subgraphs"],
            Value::Array(Vec::new())
        );

        let mut evidence = schema_one_removed;
        evidence.extend_from_slice(&schema_zero_four_removed);
        Ok(evidence)
    }

    fn exercise_link_rectangle_geometry() -> Result<Vec<u8>, Box<dyn Error>> {
        assert!(!segments_intersect(
            GraphPoint { x: 0.0, y: 0.0 },
            GraphPoint { x: 10.0, y: 0.0 },
            GraphPoint { x: 20.0, y: 0.0 },
            GraphPoint { x: 30.0, y: 0.0 },
        ));
        assert!(segments_intersect(
            GraphPoint { x: 0.0, y: 0.0 },
            GraphPoint { x: 10.0, y: 0.0 },
            GraphPoint { x: 10.0, y: 0.0 },
            GraphPoint { x: 10.0, y: 8.0 },
        ));
        assert!(segments_intersect(
            GraphPoint { x: 0.0, y: 0.0 },
            GraphPoint { x: 10.0, y: 10.0 },
            GraphPoint { x: 0.0, y: 10.0 },
            GraphPoint { x: 10.0, y: 0.0 },
        ));

        let mut graph = fixture_engine()?.document.root;
        let mut routed = link("routed-geometry", "source", "target");
        let reroute_identifier = GraphIdentifier::from("geometry-reroute");
        routed.parent_reroute = Some(reroute_identifier.clone());
        graph.reroutes.insert(
            reroute_identifier.clone(),
            GraphReroute {
                identifier: reroute_identifier,
                position: GraphPoint { x: 400.0, y: 220.0 },
                parent: None,
                floating_type: None,
                source_fields: Map::new(),
            },
        );
        graph
            .links
            .insert(routed.identifier.clone(), routed.clone());
        assert!(link_intersects_rect(
            &graph,
            &routed,
            GraphRect {
                origin: GraphPoint { x: 395.0, y: 215.0 },
                size: GraphSize {
                    width: 10.0,
                    height: 10.0,
                },
            },
        ));
        assert!(!link_intersects_rect(
            &graph,
            &routed,
            GraphRect {
                origin: GraphPoint {
                    x: 4_000.0,
                    y: 42.0
                },
                size: GraphSize {
                    width: 20.0,
                    height: 20.0,
                },
            },
        ));
        Ok(serde_json::to_vec(&graph)?)
    }

    fn published_blueprint_fixture() -> Result<GraphCommandEngine, GraphError> {
        let mut engine = fixture_engine()?;
        let nested_identifier = GraphIdentifier::from("nested-blueprint-definition");
        let nested_definition = simple_definition("nested-blueprint-definition");
        let definition_identifier = GraphIdentifier::from("published-blueprint-definition");
        let mut definition = simple_definition("published-blueprint-definition");
        let internal_identifier = GraphIdentifier::from("published-blueprint-definition-internal");
        let internal = definition
            .graph
            .nodes
            .get_mut(&internal_identifier)
            .ok_or_else(|| GraphError::UnknownNode(internal_identifier.clone()))?;
        internal.type_identifier = nested_identifier.text();
        internal.subgraph_definition = Some(nested_identifier.clone());
        internal.inputs.push(GraphPort::new(
            "image",
            GraphPortType::Concrete("IMAGE".to_owned()),
        ));
        internal.outputs.push(GraphPort::new(
            "image",
            GraphPortType::Concrete("IMAGE".to_owned()),
        ));
        let exposed_widget = integer_widget("seed", 5);
        internal.widgets.push(exposed_widget.clone());
        definition
            .graph
            .definitions
            .insert(nested_identifier, nested_definition);
        definition.inputs = vec![SubgraphPort {
            identifier: "published-input".to_owned(),
            name: "image".to_owned(),
            port_type: GraphPortType::Concrete("IMAGE".to_owned()),
            internal_node: Some(internal_identifier.clone()),
            internal_slot: 0,
            source_fields: Map::new(),
        }];
        definition.outputs = vec![SubgraphPort {
            identifier: "published-output".to_owned(),
            name: "image".to_owned(),
            port_type: GraphPortType::Concrete("IMAGE".to_owned()),
            internal_node: Some(internal_identifier.clone()),
            internal_slot: 0,
            source_fields: Map::new(),
        }];
        definition.description = "Fallback definition description".to_owned();
        definition.search_aliases = vec!["fallback".to_owned()];
        let mut exposed_widget_schema = exposed_widget.clone();
        exposed_widget_schema.identifier = "published-seed".to_owned();
        definition.exposed_widgets = vec![SubgraphExposedWidget {
            identifier: "published-seed".to_owned(),
            internal_node: internal_identifier.clone(),
            internal_widget: "seed".to_owned(),
            widget: exposed_widget_schema,
            unknown: BTreeMap::from([("future".to_owned(), Value::Bool(true))]),
        }];
        definition.unknown.insert(
            "extra".to_owned(),
            json!({
                (BLUEPRINT_DESCRIPTION_FIELD): "A deterministic native image blueprint",
                (BLUEPRINT_SEARCH_ALIASES_FIELD): ["blend", "native image"],
                "workflowRendererVersion": "1.0"
            }),
        );
        let mut instance = node(
            "published-blueprint-instance",
            Some(GraphPortType::Concrete("IMAGE".to_owned())),
            Some(GraphPortType::Concrete("IMAGE".to_owned())),
        );
        instance.title = "Suggested Native Blend".to_owned();
        instance.type_identifier = definition_identifier.text();
        instance.subgraph_definition = Some(definition_identifier.clone());
        instance.inputs[0].name = "image".to_owned();
        instance.outputs[0].name = "image".to_owned();
        let mut instance_widget = exposed_widget;
        instance_widget.identifier = "published-seed".to_owned();
        instance.widgets.push(instance_widget);
        instance.position = GraphPoint { x: 320.0, y: 240.0 };
        engine
            .document
            .root
            .definitions
            .insert(definition_identifier, definition);
        engine
            .document
            .root
            .nodes
            .insert(instance.identifier.clone(), instance.clone());
        engine.document.root.selection = GraphSelection {
            nodes: BTreeSet::from([instance.identifier]),
            ..GraphSelection::default()
        };
        GraphCommandEngine::new(engine.document)
    }

    #[test]
    fn published_subgraph_blueprint_is_minimal_deterministic_and_instantiates_once()
    -> Result<(), Box<dyn Error>> {
        let source = published_blueprint_fixture().map_err(|error| format!("fixture: {error}"))?;
        let first = source
            .document
            .export_selected_subgraph_blueprint("Native Blend")
            .map_err(|error| format!("first export: {error}"))?;
        let second = source
            .document
            .export_selected_subgraph_blueprint("Native Blend")?;
        assert_eq!(first.workflow_bytes, second.workflow_bytes);
        assert_eq!(first.metadata.filename, "Native Blend.json");
        assert_eq!(first.metadata.display_name, "Native Blend");
        assert_eq!(first.metadata.suggested_name, "Suggested Native Blend");
        assert_eq!(
            first.metadata.description,
            "A deterministic native image blueprint"
        );
        assert_eq!(
            first.metadata.search_aliases,
            ["blend".to_owned(), "native image".to_owned()]
        );
        assert_eq!(first.metadata.inputs.len(), 1);
        assert_eq!(first.metadata.outputs.len(), 1);
        assert_eq!(first.metadata.exposed_widgets.len(), 1);
        assert_eq!(
            first.metadata.exposed_widgets[0].identifier,
            "published-seed"
        );
        assert!(matches!(
            first.metadata.exposed_widgets[0].widget.kind,
            GraphWidgetKind::Integer { .. }
        ));
        assert_eq!(first.metadata.exposed_widgets[0].unknown["future"], true);

        let value: Value = serde_json::from_slice(&first.workflow_bytes)?;
        assert_eq!(value["version"], 0.4);
        assert_eq!(value["revision"], 0);
        assert_eq!(value["nodes"].as_array().map(Vec::len), Some(1));
        assert_eq!(value["nodes"][0]["inputs"][0]["name"], "image");
        assert_eq!(value["nodes"][0]["outputs"][0]["name"], "image");
        assert_eq!(value["links"], json!([]));
        assert_eq!(
            value["definitions"]["subgraphs"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(
            value["extra"][BLUEPRINT_DESCRIPTION_FIELD],
            "A deterministic native image blueprint"
        );
        assert_eq!(
            value["extra"][BLUEPRINT_SEARCH_ALIASES_FIELD],
            json!(["blend", "native image"])
        );
        assert!(
            value["definitions"]["subgraphs"][0]["extra"]
                .get(BLUEPRINT_DESCRIPTION_FIELD)
                .is_none()
        );
        assert_eq!(
            value["definitions"]["subgraphs"][0]["extra"]["workflowRendererVersion"],
            "1.0"
        );

        let restored = GraphDocument::from_workflow_bytes(&first.workflow_bytes)?;
        assert_eq!(restored.root.nodes.len(), 1);
        assert!(restored.root.links.is_empty());
        let restored_definition = restored
            .root
            .definitions
            .values()
            .next()
            .ok_or("definition")?;
        assert_eq!(restored_definition.graph.definitions.len(), 1);
        let decoded =
            PublishedSubgraphBlueprint::decode(&first.metadata.filename, &first.workflow_bytes)?;
        assert_eq!(decoded.clipboard, first.clipboard);
        assert_eq!(decoded.metadata.display_name, first.metadata.display_name);
        assert_eq!(decoded.metadata.inputs, first.metadata.inputs);
        assert_eq!(decoded.metadata.outputs, first.metadata.outputs);
        assert_eq!(
            GraphClipboard::decode(&first.clipboard.encode()?)?,
            first.clipboard
        );

        let mut destination = fixture_engine()?;
        let before = destination.document.clone();
        let original_instance_identifier = first.clipboard.nodes[0].identifier.clone();
        let original_definition_identifier = first.clipboard.definitions[0].identifier.clone();
        destination.apply(
            first.instantiate_command(&destination.document, GraphPoint { x: 40.0, y: 60.0 })?,
        )?;
        assert_eq!(destination.document.root.definitions.len(), 1);
        let pasted_identifier = destination
            .document
            .root
            .selection
            .nodes
            .iter()
            .next()
            .ok_or("pasted selection")?;
        let pasted = destination
            .document
            .root
            .nodes
            .get(pasted_identifier)
            .ok_or("pasted instance")?;
        let pasted_definition_identifier = pasted
            .subgraph_definition
            .as_ref()
            .ok_or("pasted definition identifier")?;
        assert_ne!(pasted_identifier, &original_instance_identifier);
        assert_ne!(
            pasted_definition_identifier,
            &original_definition_identifier
        );
        let pasted_definition = destination
            .document
            .root
            .definitions
            .get(pasted_definition_identifier)
            .ok_or("pasted definition")?;
        assert_eq!(pasted_definition.graph.definitions.len(), 1);
        assert_eq!(pasted_definition.description, first.metadata.description);
        assert_eq!(
            pasted.position,
            GraphPoint {
                x: first.clipboard.nodes[0].position.x + 40.0,
                y: first.clipboard.nodes[0].position.y + 60.0,
            }
        );
        assert!(destination.undo());
        assert_eq!(destination.document, before);
        assert!(!destination.undo());
        Ok(())
    }

    #[test]
    fn published_subgraph_blueprint_rejects_selection_shape_names_and_malformed_metadata()
    -> Result<(), Box<dyn Error>> {
        let mut engine =
            published_blueprint_fixture().map_err(|error| format!("fixture: {error}"))?;
        engine.document.root.selection = GraphSelection::default();
        assert_eq!(
            engine
                .document
                .export_selected_subgraph_blueprint("Native Blend"),
            Err(GraphError::InvalidBlueprintSelection(0))
        );
        engine.document.root.selection.nodes = BTreeSet::from([
            GraphIdentifier::from("published-blueprint-instance"),
            GraphIdentifier::from("source"),
        ]);
        assert_eq!(
            engine
                .document
                .export_selected_subgraph_blueprint("Native Blend"),
            Err(GraphError::InvalidBlueprintSelection(2))
        );
        engine.document.root.selection.nodes = BTreeSet::from([GraphIdentifier::from("source")]);
        assert!(matches!(
            engine
                .document
                .export_selected_subgraph_blueprint("Native Blend"),
            Err(GraphError::NotSubgraphInstance(_))
        ));

        let source = published_blueprint_fixture()?;
        for invalid in ["", ".", "..", "../escape", "folder/name", " padded"] {
            assert!(matches!(
                source.document.export_selected_subgraph_blueprint(invalid),
                Err(GraphError::InvalidGraphLabel | GraphError::InvalidBlueprintName(_))
            ));
        }
        let blueprint = source
            .document
            .export_selected_subgraph_blueprint("Native Blend")?;
        assert!(matches!(
            PublishedSubgraphBlueprint::decode("Other Name.json", &blueprint.workflow_bytes),
            Err(GraphError::BlueprintNameMismatch { .. })
        ));
        assert!(matches!(
            PublishedSubgraphBlueprint::decode("../Native Blend.json", &blueprint.workflow_bytes),
            Err(GraphError::InvalidBlueprintName(_))
        ));

        let mut malformed: Value = serde_json::from_slice(&blueprint.workflow_bytes)?;
        malformed["extra"][BLUEPRINT_SEARCH_ALIASES_FIELD] = json!(["valid", 7]);
        assert!(matches!(
            PublishedSubgraphBlueprint::decode(
                &blueprint.metadata.filename,
                &serde_json::to_vec(&malformed)?
            ),
            Err(GraphError::InvalidBlueprint(_))
        ));
        let mut mismatched_ports: Value = serde_json::from_slice(&blueprint.workflow_bytes)?;
        mismatched_ports["nodes"][0]["inputs"][0]["name"] = json!("wrong");
        assert!(matches!(
            PublishedSubgraphBlueprint::decode(
                &blueprint.metadata.filename,
                &serde_json::to_vec(&mismatched_ports)?
            ),
            Err(GraphError::InvalidBlueprint(_))
        ));
        let mut multiple: Value = serde_json::from_slice(&blueprint.workflow_bytes)?;
        let mut duplicate = multiple["nodes"][0].clone();
        duplicate["id"] = json!(999);
        multiple["nodes"]
            .as_array_mut()
            .ok_or("nodes")?
            .push(duplicate);
        assert!(matches!(
            PublishedSubgraphBlueprint::decode(
                &blueprint.metadata.filename,
                &serde_json::to_vec(&multiple)?
            ),
            Err(GraphError::InvalidBlueprint(_))
        ));
        Ok(())
    }

    #[test]
    fn subgraph_conversion_and_unpack_preserve_boundaries_and_nested_state()
    -> Result<(), Box<dyn Error>> {
        exercise_subgraph_conversion_and_unpack()?;
        Ok(())
    }

    #[test]
    fn clipboard_closes_over_groups_reroutes_and_definitions_without_collisions()
    -> Result<(), Box<dyn Error>> {
        exercise_clipboard_remapping()?;
        Ok(())
    }

    #[test]
    fn reconciliation_is_confirmed_and_history_corruption_is_rejected() -> Result<(), Box<dyn Error>>
    {
        exercise_reconciliation_and_history()?;
        Ok(())
    }

    #[test]
    fn schema_one_definitions_preserve_container_and_definition_unknown_fields()
    -> Result<(), Box<dyn Error>> {
        exercise_schema_one_definition_round_trip()?;
        Ok(())
    }

    #[test]
    fn disabled_nodes_are_omitted_from_prompts_and_undo_restores_execution()
    -> Result<(), Box<dyn Error>> {
        exercise_disabled_prompt_projection()?;
        Ok(())
    }

    #[test]
    fn imported_definition_removal_survives_save_and_reopen() -> Result<(), Box<dyn Error>> {
        exercise_imported_definition_removal()?;
        Ok(())
    }

    #[test]
    fn link_rectangle_geometry_rejects_disjoint_collinear_segments() -> Result<(), Box<dyn Error>> {
        exercise_link_rectangle_geometry()?;
        Ok(())
    }

    #[test]
    fn val_domain_003() -> Result<(), Box<dyn Error>> {
        let mut cases = Vec::new();

        let mut engine = fixture_engine()?;
        engine.apply(GraphCommand::Connect {
            link: link("link", "source", "target"),
            replace_existing: false,
        })?;
        let connected = engine.document.to_workflow_bytes()?;
        let initial = fixture_engine()?.document.to_workflow_bytes()?;
        cases.push(json!({
            "name": "typed-link-serialization-and-atomic-invalid-operation",
            "passed": engine.document.root.links.len() == 1,
            "pre_state_digest": format!("{:x}", Sha256::digest(&initial)),
            "post_state_digest": format!("{:x}", Sha256::digest(&connected)),
            "digest": format!("{:x}", Sha256::digest(&connected)),
        }));
        let before_invalid = engine.document.clone();
        let invalid = node(
            "invalid-audio",
            Some(GraphPortType::Concrete("AUDIO".to_owned())),
            None,
        );
        engine.apply(GraphCommand::AddNode {
            node: invalid,
            source: NodeCreationSource::Library,
        })?;
        let before_invalid_link = engine.document.clone();
        assert!(matches!(
            engine.apply(GraphCommand::Connect {
                link: link("invalid-link", "source", "invalid-audio"),
                replace_existing: false,
            }),
            Err(GraphError::IncompatiblePorts { .. })
        ));
        assert_eq!(engine.document, before_invalid_link);
        assert_ne!(engine.document, before_invalid);
        let invalid_state = engine.document.to_workflow_bytes()?;
        cases.push(json!({
            "name": "incompatible-link-rejection-is-atomic",
            "passed": true,
            "pre_state_digest": format!("{:x}", Sha256::digest(before_invalid_link.to_workflow_bytes()?)),
            "post_state_digest": format!("{:x}", Sha256::digest(&invalid_state)),
            "digest": format!("{:x}", Sha256::digest(&invalid_state)),
        }));

        let clipboard_bytes = exercise_clipboard_remapping()?;
        cases.push(json!({
            "name": "clipboard-group-reroute-definition-remap-and-undo",
            "passed": true,
            "digest": format!("{:x}", Sha256::digest(&clipboard_bytes)),
        }));

        let subgraph_bytes = exercise_subgraph_conversion_and_unpack()?;
        cases.push(json!({
            "name": "subgraph-boundaries-groups-reroutes-nesting-and-unpack",
            "passed": true,
            "digest": format!("{:x}", Sha256::digest(&subgraph_bytes)),
        }));

        let reconciliation_bytes = exercise_reconciliation_and_history()?;
        cases.push(json!({
            "name": "definition-reconciliation-widget-promotion-and-history-integrity",
            "passed": true,
            "digest": format!("{:x}", Sha256::digest(&reconciliation_bytes)),
        }));

        let schema_bytes = exercise_schema_one_definition_round_trip()?;
        cases.push(json!({
            "name": "schema-one-definition-and-unknown-field-round-trip",
            "passed": true,
            "digest": format!("{:x}", Sha256::digest(&schema_bytes)),
        }));

        let audited_bytes = exercise_audited_graph_failures()?;
        cases.push(json!({
            "name": "group-move-link-rectangle-replacement-and-widget-schema-atomicity",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&audited_bytes)),
            "digest": format!("{:x}", Sha256::digest(&audited_bytes)),
        }));

        let same_name_reconciliation = exercise_same_name_type_reconciliation()?;
        cases.push(json!({
            "name": "same-name-incompatible-port-reconciliation-quarantine",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&same_name_reconciliation)),
            "digest": format!("{:x}", Sha256::digest(&same_name_reconciliation)),
        }));

        let api_prompt_bytes = exercise_api_prompt_import()?;
        cases.push(json!({
            "name": "api-prompt-native-import-and-schema-one-projection",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&api_prompt_bytes)),
            "digest": format!("{:x}", Sha256::digest(&api_prompt_bytes)),
        }));

        let recursive_schema_bytes = exercise_recursive_schema_one()?;
        cases.push(json!({
            "name": "recursive-schema-one-boundary-uuid-round-trip",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&recursive_schema_bytes)),
            "digest": format!("{:x}", Sha256::digest(&recursive_schema_bytes)),
        }));

        let schema_zero_four_bytes = exercise_schema_zero_four_reroutes()?;
        cases.push(json!({
            "name": "schema-zero-four-reroute-link-extension-round-trip",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&schema_zero_four_bytes)),
            "digest": format!("{:x}", Sha256::digest(&schema_zero_four_bytes)),
        }));

        let large_graph_bytes = exercise_large_graph()?;
        cases.push(json!({
            "name": "large-graph-selection-layout-and-serialization",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&large_graph_bytes)),
            "digest": format!("{:x}", Sha256::digest(&large_graph_bytes)),
        }));

        let prompt_widget_bytes = exercise_native_widget_prompt_projection()?;
        cases.push(json!({
            "name": "native-widget-workflow-and-prompt-values-remain-separate",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&prompt_widget_bytes)),
            "digest": format!("{:x}", Sha256::digest(&prompt_widget_bytes)),
        }));

        let dynamic_port_bytes = exercise_virtual_multiple_and_dynamic_ports()?;
        cases.push(json!({
            "name": "visible-trailing-dynamic-port-expands-after-each-typed-link",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&dynamic_port_bytes)),
            "digest": format!("{:x}", Sha256::digest(&dynamic_port_bytes)),
        }));

        let disabled_prompt_bytes = exercise_disabled_prompt_projection()?;
        cases.push(json!({
            "name": "disabled-node-prompt-omission-round-trip-and-undo",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&disabled_prompt_bytes)),
            "digest": format!("{:x}", Sha256::digest(&disabled_prompt_bytes)),
        }));

        let removed_definition_bytes = exercise_imported_definition_removal()?;
        cases.push(json!({
            "name": "imported-definition-removal-survives-schema-save-and-reopen",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&removed_definition_bytes)),
            "digest": format!("{:x}", Sha256::digest(&removed_definition_bytes)),
        }));

        let link_rectangle_bytes = exercise_link_rectangle_geometry()?;
        cases.push(json!({
            "name": "link-rectangle-hit-testing-rejects-disjoint-collinear-segments",
            "passed": true,
            "post_state_digest": format!("{:x}", Sha256::digest(&link_rectangle_bytes)),
            "digest": format!("{:x}", Sha256::digest(&link_rectangle_bytes)),
        }));

        let action_ids = CatalogGraphAction::ALL
            .iter()
            .map(|action| action.command_id())
            .collect::<Vec<_>>();
        let action_bytes = serde_json::to_vec(&action_ids)?;
        cases.push(json!({
            "name": "cataloged-graph-action-identifier-reconciliation",
            "passed": action_ids.len() == 37 && action_ids.iter().copied().collect::<BTreeSet<_>>().len() == 37,
            "input_digest": format!("{:x}", Sha256::digest(&action_bytes)),
            "digest": format!("{:x}", Sha256::digest(&action_bytes)),
        }));

        if cases.iter().any(|case| case["passed"] != true) {
            return Err("VAL-DOMAIN-003 case failed".into());
        }
        let artifact = json!({
            "validation_id": "VAL-DOMAIN-003",
            "environment": {"backend": "native-rust", "device": "cpu", "schema": GRAPH_DOCUMENT_SCHEMA_VERSION},
            "fixture_digests": {
                "workflow": format!("{:x}", Sha256::digest(&connected)),
                "clipboard": format!("{:x}", Sha256::digest(&clipboard_bytes)),
                "subgraph": format!("{:x}", Sha256::digest(&subgraph_bytes)),
                "reconciliation_history": format!("{:x}", Sha256::digest(&reconciliation_bytes)),
                "schema_one": format!("{:x}", Sha256::digest(&schema_bytes)),
                "audited_atomicity": format!("{:x}", Sha256::digest(&audited_bytes)),
                "same_name_reconciliation": format!("{:x}", Sha256::digest(&same_name_reconciliation)),
                "api_prompt": format!("{:x}", Sha256::digest(&api_prompt_bytes)),
                "recursive_schema_one": format!("{:x}", Sha256::digest(&recursive_schema_bytes)),
                "schema_zero_four": format!("{:x}", Sha256::digest(&schema_zero_four_bytes)),
                "large_graph": format!("{:x}", Sha256::digest(&large_graph_bytes)),
                "prompt_widget_values": format!("{:x}", Sha256::digest(&prompt_widget_bytes)),
                "dynamic_ports": format!("{:x}", Sha256::digest(&dynamic_port_bytes)),
                "disabled_prompt": format!("{:x}", Sha256::digest(&disabled_prompt_bytes)),
                "removed_definitions": format!("{:x}", Sha256::digest(&removed_definition_bytes)),
                "link_rectangle_geometry": format!("{:x}", Sha256::digest(&link_rectangle_bytes)),
                "catalog_actions": format!("{:x}", Sha256::digest(&action_bytes)),
            },
            "cases": cases,
            "skipped": [],
        });
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/sim-parity/val-domain-003.json");
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&artifact)?)?;
        Ok(())
    }

    fn exercise_virtual_multiple_and_dynamic_ports() -> Result<Vec<u8>, Box<dyn Error>> {
        let virtual_type =
            GraphPortType::Virtual(BTreeSet::from(["IMAGE".to_owned(), "MASK".to_owned()]));
        assert_eq!(
            GraphPortType::from_name(&virtual_type.display_name()),
            virtual_type
        );
        assert_eq!(
            GraphPortType::from_name("virtual()").display_name(),
            "virtual()"
        );

        let mut engine = fixture_engine()?;
        let mut second_source = node(
            "second-source",
            None,
            Some(GraphPortType::Concrete("MASK".to_owned())),
        );
        second_source.position.x = -600.0;
        engine
            .document
            .root
            .nodes
            .insert(second_source.identifier.clone(), second_source);
        engine.apply(GraphCommand::Connect {
            link: link("single-first", "source", "target"),
            replace_existing: false,
        })?;
        let before_occupied = engine.document.clone();
        assert!(matches!(
            engine.apply(GraphCommand::Connect {
                link: link("single-second", "second-source", "target"),
                replace_existing: false,
            }),
            Err(GraphError::InputOccupied { .. })
        ));
        assert_eq!(engine.document, before_occupied);
        engine.apply(GraphCommand::RemoveLink {
            identifier: GraphIdentifier::from("single-first"),
        })?;
        let target = engine
            .document
            .root
            .nodes
            .get_mut(&GraphIdentifier::from("target"))
            .ok_or("target")?;
        target.inputs[0].port_type = virtual_type;
        target.inputs[0].multiple = true;
        engine.apply(GraphCommand::Connect {
            link: link("first", "source", "target"),
            replace_existing: false,
        })?;
        engine.apply(GraphCommand::Connect {
            link: link("second", "second-source", "target"),
            replace_existing: false,
        })?;
        assert_eq!(engine.document.root.links.len(), 2);

        let mut audio_source = node(
            "audio-source",
            None,
            Some(GraphPortType::Virtual(BTreeSet::from(["AUDIO".to_owned()]))),
        );
        audio_source.position.y = 300.0;
        engine
            .document
            .root
            .nodes
            .insert(audio_source.identifier.clone(), audio_source);
        let before_disjoint = engine.document.clone();
        assert!(matches!(
            engine.apply(GraphCommand::Connect {
                link: link("audio", "audio-source", "target"),
                replace_existing: false,
            }),
            Err(GraphError::IncompatiblePorts { .. })
        ));
        assert_eq!(engine.document, before_disjoint);

        let mut dynamic_target = node(
            "dynamic-target",
            Some(GraphPortType::Concrete("IMAGE".to_owned())),
            None,
        );
        dynamic_target.inputs[0].dynamic = true;
        dynamic_target.position.y = 600.0;
        engine
            .document
            .root
            .nodes
            .insert(dynamic_target.identifier.clone(), dynamic_target);
        let mut dynamic_link = link("dynamic-first", "source", "dynamic-target");
        dynamic_link.target_slot = 0;
        engine.apply(GraphCommand::Connect {
            link: dynamic_link,
            replace_existing: false,
        })?;
        let inputs = &engine.document.root.nodes[&GraphIdentifier::from("dynamic-target")].inputs;
        assert_eq!(inputs.len(), 2);
        assert!(!inputs[0].dynamic);
        assert!(inputs[1].dynamic);
        assert_eq!(
            engine.document.root.links[&GraphIdentifier::from("dynamic-first")].target_slot,
            0
        );

        let mut second_dynamic_link = link("dynamic-second", "source", "dynamic-target");
        second_dynamic_link.target_slot = 1;
        engine.apply(GraphCommand::Connect {
            link: second_dynamic_link,
            replace_existing: false,
        })?;
        let inputs = &engine.document.root.nodes[&GraphIdentifier::from("dynamic-target")].inputs;
        assert_eq!(inputs.len(), 3);
        assert!(!inputs[1].dynamic);
        assert!(inputs[2].dynamic);
        assert_eq!(
            engine.document.root.links[&GraphIdentifier::from("dynamic-second")].target_slot,
            1
        );

        let mut invalid = engine.document.clone();
        invalid
            .root
            .nodes
            .get_mut(&GraphIdentifier::from("dynamic-target"))
            .and_then(|node| node.inputs.first_mut())
            .ok_or("dynamic target input")?
            .dynamic = true;
        assert!(matches!(
            invalid.validate(),
            Err(GraphError::DynamicInputMustBeLast { slot: 0, .. })
        ));
        assert_ne!(before_disjoint, engine.document);
        Ok(engine.encode()?)
    }

    #[test]
    fn virtual_multiple_and_dynamic_ports_apply_native_link_rules() -> Result<(), Box<dyn Error>> {
        exercise_virtual_multiple_and_dynamic_ports()?;
        Ok(())
    }

    #[test]
    fn reroute_removal_splices_branches_and_routes_deterministically() -> Result<(), Box<dyn Error>>
    {
        let mut engine = fixture_engine()?;
        for (identifier, parent) in [
            ("grand", None),
            ("parent", Some("grand")),
            ("first-child", Some("parent")),
            ("second-child", Some("parent")),
        ] {
            let mut source_fields = Map::new();
            if identifier == "first-child" {
                source_fields.insert(
                    "floating".to_owned(),
                    json!({"slotType":"output", "type":"virtual(IMAGE,MASK)", "vendor":7}),
                );
            }
            engine.document.root.reroutes.insert(
                GraphIdentifier::from(identifier),
                GraphReroute {
                    identifier: GraphIdentifier::from(identifier),
                    position: GraphPoint::ZERO,
                    parent: parent.map(GraphIdentifier::from),
                    floating_type: (identifier == "first-child").then(|| "output".to_owned()),
                    source_fields,
                },
            );
        }
        let mut routed = link("routed", "source", "target");
        routed.parent_reroute = Some(GraphIdentifier::from("parent"));
        engine
            .document
            .root
            .links
            .insert(routed.identifier.clone(), routed);
        engine.document.validate()?;
        assert_eq!(
            engine
                .document
                .root
                .resolve_reroute_port_type(&GraphIdentifier::from("first-child"))?,
            GraphPortType::Any
        );
        assert_eq!(
            engine
                .document
                .root
                .resolve_reroute_port_type(&GraphIdentifier::from("parent"))?,
            GraphPortType::Concrete("IMAGE".to_owned())
        );

        engine.apply(GraphCommand::RemoveReroute {
            identifier: GraphIdentifier::from("parent"),
        })?;
        for child in ["first-child", "second-child"] {
            assert_eq!(
                engine.document.root.reroutes[&GraphIdentifier::from(child)].parent,
                Some(GraphIdentifier::from("grand"))
            );
        }
        assert_eq!(
            engine.document.root.links[&GraphIdentifier::from("routed")].parent_reroute,
            Some(GraphIdentifier::from("grand"))
        );
        let before_cycle = engine.document.clone();
        assert!(matches!(
            engine.apply(GraphCommand::ReparentReroute {
                identifier: GraphIdentifier::from("grand"),
                parent: Some(GraphIdentifier::from("first-child")),
            }),
            Err(GraphError::RerouteCycle(_))
        ));
        assert_eq!(engine.document, before_cycle);

        let restored = GraphDocument::from_workflow_bytes(&engine.document.to_workflow_bytes()?)?;
        let floating = restored
            .root
            .reroutes
            .values()
            .find(|reroute| reroute.floating_type.as_deref() == Some("output"))
            .ok_or("floating reroute")?;
        assert_eq!(floating.floating_type.as_deref(), Some("output"));
        assert_eq!(
            restored
                .root
                .resolve_reroute_port_type(&floating.identifier)?,
            GraphPortType::Any
        );
        assert_eq!(floating.source_fields["floating"]["vendor"], 7);
        Ok(())
    }

    #[test]
    fn native_widget_metadata_round_trips_typed_state_and_prompt_values()
    -> Result<(), Box<dyn Error>> {
        let mut document = fixture_engine()?.document;
        let widgets = vec![
            GraphWidget {
                identifier: "boolean".to_owned(),
                kind: GraphWidgetKind::Boolean,
                value: json!(true),
                prompt_value: json!(false),
                validation: WidgetValidation::Valid,
                converted_to_input: false,
                visible: true,
                unknown: BTreeMap::from([("vendor".to_owned(), json!({"x":1}))]),
            },
            GraphWidget {
                identifier: "integer".to_owned(),
                kind: GraphWidgetKind::Integer {
                    minimum: 1,
                    maximum: 9,
                    step: 2,
                },
                value: json!(5),
                prompt_value: json!(7),
                validation: WidgetValidation::Valid,
                converted_to_input: true,
                visible: false,
                unknown: BTreeMap::new(),
            },
            GraphWidget {
                identifier: "float".to_owned(),
                kind: GraphWidgetKind::Float {
                    minimum: 0.0,
                    maximum: 2.0,
                    step: 0.25,
                },
                value: json!(1.25),
                prompt_value: json!(1.5),
                validation: WidgetValidation::Valid,
                converted_to_input: false,
                visible: true,
                unknown: BTreeMap::new(),
            },
            GraphWidget {
                identifier: "text".to_owned(),
                kind: GraphWidgetKind::Text { multiline: true },
                value: json!("workflow"),
                prompt_value: json!("prompt"),
                validation: WidgetValidation::Valid,
                converted_to_input: false,
                visible: true,
                unknown: BTreeMap::new(),
            },
            GraphWidget {
                identifier: "combo".to_owned(),
                kind: GraphWidgetKind::Combo {
                    values: vec!["a".to_owned(), "b".to_owned()],
                    dynamic: false,
                },
                value: json!("a"),
                prompt_value: json!("b"),
                validation: WidgetValidation::Valid,
                converted_to_input: false,
                visible: true,
                unknown: BTreeMap::new(),
            },
        ];
        document
            .root
            .nodes
            .get_mut(&GraphIdentifier::from("source"))
            .ok_or("source")?
            .widgets = widgets.clone();
        let serialized = document.to_workflow_value()?;
        assert_eq!(serialized["nodes"][0][NATIVE_WIDGETS_FIELD]["version"], 1);
        let restored = GraphDocument::from_workflow_bytes(&serde_json::to_vec(&serialized)?)?;
        assert_eq!(
            restored.root.nodes[&GraphIdentifier::from("source")].widgets,
            widgets
        );

        let mut imported = serialized.clone();
        let source = imported["nodes"]
            .as_array_mut()
            .and_then(|nodes| nodes.iter_mut().find(|node| node["id"] == "source"))
            .and_then(Value::as_object_mut)
            .ok_or("serialized source")?;
        source.remove(NATIVE_WIDGETS_FIELD);
        let imported = GraphDocument::from_workflow_bytes(&serde_json::to_vec(&imported)?)?;
        assert!(
            imported.root.nodes[&GraphIdentifier::from("source")]
                .widgets
                .iter()
                .all(|widget| matches!(widget.kind, GraphWidgetKind::Preserved { .. }))
        );

        let mut invalid_schema = serialized.clone();
        let mut invalid_widgets = widgets;
        invalid_widgets[1].kind = GraphWidgetKind::Integer {
            minimum: 1,
            maximum: 9,
            step: 0,
        };
        let invalid_envelope = serde_json::to_value(NativeWidgetsEnvelope {
            version: NATIVE_WIDGETS_VERSION,
            widgets: invalid_widgets,
            unknown: BTreeMap::new(),
        })?;
        invalid_schema["nodes"]
            .as_array_mut()
            .and_then(|nodes| nodes.iter_mut().find(|node| node["id"] == "source"))
            .and_then(Value::as_object_mut)
            .ok_or("invalid schema source")?
            .insert(NATIVE_WIDGETS_FIELD.to_owned(), invalid_envelope);
        assert!(matches!(
            GraphDocument::from_workflow_bytes(&serde_json::to_vec(&invalid_schema)?),
            Err(GraphError::InvalidWidgetSchema { widget, .. }) if widget == "integer"
        ));

        let mut unsupported = serialized;
        let native = unsupported["nodes"]
            .as_array_mut()
            .and_then(|nodes| nodes.iter_mut().find(|node| node["id"] == "source"))
            .and_then(Value::as_object_mut)
            .and_then(|node| node.get_mut(NATIVE_WIDGETS_FIELD))
            .and_then(Value::as_object_mut)
            .ok_or("native metadata")?;
        native.insert("version".to_owned(), json!(2));
        assert!(matches!(
            GraphDocument::from_workflow_bytes(&serde_json::to_vec(&unsupported)?),
            Err(GraphError::InvalidWorkflow { path, .. }) if path.ends_with("zed:native-widgets.version")
        ));
        Ok(())
    }

    fn exercise_native_widget_prompt_projection() -> Result<Vec<u8>, Box<dyn Error>> {
        let mut document = fixture_engine()?.document;
        document
            .root
            .nodes
            .get_mut(&GraphIdentifier::from("source"))
            .ok_or("source")?
            .widgets = vec![GraphWidget {
            identifier: "text".to_owned(),
            kind: GraphWidgetKind::Text { multiline: false },
            value: json!("workflow-value"),
            prompt_value: json!("prompt-value"),
            validation: WidgetValidation::Valid,
            converted_to_input: false,
            visible: true,
            unknown: BTreeMap::new(),
        }];
        let workflow_bytes = document.to_workflow_bytes()?;
        let workflow = WorkflowFormatDocument::parse(&workflow_bytes)?;
        let descriptors = BTreeMap::from([(
            "Fixture".to_owned(),
            NodeDescriptor {
                type_name: "Fixture".to_owned(),
                display_name: "Fixture".to_owned(),
                inputs: vec![PortDescriptor {
                    name: "text".to_owned(),
                    type_name: "STRING".to_owned(),
                    required: true,
                }],
                outputs: vec![PortDescriptor {
                    name: "output".to_owned(),
                    type_name: "IMAGE".to_owned(),
                    required: true,
                }],
            },
        )]);
        let submission = crate::graph_to_prompt(&workflow, &descriptors, "test")?;
        assert_eq!(
            submission.prompt.0[&sim_types::NodeId("source".to_owned())].inputs["text"],
            "prompt-value"
        );
        Ok(serde_json::to_vec(&submission)?)
    }

    #[test]
    fn native_widget_prompt_projection_uses_prompt_value() -> Result<(), Box<dyn Error>> {
        exercise_native_widget_prompt_projection()?;
        Ok(())
    }

    #[test]
    fn subgraph_widget_exposure_preserves_typed_instance_state_and_reports_dynamic_inputs()
    -> Result<(), Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        let mut definition = simple_definition("widget-definition");
        let internal_identifier = GraphIdentifier::from("widget-definition-internal");
        let internal = definition
            .graph
            .nodes
            .get_mut(&internal_identifier)
            .ok_or("internal")?;
        internal.widgets.push(integer_widget("seed", 5));
        internal.inputs.push(GraphPort::new(
            "seed",
            GraphPortType::Concrete("INT".to_owned()),
        ));
        engine
            .document
            .root
            .definitions
            .insert(definition.identifier.clone(), definition);
        let mut instance = node("widget-instance", None, None);
        instance.subgraph_definition = Some(GraphIdentifier::from("widget-definition"));
        engine
            .document
            .root
            .nodes
            .insert(instance.identifier.clone(), instance);
        engine.apply(GraphCommand::SetSubgraphWidgetExposure {
            definition_identifier: GraphIdentifier::from("widget-definition"),
            internal_node: internal_identifier.clone(),
            widget: "seed".to_owned(),
            exposed: true,
        })?;
        let exposure = engine.document.root.definitions
            [&GraphIdentifier::from("widget-definition")]
            .exposed_widgets[0]
            .clone();
        assert!(matches!(
            exposure.widget.kind,
            GraphWidgetKind::Integer { .. }
        ));
        let instance_widget = engine.document.root.nodes[&GraphIdentifier::from("widget-instance")]
            .widgets
            .iter()
            .find(|widget| widget.identifier == exposure.identifier)
            .ok_or("instance widget")?;
        assert_eq!(instance_widget.value, json!(5));
        let instance_widget = engine
            .document
            .root
            .nodes
            .get_mut(&GraphIdentifier::from("widget-instance"))
            .and_then(|node| {
                node.widgets
                    .iter_mut()
                    .find(|widget| widget.identifier == exposure.identifier)
            })
            .ok_or("mutable instance widget")?;
        instance_widget.value = json!(7);
        instance_widget.prompt_value = json!(9);
        engine.apply(GraphCommand::SetSubgraphWidgetExposure {
            definition_identifier: GraphIdentifier::from("widget-definition"),
            internal_node: internal_identifier.clone(),
            widget: "seed".to_owned(),
            exposed: true,
        })?;
        let instance_widget = engine.document.root.nodes[&GraphIdentifier::from("widget-instance")]
            .widgets
            .iter()
            .find(|widget| widget.identifier == exposure.identifier)
            .ok_or("updated instance widget")?;
        assert_eq!(instance_widget.value, json!(7));
        assert_eq!(instance_widget.prompt_value, json!(9));
        let restored = GraphDocument::from_workflow_bytes(&engine.document.to_workflow_bytes()?)?;
        let restored_definition = restored
            .root
            .definitions
            .values()
            .find(|definition| definition.name == "widget-definition definition")
            .ok_or("restored widget definition")?;
        assert_eq!(
            restored_definition.exposed_widgets[0].widget.kind,
            exposure.widget.kind
        );
        let restored_instance_widget = restored.root.nodes
            [&GraphIdentifier::from("widget-instance")]
            .widgets
            .iter()
            .find(|widget| widget.identifier == exposure.identifier)
            .ok_or("restored instance widget")?;
        assert_eq!(restored_instance_widget.value, json!(7));
        assert_eq!(restored_instance_widget.prompt_value, json!(9));

        engine
            .document
            .root
            .definitions
            .get_mut(&GraphIdentifier::from("widget-definition"))
            .and_then(|definition| definition.graph.nodes.get_mut(&internal_identifier))
            .and_then(|node| node.inputs.first_mut())
            .ok_or("internal input")?
            .dynamic = true;
        let before_dynamic = engine.document.clone();
        assert!(matches!(
            engine.apply(GraphCommand::SetSubgraphWidgetExposure {
                definition_identifier: GraphIdentifier::from("widget-definition"),
                internal_node: internal_identifier,
                widget: "seed".to_owned(),
                exposed: true,
            }),
            Err(GraphError::UnsupportedDynamicSubgraphInput { .. })
        ));
        assert_eq!(engine.document, before_dynamic);
        Ok(())
    }

    #[test]
    fn subgraph_definition_removal_requires_explicit_instance_lifecycle()
    -> Result<(), Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        let definition = simple_definition("removable-definition");
        engine
            .document
            .root
            .definitions
            .insert(definition.identifier.clone(), definition);
        let mut instance = node("removable-instance", None, None);
        instance.subgraph_definition = Some(GraphIdentifier::from("removable-definition"));
        engine
            .document
            .root
            .nodes
            .insert(instance.identifier.clone(), instance);
        let before = engine.document.clone();
        assert!(matches!(
            engine.apply(GraphCommand::RemoveSubgraphDefinition {
                definition_identifier: GraphIdentifier::from("removable-definition"),
                remove_instances: false,
            }),
            Err(GraphError::SubgraphDefinitionInUse { .. })
        ));
        assert_eq!(engine.document, before);
        engine.apply(GraphCommand::RemoveSubgraphDefinition {
            definition_identifier: GraphIdentifier::from("removable-definition"),
            remove_instances: true,
        })?;
        assert!(
            !engine
                .document
                .root
                .nodes
                .contains_key(&GraphIdentifier::from("removable-instance"))
        );
        assert!(
            !engine
                .document
                .root
                .definitions
                .contains_key(&GraphIdentifier::from("removable-definition"))
        );
        Ok(())
    }

    #[test]
    fn group_membership_is_derived_on_import_mutation_and_serialization()
    -> Result<(), Box<dyn Error>> {
        let source = br#"{
          "id":"8f7bd92e-0d48-542e-945b-079f03ec3d80","revision":1,"version":1,
          "state":{"lastGroupId":1,"lastNodeId":2,"lastLinkId":0,"lastRerouteId":0},
          "nodes":[
            {"id":1,"type":"Fixture","pos":[10,10],"size":[20,20],"flags":{},"order":0,"mode":0,"properties":{},"inputs":[],"outputs":[]},
            {"id":2,"type":"Fixture","pos":[200,10],"size":[20,20],"flags":{},"order":1,"mode":0,"properties":{},"inputs":[],"outputs":[]}
          ],"links":[],
          "groups":[{"id":1,"title":"Derived","bounding":[0,0,100,100],"nodes":[2,999]}],
          "reroutes":[],"config":{},"extra":{}
        }"#;
        let document = GraphDocument::from_workflow_bytes(source)?;
        let group_identifier = GraphIdentifier::Integer(1);
        assert_eq!(
            document.root.groups[&group_identifier].node_ids,
            BTreeSet::from([GraphIdentifier::Integer(1)])
        );
        assert!(
            !document.root.groups[&group_identifier]
                .source_fields
                .contains_key("nodes")
        );

        let serialized = document.to_workflow_value()?;
        assert!(serialized["groups"][0].get("nodes").is_none());

        let mut engine = GraphCommandEngine::new(document)?;
        engine.apply(GraphCommand::AddNodesToGroup {
            identifier: group_identifier.clone(),
            nodes: BTreeSet::from([GraphIdentifier::Integer(2)]),
            padding: 5.0,
        })?;
        let group = &engine.document.root.groups[&group_identifier];
        assert_eq!(
            group.node_ids,
            BTreeSet::from([GraphIdentifier::Integer(1), GraphIdentifier::Integer(2)])
        );
        assert_eq!(group.bounds.origin, GraphPoint { x: 5.0, y: 5.0 });
        assert_eq!(
            group.bounds.size,
            GraphSize {
                width: 220.0,
                height: 30.0
            }
        );

        engine.apply(GraphCommand::SetSelection {
            selection: GraphSelection {
                nodes: BTreeSet::from([GraphIdentifier::Integer(2)]),
                ..GraphSelection::default()
            },
            mode: SelectionMode::Replace,
        })?;
        engine.apply(GraphCommand::MoveSelection {
            delta: GraphPoint { x: 500.0, y: 0.0 },
            snap: None,
        })?;
        assert_eq!(
            engine.document.root.groups[&group_identifier].node_ids,
            BTreeSet::from([GraphIdentifier::Integer(1)])
        );
        Ok(())
    }

    #[test]
    fn command_validation_is_non_mutating_and_enforces_hidden_invariants()
    -> Result<(), Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        let source = GraphIdentifier::from("source");
        let before = engine.document.clone();
        assert_eq!(
            engine.validate_command(&GraphCommand::RenameNode {
                identifier: source.clone(),
                title: " \n ".to_owned(),
            }),
            Err(GraphError::InvalidGraphLabel)
        );
        assert_eq!(engine.document, before);

        assert_eq!(
            engine.validate_command(&GraphCommand::SetNodeAdvancedVisibility {
                identifiers: BTreeSet::from([source.clone()]),
                visible: true,
            }),
            Err(GraphError::NodeHasNoAdvancedWidgets(source.clone()))
        );
        assert_eq!(engine.document, before);

        assert_eq!(
            engine.validate_command(&GraphCommand::SetNodeMode {
                identifiers: BTreeSet::new(),
                mode: GraphNodeMode::Never,
            }),
            Err(GraphError::EmptySelection)
        );
        assert_eq!(engine.document, before);

        engine.apply(GraphCommand::RenameNode {
            identifier: source.clone(),
            title: "  Normalized node  ".to_owned(),
        })?;
        assert_eq!(engine.document.root.nodes[&source].title, "Normalized node");

        let group_identifier = GraphIdentifier::from("validation-group");
        engine.apply(GraphCommand::CreateGroup {
            group: GraphGroup {
                identifier: group_identifier.clone(),
                title: "Validation group".to_owned(),
                bounds: GraphRect {
                    origin: GraphPoint {
                        x: -400.0,
                        y: -50.0,
                    },
                    size: GraphSize {
                        width: 400.0,
                        height: 220.0,
                    },
                },
                node_ids: BTreeSet::from([GraphIdentifier::from("target")]),
                collapsed: false,
                pinned: false,
                color: None,
                source_fields: Map::new(),
            },
        })?;
        assert_eq!(
            engine.document.root.groups[&group_identifier].node_ids,
            BTreeSet::from([source])
        );
        let before_invalid_group = engine.document.clone();
        assert_eq!(
            engine.validate_command(&GraphCommand::AddNodesToGroup {
                identifier: group_identifier,
                nodes: BTreeSet::from([GraphIdentifier::from("target")]),
                padding: -1.0,
            }),
            Err(GraphError::InvalidGroupPadding(-1.0))
        );
        assert_eq!(engine.document, before_invalid_group);
        Ok(())
    }

    #[test]
    fn no_op_commands_do_not_create_history_entries() -> Result<(), Box<dyn Error>> {
        let mut engine = fixture_engine()?;
        engine.apply(GraphCommand::SetNodeMode {
            identifiers: BTreeSet::from([GraphIdentifier::from("source")]),
            mode: GraphNodeMode::Always,
        })?;
        assert!(!engine.undo());
        Ok(())
    }

    #[test]
    fn node_modes_preserve_legacy_aliases_and_exact_workflow_wire_values()
    -> Result<(), Box<dyn Error>> {
        for (legacy, expected) in [
            ("active", GraphNodeMode::Always),
            ("disabled", GraphNodeMode::OnEvent),
            ("muted", GraphNodeMode::Never),
            ("bypassed", GraphNodeMode::Bypass),
        ] {
            assert_eq!(
                serde_json::from_value::<GraphNodeMode>(json!(legacy))?,
                expected
            );
        }
        for (wire, expected) in [
            (0, GraphNodeMode::Always),
            (1, GraphNodeMode::OnEvent),
            (2, GraphNodeMode::Never),
            (3, GraphNodeMode::OnTrigger),
            (4, GraphNodeMode::Bypass),
        ] {
            let parsed = parse_node(&json!({
                "id": wire + 1,
                "type": "Fixture",
                "pos": [0, 0],
                "size": [20, 20],
                "mode": wire,
                "inputs": [],
                "outputs": []
            }))?;
            assert_eq!(parsed.mode, expected);
            let graph = GraphLevel {
                nodes: BTreeMap::from([(parsed.identifier.clone(), parsed.clone())]),
                ..GraphLevel::default()
            };
            assert_eq!(
                serialize_node(&parsed, &graph, 0, Some(WorkflowFormat::Schema1), None)?["mode"],
                json!(wire)
            );
        }
        Ok(())
    }

    #[test]
    fn subgraph_slot_commands_validate_and_preserve_boundary_semantics()
    -> Result<(), Box<dyn Error>> {
        let source = br#"{
          "id":"8f7bd92e-0d48-542e-945b-079f03ec3d80","revision":1,"version":1,
          "state":{"lastGroupId":0,"lastNodeId":0,"lastLinkId":0,"lastRerouteId":0},
          "nodes":[],"links":[],"groups":[],"reroutes":[],
          "definitions":{"subgraphs":[{
            "id":"1cb1644a-b9d7-5f3e-a907-15c5720f8729","revision":1,"version":1,"name":"Slots",
            "state":{"lastGroupId":0,"lastNodeId":1,"lastLinkId":1,"lastRerouteId":0},
            "inputNode":{"id":-10,"bounding":[0,0,120,60]},
            "outputNode":{"id":-20,"bounding":[500,0,120,60]},
            "inputs":[{"id":"bf3be1d0-d92c-561a-a947-d371d9801587","name":"image","type":"IMAGE","linkIds":[1]}],
            "outputs":[],"widgets":[],
            "nodes":[{"id":1,"type":"PreviewImage","pos":[200,0],"size":[240,120],"flags":{},"order":0,"mode":0,"properties":{},"inputs":[{"name":"images","type":"IMAGE","link":1}],"outputs":[]}],
            "links":[{"id":1,"origin_id":-10,"origin_slot":0,"target_id":1,"target_slot":0,"type":"IMAGE"}],
            "groups":[],"reroutes":[],"config":{},"extra":{}
          }]},"config":{},"extra":{}
        }"#;
        let mut document = GraphDocument::from_workflow_bytes(source)?;
        let definition_identifier = GraphIdentifier::from("1cb1644a-b9d7-5f3e-a907-15c5720f8729");
        document.navigation.push(definition_identifier);
        let mut engine = GraphCommandEngine::new(document)?;

        assert_eq!(
            engine.validate_command(&GraphCommand::RenameSubgraphSlot {
                direction: GraphSlotDirection::Input,
                slot: 0,
                name: "\u{0000}".to_owned(),
            }),
            Err(GraphError::InvalidGraphLabel)
        );
        engine.apply(GraphCommand::RenameSubgraphSlot {
            direction: GraphSlotDirection::Input,
            slot: 0,
            name: "  pixels  ".to_owned(),
        })?;
        assert_eq!(
            engine.document.active_subgraph_definition()?.inputs[0].name,
            "pixels"
        );

        engine.apply(GraphCommand::DisconnectSubgraphSlot {
            direction: GraphSlotDirection::Input,
            slot: 0,
        })?;
        let definition = engine.document.active_subgraph_definition()?;
        assert!(definition.inputs[0].internal_node.is_none());
        assert!(definition.graph.links.is_empty());
        assert!(matches!(
            engine.validate_command(&GraphCommand::DisconnectSubgraphSlot {
                direction: GraphSlotDirection::Input,
                slot: 0,
            }),
            Err(GraphError::SubgraphSlotHasNoLinks { .. })
        ));

        engine.apply(GraphCommand::RemoveSubgraphSlot {
            direction: GraphSlotDirection::Input,
            slot: 0,
        })?;
        assert!(
            engine
                .document
                .active_subgraph_definition()?
                .inputs
                .is_empty()
        );
        assert!(matches!(
            engine.validate_command(&GraphCommand::RemoveSubgraphSlot {
                direction: GraphSlotDirection::Input,
                slot: 0,
            }),
            Err(GraphError::UnknownSubgraphPort { .. })
        ));
        Ok(())
    }

    #[test]
    fn imported_node_operation_restrictions_are_enforced_by_commands() -> Result<(), Box<dyn Error>>
    {
        let source = br#"{
          "id":"8f7bd92e-0d48-542e-945b-079f03ec3d80","revision":1,"version":1,
          "state":{"lastGroupId":0,"lastNodeId":2,"lastLinkId":0,"lastRerouteId":0},
          "nodes":[
            {"id":1,"type":"Restricted","pos":[0,0],"size":[120,80],"flags":{},"order":0,"mode":0,"properties":{},"inputs":[],"outputs":[],"resizable":false,"collapsable":false,"clonable":false,"removable":false},
            {"id":2,"type":"Blocked","pos":[200,0],"size":[120,80],"flags":{},"order":1,"mode":0,"properties":{},"inputs":[],"outputs":[],"block_delete":true}
          ],"links":[],"groups":[],"reroutes":[],"config":{},"extra":{}
        }"#;
        let document = GraphDocument::from_workflow_bytes(source)?;
        let restricted = GraphIdentifier::Integer(1);
        let blocked = GraphIdentifier::Integer(2);
        let node = &document.root.nodes[&restricted];
        assert!(!node.is_resizable());
        assert!(!node.is_collapsible());
        assert!(!node.is_clonable());
        assert!(!node.is_removable());
        assert!(document.root.nodes[&blocked].blocks_deletion());

        let mut engine = GraphCommandEngine::new(document)?;
        let before = engine.document.clone();
        for (command, operation) in [
            (
                GraphCommand::ResizeNode {
                    identifier: restricted.clone(),
                    size: GraphSize {
                        width: 240.0,
                        height: 160.0,
                    },
                },
                "resize",
            ),
            (
                GraphCommand::ToggleNodes {
                    identifiers: BTreeSet::from([restricted.clone()]),
                    toggle: NodeToggle::Collapse,
                },
                "collapse",
            ),
            (
                GraphCommand::DuplicateSelection {
                    selection: GraphSelection {
                        nodes: BTreeSet::from([restricted.clone()]),
                        ..GraphSelection::default()
                    },
                    offset: GraphPoint { x: 24.0, y: 24.0 },
                },
                "clone",
            ),
            (
                GraphCommand::RemoveItems {
                    selection: GraphSelection {
                        nodes: BTreeSet::from([restricted.clone()]),
                        ..GraphSelection::default()
                    },
                },
                "remove",
            ),
        ] {
            assert_eq!(
                engine.validate_command(&command),
                Err(GraphError::NodeOperationRestricted {
                    node: restricted.clone(),
                    operation,
                })
            );
            assert_eq!(engine.document, before);
        }
        assert_eq!(
            engine.validate_command(&GraphCommand::RemoveItems {
                selection: GraphSelection {
                    nodes: BTreeSet::from([blocked.clone()]),
                    ..GraphSelection::default()
                },
            }),
            Err(GraphError::NodeOperationRestricted {
                node: blocked,
                operation: "remove",
            })
        );
        assert_eq!(engine.document, before);

        engine.apply(GraphCommand::SetSelection {
            selection: GraphSelection {
                nodes: BTreeSet::from([restricted.clone()]),
                ..GraphSelection::default()
            },
            mode: SelectionMode::Replace,
        })?;
        assert_eq!(
            GraphClipboard::copy(&engine.document),
            Err(GraphError::NodeOperationRestricted {
                node: restricted,
                operation: "clone",
            })
        );
        Ok(())
    }

    #[test]
    fn subgraph_conversion_validation_matches_selection_semantics() -> Result<(), Box<dyn Error>> {
        let mut document = fixture_engine()?.document;
        let definition = simple_definition("existing-definition");
        document
            .root
            .definitions
            .insert(definition.identifier.clone(), definition);
        let mut instance = node("existing-instance", None, None);
        instance.subgraph_definition = Some(GraphIdentifier::from("existing-definition"));
        document
            .root
            .nodes
            .insert(instance.identifier.clone(), instance);
        document.root.selection.nodes =
            BTreeSet::from([GraphIdentifier::from("existing-instance")]);
        let mut engine = GraphCommandEngine::new(document)?;
        assert_eq!(
            engine
                .document
                .root
                .validate_subgraph_conversion_selection(),
            Err(GraphError::InvalidSubgraphConversion)
        );
        assert_eq!(
            engine.validate_command(&GraphCommand::ConvertSelectionToSubgraph {
                definition_identifier: GraphIdentifier::from("new-definition"),
                instance_identifier: GraphIdentifier::from("new-instance"),
                name: "Nested".to_owned(),
            }),
            Err(GraphError::InvalidSubgraphConversion)
        );

        engine.apply(GraphCommand::SetSelection {
            selection: GraphSelection {
                nodes: BTreeSet::from([
                    GraphIdentifier::from("existing-instance"),
                    GraphIdentifier::from("source"),
                ]),
                ..GraphSelection::default()
            },
            mode: SelectionMode::Replace,
        })?;
        engine
            .document
            .root
            .validate_subgraph_conversion_selection()?;
        engine.validate_command(&GraphCommand::ConvertSelectionToSubgraph {
            definition_identifier: GraphIdentifier::from("new-definition"),
            instance_identifier: GraphIdentifier::from("new-instance"),
            name: "Nested".to_owned(),
        })?;
        assert_eq!(
            engine.validate_command(&GraphCommand::ConvertSelectionToSubgraph {
                definition_identifier: GraphIdentifier::from("same-identifier"),
                instance_identifier: GraphIdentifier::from("same-identifier"),
                name: "Nested".to_owned(),
            }),
            Err(GraphError::InvalidSubgraphConversion)
        );
        assert_eq!(
            engine.validate_command(&GraphCommand::ConvertSelectionToSubgraph {
                definition_identifier: GraphIdentifier::from("new-definition"),
                instance_identifier: GraphIdentifier::from("new-instance"),
                name: " \n ".to_owned(),
            }),
            Err(GraphError::InvalidGraphLabel)
        );
        Ok(())
    }

    #[test]
    fn reroute_types_resolve_from_migrated_links_and_parent_chains() -> Result<(), Box<dyn Error>> {
        for (name, source) in [
            (
                "single_connected",
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/legacy/single_connected.json"
                ),
            ),
            (
                "branching",
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/legacy/branching.json"
                ),
            ),
            (
                "floating",
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/legacy/floating.json"
                ),
            ),
            (
                "floating_branch",
                include_str!(
                    "../../sim_test_support/fixtures/frontend_reroute/legacy/floating_branch.json"
                ),
            ),
        ] {
            let source_document = WorkflowFormatDocument::parse(source.as_bytes())?;
            let migrated = crate::apply_workflow_migrations(
                &source_document,
                &[crate::WorkflowMigrationId::LegacyRerouteNative],
            )?;
            let document = GraphDocument::from_workflow(&migrated)
                .map_err(|error| format!("{name} graph import failed: {error}"))?;
            assert!(!document.root.reroutes.is_empty(), "{name}");
            for identifier in document.root.reroutes.keys() {
                assert_eq!(
                    document.root.resolve_reroute_port_type(identifier)?,
                    GraphPortType::Concrete("VAE".to_owned()),
                    "{name} reroute {identifier:?}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn round_shape_is_command_validated_and_serde_compatible() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            serde_json::from_value::<GraphVisualShape>(json!("round"))?,
            GraphVisualShape::Round
        );
        assert_eq!(
            serde_json::from_value::<GraphVisualShape>(json!("Round"))?,
            GraphVisualShape::Round
        );
        assert_eq!(GraphVisualShape::Round.source_name(), "round");

        let mut engine = fixture_engine()?;
        let command = GraphCommand::SetNodeShape {
            identifiers: BTreeSet::from([GraphIdentifier::from("source")]),
            shape: GraphVisualShape::Round,
        };
        let before = engine.document.clone();
        engine.validate_command(&command)?;
        assert_eq!(engine.document, before);
        engine.apply(command)?;
        assert_eq!(
            engine.document.root.nodes[&GraphIdentifier::from("source")].source_fields["shape"],
            json!("round")
        );
        let restored = GraphCommandEngine::decode(&engine.encode()?)?;
        assert_eq!(restored.document, engine.document);
        Ok(())
    }
}
