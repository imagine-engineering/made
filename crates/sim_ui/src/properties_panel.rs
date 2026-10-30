use crate::{
    GraphWorkspaceItem, NativeImageImportBinding, PreparedNativeImageImport,
    ToggleGraphPropertiesPanel,
};
use anyhow::anyhow;
use editor::Editor;
use gpui::{
    App, AppContext as _, AsyncWindowContext, Context, Entity, EventEmitter, FocusHandle,
    Focusable, IntoElement, PathPromptOptions, Pixels, PromptLevel, Render, Role, Subscription,
    Task, WeakEntity, Window, px,
};
use serde_json::{Map, Value};
use sim_runtime::{
    AssetNamespace, ContentRevision, GraphCommand, GraphIdentifier, GraphNode, GraphNodeMode,
    GraphPaletteColor, GraphSelection, GraphWidgetKind,
};
use sim_types::CancellationToken;
use std::{
    collections::{BTreeSet, HashSet},
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};
use thiserror::Error;
use ui::{Button, ButtonCommon, ButtonStyle, Color, IconName, TintColor, prelude::*};
use uuid::Uuid;
use workspace::{
    Toast, Workspace,
    dock::{DockPosition, Panel, PanelEvent},
    notifications::NotificationId,
};

const IMAGE_IMPORT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

fn load_image_upload_declared() -> Result<bool, &'static str> {
    static DECLARED: OnceLock<Result<bool, String>> = OnceLock::new();
    match DECLARED.get_or_init(|| {
        let bindings =
            sim_runtime::native_image_catalog_bindings().map_err(|error| error.to_string())?;
        Ok(bindings.get("LoadImage").is_some_and(|binding| {
            binding.native.inputs.iter().any(|input| {
                input.name == "image"
                    && input.type_name == "STRING"
                    && input.choices_from_input_assets
                    && input.options.get("image_upload").and_then(Value::as_bool) == Some(true)
            })
        }))
    }) {
        Ok(declared) => Ok(*declared),
        Err(error) => {
            log::error!("native image upload descriptor is unavailable: {error}");
            Err("The native image upload descriptor is unavailable.")
        }
    }
}

pub(crate) fn load_image_import_widget(node: &GraphNode) -> Option<&str> {
    if node.type_identifier != "LoadImage" || load_image_upload_declared() != Ok(true) {
        return None;
    }
    node.widgets.iter().find_map(|widget| {
        (widget.identifier == "image"
            && widget.visible
            && !widget.converted_to_input
            && match &widget.kind {
                GraphWidgetKind::Text { .. } | GraphWidgetKind::Combo { dynamic: true, .. } => true,
                GraphWidgetKind::Preserved { .. } => {
                    widget.value.is_string() && widget.prompt_value.is_string()
                }
                _ => false,
            })
        .then_some(widget.identifier.as_str())
    })
}

pub(crate) fn read_bounded_image_import(
    reader: impl Read,
    maximum_bytes: u64,
    cancellation: &CancellationToken,
) -> anyhow::Result<Vec<u8>> {
    cancellation.check()?;
    let limit = maximum_bytes
        .checked_add(1)
        .ok_or_else(|| anyhow!("Image input limit is invalid."))?;
    let mut bytes = Vec::new();
    let mut reader = reader.take(limit);
    let mut chunk = [0_u8; 16 * 1024];
    loop {
        cancellation.check()?;
        let count = reader.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        let chunk = chunk
            .get(..count)
            .ok_or_else(|| anyhow!("The image reader returned an invalid byte count."))?;
        anyhow::ensure!(
            (bytes.len() as u64).saturating_add(chunk.len() as u64) <= maximum_bytes,
            "The selected image exceeds the input byte limit."
        );
        bytes.try_reserve(chunk.len())?;
        bytes.extend_from_slice(chunk);
    }
    cancellation.check()?;
    Ok(bytes)
}

async fn prepare_selected_image_import(
    file_system: Arc<dyn fs::Fs>,
    selected: PathBuf,
    maximum_bytes: u64,
    cancellation: CancellationToken,
) -> anyhow::Result<PreparedNativeImageImport> {
    cancellation.check()?;
    let filename = selected
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("The selected image has no supported filename."))?
        .to_owned();
    let selected = file_system.canonicalize(&selected).await?;
    let parent = selected
        .parent()
        .ok_or_else(|| anyhow!("The selected image has no parent directory."))?;
    let relative_name = selected
        .file_name()
        .ok_or_else(|| anyhow!("The selected image has no filename."))?;
    let directory = file_system
        .metadata(parent)
        .await?
        .ok_or_else(|| anyhow!("The selected image directory is unavailable."))?;
    let file = file_system
        .metadata(&selected)
        .await?
        .ok_or_else(|| anyhow!("The selected image is unavailable."))?;
    anyhow::ensure!(
        directory.is_dir && !directory.is_symlink,
        "The selected image directory changed."
    );
    anyhow::ensure!(
        !file.is_dir && !file.is_fifo && !file.is_symlink,
        "Select a regular image file."
    );
    anyhow::ensure!(
        file.len <= maximum_bytes,
        "The selected image exceeds the input byte limit."
    );
    cancellation.check()?;
    let reader = file_system
        .open_sync_confined_file(
            parent,
            Path::new(relative_name),
            directory.inode,
            file.inode,
        )
        .await?;
    let bytes = read_bounded_image_import(reader, maximum_bytes, &cancellation)?;
    Ok(PreparedNativeImageImport::new(filename, bytes))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ImageImportTarget {
    graph: gpui::EntityId,
    document: Uuid,
    profile: Uuid,
    navigation: Vec<GraphIdentifier>,
    node: GraphIdentifier,
    widget: String,
    revision: ContentRevision,
    binding: NativeImageImportBinding,
}

struct ImageImportOperation {
    identifier: Uuid,
    target: ImageImportTarget,
    cancellation: CancellationToken,
}

const GRAPH_PROPERTIES_PANEL_KEY: &str = "sim-graph-properties-panel";
const OPENROUTER_COST_NOTICE: &str = "Paid OpenRouter runs require approval of a USD cost estimate for each run. Actual charges may exceed the estimate. A key or account budget is a separate backstop.";

pub(crate) fn openrouter_cost_notice(node_type: &str) -> Option<&'static str> {
    (node_type == "OpenRouterLLMNode").then_some(OPENROUTER_COST_NOTICE)
}
const GRAPH_PROPERTIES_PANEL_UNAVAILABLE_NOTIFICATION_ID: &str =
    "sim-graph-properties-panel-unavailable";
const GRAPH_NODE_MODES: [GraphNodeMode; 5] = [
    GraphNodeMode::Always,
    GraphNodeMode::OnEvent,
    GraphNodeMode::Never,
    GraphNodeMode::OnTrigger,
    GraphNodeMode::Bypass,
];
const MAX_NODE_PROPERTY_DESCRIPTORS: usize = 256;
const MAX_NODE_PROPERTY_CHOICES: usize = 256;
const MAX_NODE_PROPERTY_LABEL_CHARACTERS: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GraphNodePropertyKind {
    Boolean,
    Number,
    Text,
    Choice {
        choices: Vec<GraphNodePropertyChoice>,
    },
    Json,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphNodePropertyChoice {
    pub label: String,
    pub value: Value,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphNodePropertyDescriptor {
    pub key: String,
    pub label: String,
    pub kind: GraphNodePropertyKind,
    pub value: Value,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum GraphNodePropertyAdapterError {
    #[error("node properties must be encoded as a JSON object")]
    InvalidProperties,
    #[error("node properties_info must be encoded as an array")]
    InvalidDescriptorList,
    #[error("node property descriptor {index} must be an object")]
    InvalidDescriptor { index: usize },
    #[error("node property descriptor {index} has no bounded property key")]
    InvalidKey { index: usize },
    #[error("node property map has an invalid key `{key}`")]
    InvalidMapKey { key: String },
    #[error("node property `{key}` has a duplicate descriptor")]
    DuplicateKey { key: String },
    #[error("node property `{key}` has no bounded label")]
    InvalidLabel { key: String },
    #[error("node property `{key}` declares an invalid choice list")]
    InvalidChoices { key: String },
    #[error("node property descriptor count exceeds 256")]
    TooManyDescriptors,
    #[error("node property `{key}` is not declared")]
    UnknownProperty { key: String },
    #[error("node property `{key}` does not accept {value}")]
    InvalidValue { key: String, value: String },
}

pub(crate) fn graph_node_properties(
    node: &GraphNode,
) -> Result<Map<String, Value>, GraphNodePropertyAdapterError> {
    match node.source_fields.get("properties") {
        Some(Value::Object(properties)) => Ok(properties.clone()),
        Some(_) => Err(GraphNodePropertyAdapterError::InvalidProperties),
        None => Ok(Map::new()),
    }
}

pub(crate) fn graph_node_property_descriptors(
    node: &GraphNode,
) -> Result<Vec<GraphNodePropertyDescriptor>, GraphNodePropertyAdapterError> {
    let properties = graph_node_properties(node)?;
    let mut descriptors = Vec::new();
    let mut described_keys = HashSet::new();
    if let Some(properties_info) = node.source_fields.get("properties_info") {
        let Value::Array(properties_info) = properties_info else {
            return Err(GraphNodePropertyAdapterError::InvalidDescriptorList);
        };
        if properties_info.len() > MAX_NODE_PROPERTY_DESCRIPTORS {
            return Err(GraphNodePropertyAdapterError::TooManyDescriptors);
        }
        for (index, descriptor) in properties_info.iter().enumerate() {
            let Value::Object(descriptor) = descriptor else {
                return Err(GraphNodePropertyAdapterError::InvalidDescriptor { index });
            };
            let key = descriptor
                .get("property")
                .or_else(|| descriptor.get("name"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|key| is_bounded_property_text(key))
                .ok_or(GraphNodePropertyAdapterError::InvalidKey { index })?
                .to_owned();
            if !described_keys.insert(key.clone()) {
                return Err(GraphNodePropertyAdapterError::DuplicateKey { key });
            }
            let label = descriptor
                .get("label")
                .or_else(|| descriptor.get("name"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|label| is_bounded_property_text(label))
                .ok_or_else(|| GraphNodePropertyAdapterError::InvalidLabel { key: key.clone() })?
                .to_owned();
            let property_type = descriptor
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("json")
                .to_ascii_lowercase();
            let choices = descriptor
                .get("values")
                .or_else(|| descriptor.get("options"));
            let kind = if let Some(choices) = choices {
                let choices = parse_property_choices(&key, choices)?;
                GraphNodePropertyKind::Choice { choices }
            } else {
                match property_type.as_str() {
                    "boolean" | "bool" => GraphNodePropertyKind::Boolean,
                    "number" | "float" | "integer" | "int" => GraphNodePropertyKind::Number,
                    "string" | "text" => GraphNodePropertyKind::Text,
                    "enum" | "combo" => {
                        return Err(GraphNodePropertyAdapterError::InvalidChoices { key });
                    }
                    _ => GraphNodePropertyKind::Json,
                }
            };
            let value = properties
                .get(&key)
                .cloned()
                .or_else(|| descriptor.get("default_value").cloned())
                .or_else(|| descriptor.get("default").cloned())
                .unwrap_or(Value::Null);
            if !value.is_null() {
                validate_graph_node_property_value(&key, &kind, &value)?;
            }
            descriptors.push(GraphNodePropertyDescriptor {
                key,
                label,
                kind,
                value,
            });
        }
    }
    let mut undescribed = properties
        .into_iter()
        .filter(|(key, _)| !described_keys.contains(key))
        .collect::<Vec<_>>();
    undescribed.sort_by(|left, right| left.0.cmp(&right.0));
    if descriptors.len().saturating_add(undescribed.len()) > MAX_NODE_PROPERTY_DESCRIPTORS {
        return Err(GraphNodePropertyAdapterError::TooManyDescriptors);
    }
    descriptors.extend(
        undescribed
            .into_iter()
            .map(|(key, value)| GraphNodePropertyDescriptor {
                label: key.clone(),
                key,
                kind: inferred_property_kind(&value),
                value,
            }),
    );
    Ok(descriptors)
}

pub(crate) fn parse_graph_node_property_value(
    descriptor: &GraphNodePropertyDescriptor,
    input: &str,
) -> Result<Value, GraphNodePropertyAdapterError> {
    let input = input.trim();
    let value = match &descriptor.kind {
        GraphNodePropertyKind::Boolean => match input {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => {
                return Err(GraphNodePropertyAdapterError::InvalidValue {
                    key: descriptor.key.clone(),
                    value: input.to_owned(),
                });
            }
        },
        GraphNodePropertyKind::Number => serde_json::from_str(input).map_err(|_| {
            GraphNodePropertyAdapterError::InvalidValue {
                key: descriptor.key.clone(),
                value: input.to_owned(),
            }
        })?,
        GraphNodePropertyKind::Text => Value::String(input.to_owned()),
        GraphNodePropertyKind::Choice { choices } => choices
            .iter()
            .find(|choice| choice.label == input)
            .map(|choice| choice.value.clone())
            .or_else(|| {
                serde_json::from_str::<Value>(input)
                    .ok()
                    .filter(|value| choices.iter().any(|choice| choice.value == *value))
            })
            .ok_or_else(|| GraphNodePropertyAdapterError::InvalidValue {
                key: descriptor.key.clone(),
                value: input.to_owned(),
            })?,
        GraphNodePropertyKind::Json => serde_json::from_str(input).map_err(|_| {
            GraphNodePropertyAdapterError::InvalidValue {
                key: descriptor.key.clone(),
                value: input.to_owned(),
            }
        })?,
    };
    validate_graph_node_property_value(&descriptor.key, &descriptor.kind, &value)?;
    Ok(value)
}

pub(crate) fn graph_node_properties_with_value(
    node: &GraphNode,
    key: &str,
    value: Value,
) -> Result<Map<String, Value>, GraphNodePropertyAdapterError> {
    let descriptor = graph_node_property_descriptors(node)?
        .into_iter()
        .find(|descriptor| descriptor.key == key)
        .ok_or_else(|| GraphNodePropertyAdapterError::UnknownProperty {
            key: key.to_owned(),
        })?;
    validate_graph_node_property_value(&descriptor.key, &descriptor.kind, &value)?;
    let mut properties = graph_node_properties(node)?;
    properties.insert(key.to_owned(), value);
    Ok(properties)
}

pub(crate) fn validate_graph_node_properties(
    node: &GraphNode,
    properties: Map<String, Value>,
) -> Result<Map<String, Value>, GraphNodePropertyAdapterError> {
    if properties.len() > MAX_NODE_PROPERTY_DESCRIPTORS {
        return Err(GraphNodePropertyAdapterError::TooManyDescriptors);
    }
    let descriptors = graph_node_property_descriptors(node)?;
    for (key, value) in &properties {
        if !is_bounded_property_text(key) {
            return Err(GraphNodePropertyAdapterError::InvalidMapKey { key: key.clone() });
        }
        if let Some(descriptor) = descriptors.iter().find(|descriptor| descriptor.key == *key) {
            validate_graph_node_property_value(key, &descriptor.kind, value)?;
        }
    }
    Ok(properties)
}

pub(crate) fn graph_node_property_value_label(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        other => other.to_string(),
    }
}

fn validate_graph_node_property_value(
    key: &str,
    kind: &GraphNodePropertyKind,
    value: &Value,
) -> Result<(), GraphNodePropertyAdapterError> {
    let valid = match kind {
        GraphNodePropertyKind::Boolean => value.is_boolean(),
        GraphNodePropertyKind::Number => value.is_number(),
        GraphNodePropertyKind::Text => value.is_string(),
        GraphNodePropertyKind::Choice { choices } => {
            choices.iter().any(|choice| choice.value == *value)
        }
        GraphNodePropertyKind::Json => true,
    };
    valid
        .then_some(())
        .ok_or_else(|| GraphNodePropertyAdapterError::InvalidValue {
            key: key.to_owned(),
            value: graph_node_property_value_label(value),
        })
}

fn parse_property_choices(
    key: &str,
    choices: &Value,
) -> Result<Vec<GraphNodePropertyChoice>, GraphNodePropertyAdapterError> {
    let choices = match choices {
        Value::Array(values) => values
            .iter()
            .map(|value| GraphNodePropertyChoice {
                label: graph_node_property_value_label(value),
                value: value.clone(),
            })
            .collect::<Vec<_>>(),
        Value::Object(values) => values
            .iter()
            .map(|(label, value)| GraphNodePropertyChoice {
                label: label.clone(),
                value: value.clone(),
            })
            .collect::<Vec<_>>(),
        _ => {
            return Err(GraphNodePropertyAdapterError::InvalidChoices {
                key: key.to_owned(),
            });
        }
    };
    if choices.is_empty() || choices.len() > MAX_NODE_PROPERTY_CHOICES {
        return Err(GraphNodePropertyAdapterError::InvalidChoices {
            key: key.to_owned(),
        });
    }
    let mut labels = HashSet::new();
    let mut values = Vec::new();
    for choice in &choices {
        if !is_bounded_property_text(&choice.label)
            || !labels.insert(choice.label.clone())
            || values.contains(&choice.value)
        {
            return Err(GraphNodePropertyAdapterError::InvalidChoices {
                key: key.to_owned(),
            });
        }
        values.push(choice.value.clone());
    }
    Ok(choices)
}

fn inferred_property_kind(value: &Value) -> GraphNodePropertyKind {
    match value {
        Value::Bool(_) => GraphNodePropertyKind::Boolean,
        Value::Number(_) => GraphNodePropertyKind::Number,
        Value::String(_) => GraphNodePropertyKind::Text,
        Value::Null | Value::Array(_) | Value::Object(_) => GraphNodePropertyKind::Json,
    }
}

fn is_bounded_property_text(value: &str) -> bool {
    !value.is_empty()
        && value.chars().count() <= MAX_NODE_PROPERTY_LABEL_CHARACTERS
        && !value.chars().any(char::is_control)
}

#[derive(Clone, Debug)]
struct BoundNodeSnapshot {
    identifier: GraphIdentifier,
    node: GraphNode,
    properties: Map<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BoundNodeAvailability {
    Available,
    NoTarget,
    GraphClosed,
    ReadOnly(String),
    Stale(GraphIdentifier),
    InvalidProperties(String),
}

impl BoundNodeAvailability {
    fn message(&self) -> Option<String> {
        match self {
            Self::Available => None,
            Self::NoTarget => Some("Select a graph node to inspect its properties.".to_owned()),
            Self::GraphClosed => Some("The requested native graph is no longer open.".to_owned()),
            Self::ReadOnly(diagnostic) => Some(format!(
                "Node properties are unavailable because the workflow is read-only: {diagnostic}"
            )),
            Self::Stale(identifier) => Some(format!(
                "The requested node `{identifier:?}` no longer exists in the active graph."
            )),
            Self::InvalidProperties(error) => Some(format!(
                "The requested node has invalid compatibility property metadata: {error}"
            )),
        }
    }
}

pub struct GraphPropertiesPanel {
    workspace: WeakEntity<Workspace>,
    graph: WeakEntity<GraphWorkspaceItem>,
    node_identifier: Option<GraphIdentifier>,
    title_editor: Entity<Editor>,
    properties_editor: Entity<Editor>,
    title_dirty: bool,
    properties_dirty: bool,
    synchronizing_editors: bool,
    loaded_target: Option<(gpui::EntityId, GraphIdentifier)>,
    focus_handle: FocusHandle,
    active: bool,
    status_message: Option<String>,
    confirmation_task: Option<Task<()>>,
    image_import_task: Option<Task<()>>,
    image_import_operation: Option<ImageImportOperation>,
    graph_subscription: Option<Subscription>,
    _subscriptions: Vec<Subscription>,
}

impl GraphPropertiesPanel {
    fn new(workspace: WeakEntity<Workspace>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let title_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("Node title", window, cx);
            editor
        });
        let properties_editor = cx.new(|cx| {
            let mut editor = Editor::multi_line(window, cx);
            editor.set_placeholder_text("JSON object containing node properties", window, cx);
            editor
        });
        let title_subscription = cx.subscribe(&title_editor, |this: &mut Self, _, event, cx| {
            if matches!(event, editor::EditorEvent::BufferEdited) && !this.synchronizing_editors {
                this.title_dirty = true;
                cx.notify();
            }
        });
        let properties_subscription =
            cx.subscribe(&properties_editor, |this: &mut Self, _, event, cx| {
                if matches!(event, editor::EditorEvent::BufferEdited) && !this.synchronizing_editors
                {
                    this.properties_dirty = true;
                    cx.notify();
                }
            });
        let workspace_subscription = workspace.upgrade().map(|workspace| {
            cx.observe(&workspace, |this: &mut Self, _, cx| {
                this.cancel_stale_image_import(cx);
                cx.notify();
            })
        });
        let mut subscriptions = vec![title_subscription, properties_subscription];
        subscriptions.extend(workspace_subscription);
        subscriptions.push(cx.on_release(|this, _| {
            if let Some(operation) = this.image_import_operation.take() {
                operation.cancellation.cancel();
            }
        }));
        subscriptions.push(
            cx.observe_global::<crate::GlobalNativeSubgraphCatalogRevision>(|this, cx| {
                this.cancel_stale_image_import(cx);
                cx.notify();
            }),
        );
        if let Some(model) = crate::execution_ui_model(cx) {
            subscriptions.push(cx.observe(&model, |this, _, cx| {
                this.cancel_stale_image_import(cx);
                cx.notify();
            }));
        }
        Self {
            workspace,
            graph: WeakEntity::new_invalid(),
            node_identifier: None,
            title_editor,
            properties_editor,
            title_dirty: false,
            properties_dirty: false,
            synchronizing_editors: false,
            loaded_target: None,
            focus_handle: cx.focus_handle(),
            active: false,
            status_message: None,
            confirmation_task: None,
            image_import_task: None,
            image_import_operation: None,
            graph_subscription: None,
            _subscriptions: subscriptions,
        }
    }

    pub fn load(
        workspace: WeakEntity<Workspace>,
        cx: &mut AsyncWindowContext,
    ) -> Task<anyhow::Result<Entity<Self>>> {
        cx.spawn(async move |cx| {
            workspace.update_in(cx, |_workspace, window, cx| {
                let workspace = cx.entity().downgrade();
                Ok(cx.new(|cx| Self::new(workspace, window, cx)))
            })?
        })
    }

    fn image_import_target(&self, cx: &App) -> anyhow::Result<ImageImportTarget> {
        let graph = self
            .graph
            .upgrade()
            .ok_or_else(|| anyhow!("The native graph is no longer open."))?;
        let workspace = self
            .workspace
            .upgrade()
            .ok_or_else(|| anyhow!("The native workspace is no longer open."))?;
        anyhow::ensure!(
            workspace
                .read(cx)
                .active_item_as::<GraphWorkspaceItem>(cx)
                .is_some_and(|active| active.entity_id() == graph.entity_id()),
            "The active graph changed."
        );
        let model = graph.read(cx).model();
        anyhow::ensure!(!model.is_read_only(), "The native graph is read-only.");
        let document = model
            .document()
            .ok_or_else(|| anyhow!("The native graph is unavailable."))?;
        let profile = document
            .profile_identity
            .ok_or_else(|| anyhow!("The native graph has no execution profile."))?;
        anyhow::ensure!(
            crate::initialized_profile_id(cx).is_some_and(|active| active.0 == profile),
            "The active execution profile changed."
        );
        let node = self
            .node_identifier
            .clone()
            .ok_or_else(|| anyhow!("Select a Load Image node."))?;
        let target = document
            .active_graph()?
            .nodes
            .get(&node)
            .ok_or_else(|| anyhow!("The image input node is no longer available."))?;
        let widget = load_image_import_widget(target)
            .ok_or_else(|| anyhow!("This node has no native image upload widget."))?;
        let services = crate::native_asset_services(cx)
            .ok_or_else(|| anyhow!("Native image import is unavailable."))?;
        let importer = services
            .image_importer()
            .ok_or_else(|| anyhow!("Native image import is unavailable."))?;
        let binding = importer.binding();
        anyhow::ensure!(
            Uuid::parse_str(binding.profile_id())? == profile
                && services.profile_id() == binding.profile_id(),
            "The image import profile changed."
        );
        Ok(ImageImportTarget {
            graph: graph.entity_id(),
            document: document.document_identity,
            profile,
            navigation: document.navigation.clone(),
            node,
            widget: widget.to_owned(),
            revision: ContentRevision::from_bytes(model.save_coordinator.local_bytes()),
            binding,
        })
    }

    fn cancel_image_import(&mut self, message: &str, cx: &mut Context<Self>) {
        if let Some(operation) = self.image_import_operation.take() {
            operation.cancellation.cancel();
            self.status_message = Some(message.to_owned());
            cx.notify();
        }
        if let Some(task) = self.image_import_task.take() {
            // Cancellation can be triggered by a stale completion inside this task.
            cx.defer(move |_| drop(task));
        }
    }

    fn cancel_stale_image_import(&mut self, cx: &mut Context<Self>) {
        if self
            .image_import_operation
            .as_ref()
            .is_some_and(|operation| {
                self.image_import_target(cx).as_ref().ok() != Some(&operation.target)
            })
        {
            self.cancel_image_import(
                "Image import cancelled because its graph or profile changed.",
                cx,
            );
        }
    }

    fn image_import_is_current(
        &self,
        identifier: Uuid,
        target: &ImageImportTarget,
        cx: &App,
    ) -> bool {
        self.image_import_operation
            .as_ref()
            .is_some_and(|operation| {
                operation.identifier == identifier
                    && !operation.cancellation.is_cancelled()
                    && &operation.target == target
                    && self.image_import_target(cx).as_ref().ok() == Some(target)
            })
    }

    fn start_image_import(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let target = match self.image_import_target(cx) {
            Ok(target) => target,
            Err(error) => {
                self.status_message = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        let Some(importer) =
            crate::native_asset_services(cx).and_then(|services| services.image_importer())
        else {
            return;
        };
        self.cancel_image_import("The previous image import was cancelled.", cx);
        let identifier = Uuid::new_v4();
        let cancellation = CancellationToken::default();
        self.image_import_operation = Some(ImageImportOperation {
            identifier,
            target: target.clone(),
            cancellation: cancellation.clone(),
        });
        self.status_message = Some("Choose a PNG image to import.".to_owned());
        let prompt = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Import PNG image".into()),
        });
        let file_system = <dyn fs::Fs>::global(cx);
        self.image_import_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result: anyhow::Result<Option<crate::ImportedNativeImage>> = async {
                let Some(paths) = prompt.await?? else {
                    return Ok(None);
                };
                let [path] = paths.as_slice() else {
                    return Err(anyhow!("Select exactly one image file."));
                };
                cancellation.check()?;
                anyhow::ensure!(
                    this.read_with(cx, |this, cx| this
                        .image_import_is_current(identifier, &target, cx))?,
                    "The image import target changed."
                );
                let read_cancellation = cancellation.clone();
                let prepare = cx.background_spawn(prepare_selected_image_import(
                    file_system,
                    path.clone(),
                    importer.maximum_input_bytes(),
                    read_cancellation,
                ));
                let timeout = cx.background_executor().timer(IMAGE_IMPORT_READ_TIMEOUT);
                let prepared = match futures::future::select(prepare, timeout).await {
                    futures::future::Either::Left((prepared, _)) => prepared?,
                    futures::future::Either::Right((_, _)) => {
                        cancellation.cancel();
                        return Err(anyhow!("Reading the selected image timed out."));
                    }
                };
                anyhow::ensure!(
                    this.read_with(cx, |this, cx| this
                        .image_import_is_current(identifier, &target, cx))?,
                    "The image import target changed."
                );
                cancellation.check()?;
                let import_cancellation = cancellation.clone();
                let imported = cx
                    .background_spawn(async move {
                        importer
                            .import(prepared, &import_cancellation)
                            .map_err(|failure| anyhow!("{}: {}", failure.code, failure.message))
                    })
                    .await?;
                Ok(Some(imported))
            }
            .await;
            if let Err(error) = this.update(cx, |this, cx| {
                this.finish_image_import(identifier, &target, result, cx);
            }) {
                cancellation.cancel();
                log::error!("native image import panel closed: {error}");
            }
        }));
        cx.notify();
    }

    fn finish_image_import(
        &mut self,
        identifier: Uuid,
        target: &ImageImportTarget,
        result: anyhow::Result<Option<crate::ImportedNativeImage>>,
        cx: &mut Context<Self>,
    ) {
        let Some(operation) = self.image_import_operation.as_ref() else {
            return;
        };
        if operation.identifier != identifier {
            return;
        }
        if self.image_import_target(cx).as_ref().ok() != Some(target) {
            self.cancel_stale_image_import(cx);
            return;
        }
        let cancelled = operation.cancellation.is_cancelled();
        self.image_import_operation = None;
        match result {
            Ok(Some(_)) if cancelled => {
                self.status_message = Some("Image import cancelled.".to_owned())
            }
            Ok(Some(imported)) => {
                if imported.identity().namespace != AssetNamespace::Input
                    || imported.identity().profile_id != target.binding.profile_id()
                    || imported.identity().relative_path.to_str() != Some(imported.source_name())
                {
                    self.status_message = Some(
                        "The image importer returned an incompatible asset identity.".to_owned(),
                    );
                } else {
                    self.apply_command(
                        GraphCommand::SetWidget {
                            node: target.node.clone(),
                            widget: target.widget.clone(),
                            value: Value::String(imported.source_name().to_owned()),
                        },
                        "Imported PNG image into the native graph.",
                        cx,
                    );
                }
            }
            Ok(None) => self.status_message = Some("Image selection cancelled.".to_owned()),
            Err(error) => self.status_message = Some(format!("Image import failed: {error}")),
        }
        cx.notify();
    }

    fn bind_node(
        &mut self,
        graph: Entity<GraphWorkspaceItem>,
        identifier: GraphIdentifier,
        cx: &mut Context<Self>,
    ) {
        self.cancel_image_import(
            "Image import cancelled because its node binding changed.",
            cx,
        );
        self.graph = graph.downgrade();
        self.node_identifier = Some(identifier);
        self.graph_subscription = Some(cx.observe(&graph, |this: &mut Self, _, cx| {
            this.cancel_stale_image_import(cx);
            cx.notify();
        }));
        self.loaded_target = None;
        self.title_dirty = false;
        self.properties_dirty = false;
        self.status_message = None;
        cx.notify();
    }

    fn follow_active_graph(&mut self, cx: &mut Context<Self>) {
        self.cancel_image_import(
            "Image import cancelled because its node binding changed.",
            cx,
        );
        let Some(workspace) = self.workspace.upgrade() else {
            self.graph = WeakEntity::new_invalid();
            self.node_identifier = None;
            self.loaded_target = None;
            self.status_message = Some("The native workspace is no longer available.".to_owned());
            cx.notify();
            return;
        };
        let graph = workspace.read(cx).active_item_as::<GraphWorkspaceItem>(cx);
        self.follow_graph(graph, cx);
    }

    fn follow_graph(&mut self, graph: Option<Entity<GraphWorkspaceItem>>, cx: &mut Context<Self>) {
        self.cancel_image_import(
            "Image import cancelled because its node binding changed.",
            cx,
        );
        let Some(graph) = graph else {
            self.graph = WeakEntity::new_invalid();
            self.node_identifier = None;
            self.loaded_target = None;
            self.status_message =
                Some("The active workspace item is not a native graph.".to_owned());
            cx.notify();
            return;
        };
        let selected = graph.read(cx).model().selection().and_then(|selection| {
            (selection.nodes.len() == 1)
                .then(|| selection.nodes.iter().next().cloned())
                .flatten()
        });
        match selected {
            Some(identifier) => self.bind_node(graph, identifier, cx),
            None => {
                self.graph = graph.downgrade();
                self.node_identifier = None;
                self.graph_subscription = Some(cx.observe(&graph, |this: &mut Self, _, cx| {
                    this.cancel_stale_image_import(cx);
                    cx.notify();
                }));
                self.loaded_target = None;
                self.status_message = Some(
                    "Select exactly one node in the active graph to inspect its properties."
                        .to_owned(),
                );
                cx.notify();
            }
        }
    }

    fn bound_snapshot(&self, cx: &App) -> (Option<BoundNodeSnapshot>, BoundNodeAvailability) {
        let Some(identifier) = self.node_identifier.clone() else {
            return (None, BoundNodeAvailability::NoTarget);
        };
        let Some(graph) = self.graph.upgrade() else {
            return (None, BoundNodeAvailability::GraphClosed);
        };
        let graph = graph.read(cx);
        if graph.model().is_read_only() {
            return (
                None,
                BoundNodeAvailability::ReadOnly(
                    graph
                        .model()
                        .read_only_diagnostic()
                        .unwrap_or("the workflow format could not be edited")
                        .to_owned(),
                ),
            );
        }
        let Some(node) = graph
            .model()
            .document()
            .and_then(|document| document.active_graph().ok())
            .and_then(|active_graph| active_graph.nodes.get(&identifier))
            .cloned()
        else {
            return (None, BoundNodeAvailability::Stale(identifier));
        };
        let properties = match graph_node_properties(&node) {
            Ok(properties) => properties,
            Err(error) => {
                return (
                    None,
                    BoundNodeAvailability::InvalidProperties(error.to_string()),
                );
            }
        };
        (
            Some(BoundNodeSnapshot {
                identifier,
                node,
                properties,
            }),
            BoundNodeAvailability::Available,
        )
    }

    fn synchronize_editors(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(graph) = self.graph.upgrade() else {
            return;
        };
        let (snapshot, _) = self.bound_snapshot(cx);
        let Some(snapshot) = snapshot else {
            return;
        };
        let target = (graph.entity_id(), snapshot.identifier.clone());
        if self.loaded_target.as_ref() == Some(&target) {
            return;
        }
        let properties = match serde_json::to_string_pretty(&snapshot.properties) {
            Ok(properties) => properties,
            Err(error) => {
                self.status_message = Some(format!(
                    "Failed to present the canonical node properties: {error}"
                ));
                cx.notify();
                return;
            }
        };
        self.synchronizing_editors = true;
        self.title_editor.update(cx, |editor, cx| {
            editor.set_text(snapshot.node.title, window, cx)
        });
        self.properties_editor
            .update(cx, |editor, cx| editor.set_text(properties, window, cx));
        self.synchronizing_editors = false;
        self.title_dirty = false;
        self.properties_dirty = false;
        self.loaded_target = Some(target);
    }

    fn target_command_unavailable_reason(
        &self,
        command: &GraphCommand,
        cx: &App,
    ) -> Option<String> {
        let Some(graph) = self.graph.upgrade() else {
            return Some("the requested native graph is no longer open".to_owned());
        };
        let graph = graph.read(cx);
        let Some(engine) = graph.model().engine() else {
            return Some(
                graph
                    .model()
                    .read_only_diagnostic()
                    .unwrap_or("the workflow is read-only")
                    .to_owned(),
            );
        };
        engine
            .validate_command(command)
            .err()
            .map(|error| error.to_string())
    }

    fn apply_command(
        &mut self,
        command: GraphCommand,
        success: &'static str,
        cx: &mut Context<Self>,
    ) -> bool {
        if let Some(reason) = self.target_command_unavailable_reason(&command, cx) {
            self.status_message = Some(format!("Cannot update node properties: {reason}"));
            cx.notify();
            return false;
        }
        let Some(graph) = self.graph.upgrade() else {
            self.status_message = Some("The requested native graph is no longer open.".to_owned());
            cx.notify();
            return false;
        };
        if graph.update(cx, |graph, cx| graph.apply_graph_command(command, cx)) {
            self.status_message = Some(success.to_owned());
            cx.notify();
            true
        } else {
            self.status_message =
                Some("The graph command engine rejected the node property update.".to_owned());
            cx.notify();
            false
        }
    }

    fn apply_title(&mut self, cx: &mut Context<Self>) {
        let Some(identifier) = self.node_identifier.clone() else {
            self.status_message = Some("No native node is selected.".to_owned());
            cx.notify();
            return;
        };
        let title = self.title_editor.read(cx).text(cx);
        if self.apply_command(
            GraphCommand::RenameNode { identifier, title },
            "Updated native node title.",
            cx,
        ) {
            self.title_dirty = false;
        }
    }

    fn apply_properties(&mut self, cx: &mut Context<Self>) {
        let Some(identifier) = self.node_identifier.clone() else {
            self.status_message = Some("No native node is selected.".to_owned());
            cx.notify();
            return;
        };
        let text = self.properties_editor.read(cx).text(cx);
        let properties = match serde_json::from_str::<Map<String, Value>>(&text) {
            Ok(properties) => {
                let (snapshot, _) = self.bound_snapshot(cx);
                let Some(snapshot) = snapshot.filter(|snapshot| snapshot.identifier == identifier)
                else {
                    self.status_message =
                        Some("The requested native node property target is stale.".to_owned());
                    cx.notify();
                    return;
                };
                match validate_graph_node_properties(&snapshot.node, properties) {
                    Ok(properties) => properties,
                    Err(error) => {
                        self.status_message = Some(format!(
                            "Node properties do not match their compatibility metadata: {error}"
                        ));
                        cx.notify();
                        return;
                    }
                }
            }
            Err(error) => {
                self.status_message = Some(format!(
                    "Node properties must be a JSON object before they can be applied: {error}"
                ));
                cx.notify();
                return;
            }
        };
        if self.apply_command(
            GraphCommand::SetNodeProperties {
                identifier,
                properties,
            },
            "Updated native node properties.",
            cx,
        ) {
            self.properties_dirty = false;
        }
    }

    fn set_mode(&mut self, mode: GraphNodeMode, cx: &mut Context<Self>) {
        let Some(identifier) = self.node_identifier.clone() else {
            self.status_message = Some("No native node is selected.".to_owned());
            cx.notify();
            return;
        };
        self.apply_command(
            GraphCommand::SetNodeMode {
                identifiers: BTreeSet::from([identifier]),
                mode,
            },
            "Updated native node mode.",
            cx,
        );
    }

    fn set_color(&mut self, color: Option<GraphPaletteColor>, cx: &mut Context<Self>) {
        let Some(identifier) = self.node_identifier.clone() else {
            self.status_message = Some("No native node is selected.".to_owned());
            cx.notify();
            return;
        };
        self.apply_command(
            GraphCommand::SetNodePalette {
                identifiers: BTreeSet::from([identifier]),
                color,
            },
            "Updated native node color.",
            cx,
        );
    }

    fn set_property_value(
        &mut self,
        identifier: GraphIdentifier,
        key: String,
        value: Value,
        cx: &mut Context<Self>,
    ) {
        let (snapshot, _) = self.bound_snapshot(cx);
        let Some(snapshot) = snapshot.filter(|snapshot| snapshot.identifier == identifier) else {
            self.status_message =
                Some("The requested native node property target is stale.".to_owned());
            cx.notify();
            return;
        };
        let descriptor =
            match graph_node_property_descriptors(&snapshot.node).and_then(|descriptors| {
                descriptors
                    .into_iter()
                    .find(|descriptor| descriptor.key == key)
                    .ok_or_else(|| GraphNodePropertyAdapterError::UnknownProperty {
                        key: key.clone(),
                    })
            }) {
                Ok(descriptor) => descriptor,
                Err(error) => {
                    self.status_message = Some(format!(
                        "Cannot update native node property `{key}`: {error}"
                    ));
                    cx.notify();
                    return;
                }
            };
        let value = match parse_graph_node_property_value(
            &descriptor,
            &graph_node_property_value_label(&value),
        ) {
            Ok(value) => value,
            Err(error) => {
                self.status_message = Some(format!(
                    "Cannot update native node property `{key}`: {error}"
                ));
                cx.notify();
                return;
            }
        };
        let properties = match graph_node_properties_with_value(&snapshot.node, &key, value) {
            Ok(properties) => properties,
            Err(error) => {
                self.status_message = Some(format!(
                    "Cannot update native node property `{key}`: {error}"
                ));
                cx.notify();
                return;
            }
        };
        if self.apply_command(
            GraphCommand::SetNodeProperties {
                identifier,
                properties,
            },
            "Updated native node property.",
            cx,
        ) {
            self.loaded_target = None;
        }
    }

    fn delete_command(&self) -> Option<GraphCommand> {
        self.node_identifier
            .clone()
            .map(|identifier| GraphCommand::RemoveItems {
                selection: GraphSelection {
                    nodes: BTreeSet::from([identifier]),
                    ..GraphSelection::default()
                },
            })
    }

    fn confirm_delete(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(command) = self.delete_command() else {
            self.status_message = Some("No native node is selected.".to_owned());
            cx.notify();
            return;
        };
        if let Some(reason) = self.target_command_unavailable_reason(&command, cx) {
            self.status_message = Some(format!("Cannot delete the native node: {reason}"));
            cx.notify();
            return;
        }
        let prompt = window.prompt(
            PromptLevel::Warning,
            "Delete native graph node?",
            Some(
                "This removes the node and its connected links in one undoable graph transaction.",
            ),
            &["Delete", "Cancel"],
            cx,
        );
        self.confirmation_task = Some(cx.spawn_in(window, async move |this, cx| {
            match prompt.await {
                Ok(0) => {
                    if let Err(error) = this.update(cx, |this, cx| {
                        if this.apply_command(command, "Deleted native graph node.", cx) {
                            this.loaded_target = None;
                        }
                    }) {
                        log::error!(
                            "native graph properties panel closed during deletion: {error}"
                        );
                    }
                }
                Ok(_) => {
                    if let Err(error) = this.update(cx, |this, cx| {
                        this.status_message = Some("Native node deletion cancelled.".to_owned());
                        cx.notify();
                    }) {
                        log::error!(
                            "native graph properties panel closed during cancellation: {error}"
                        );
                    }
                }
                Err(error) => {
                    if let Err(update_error) = this.update(cx, |this, cx| {
                        this.status_message =
                            Some(format!("Native node deletion confirmation failed: {error}"));
                        cx.notify();
                    }) {
                        log::error!(
                            "native graph properties panel closed after prompt failure: {update_error}"
                        );
                    }
                }
            }
        }));
    }

    fn render_mode_controls(
        &self,
        snapshot: &BoundNodeSnapshot,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        h_flex()
            .flex_wrap()
            .gap_1()
            .children(GRAPH_NODE_MODES.into_iter().map(|mode| {
                let command = GraphCommand::SetNodeMode {
                    identifiers: BTreeSet::from([snapshot.identifier.clone()]),
                    mode,
                };
                let unavailable = self.target_command_unavailable_reason(&command, cx);
                Button::new(mode_element_id(mode), mode_label(mode))
                    .style(if snapshot.node.mode == mode {
                        ButtonStyle::Tinted(TintColor::Accent)
                    } else {
                        ButtonStyle::Subtle
                    })
                    .disabled(unavailable.is_some())
                    .when_some(unavailable, |button, reason| {
                        button.tooltip(ui::Tooltip::text(reason))
                    })
                    .on_click(cx.listener(move |this, _, _, cx| this.set_mode(mode, cx)))
            }))
    }

    fn render_color_controls(
        &self,
        snapshot: &BoundNodeSnapshot,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let selected_palette = GraphPaletteColor::ALL
            .into_iter()
            .find(|color| snapshot.node.color.as_deref() == Some(color.node_header()));
        h_flex()
            .flex_wrap()
            .gap_1()
            .child(
                Button::new("sim-properties-color-default", "Default")
                    .style(if selected_palette.is_none() {
                        ButtonStyle::Tinted(TintColor::Accent)
                    } else {
                        ButtonStyle::Subtle
                    })
                    .on_click(cx.listener(|this, _, _, cx| this.set_color(None, cx))),
            )
            .children(GraphPaletteColor::ALL.into_iter().map(|color| {
                Button::new(color_element_id(color), color.label())
                    .style(if selected_palette == Some(color) {
                        ButtonStyle::Tinted(TintColor::Accent)
                    } else {
                        ButtonStyle::Subtle
                    })
                    .on_click(cx.listener(move |this, _, _, cx| this.set_color(Some(color), cx)))
            }))
    }

    fn render_property_descriptor(
        &self,
        identifier: GraphIdentifier,
        descriptor: GraphNodePropertyDescriptor,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let current_label = graph_node_property_value_label(&descriptor.value);
        let boolean_identifier = identifier.clone();
        v_flex()
            .id(format!("sim-properties-property-{}", descriptor.key))
            .gap_1()
            .child(
                h_flex()
                    .justify_between()
                    .child(
                        div()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child(descriptor.label.clone()),
                    )
                    .child(div().child(current_label)),
            )
            .when(
                matches!(descriptor.kind, GraphNodePropertyKind::Boolean),
                |this| {
                    let key = descriptor.key.clone();
                    let next = Value::Bool(!descriptor.value.as_bool().unwrap_or(false));
                    this.child(
                        Button::new(
                            format!("sim-properties-property-boolean-{key}"),
                            if next == Value::Bool(true) {
                                "Enable"
                            } else {
                                "Disable"
                            },
                        )
                        .style(ButtonStyle::Subtle)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.set_property_value(
                                boolean_identifier.clone(),
                                key.clone(),
                                next.clone(),
                                cx,
                            )
                        })),
                    )
                },
            )
            .when_some(
                match &descriptor.kind {
                    GraphNodePropertyKind::Choice { choices } => Some(choices.clone()),
                    _ => None,
                },
                |this, choices| {
                    this.child(
                        h_flex()
                            .flex_wrap()
                            .gap_1()
                            .children(choices.into_iter().map(|choice| {
                                let selected = choice.value == descriptor.value;
                                let label = choice.label;
                                let value = choice.value;
                                let identifier = identifier.clone();
                                let key = descriptor.key.clone();
                                Button::new(
                                    format!("sim-properties-property-choice-{key}-{label}"),
                                    label,
                                )
                                .style(if selected {
                                    ButtonStyle::Tinted(TintColor::Accent)
                                } else {
                                    ButtonStyle::Subtle
                                })
                                .on_click(cx.listener(
                                    move |this, _, _, cx| {
                                        this.set_property_value(
                                            identifier.clone(),
                                            key.clone(),
                                            value.clone(),
                                            cx,
                                        )
                                    },
                                ))
                            })),
                    )
                },
            )
            .into_any_element()
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn start_image_import_for_test(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.start_image_import(window, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn stage_image_import_for_test(
        &mut self,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<(Uuid, CancellationToken)> {
        let target = self.image_import_target(cx)?;
        let identifier = Uuid::new_v4();
        let cancellation = CancellationToken::default();
        self.image_import_operation = Some(ImageImportOperation {
            identifier,
            target,
            cancellation: cancellation.clone(),
        });
        Ok((identifier, cancellation))
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn finish_image_import_for_test(
        &mut self,
        identifier: Uuid,
        result: anyhow::Result<Option<crate::ImportedNativeImage>>,
        cx: &mut Context<Self>,
    ) {
        if let Some(target) = self
            .image_import_operation
            .as_ref()
            .map(|operation| operation.target.clone())
        {
            self.finish_image_import(identifier, &target, result, cx);
        }
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn cancel_image_import_for_test(&mut self, cx: &mut Context<Self>) {
        self.cancel_image_import("Image import cancelled.", cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn status_message_for_test(&self) -> Option<&str> {
        self.status_message.as_deref()
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn has_image_import_task_for_test(&self) -> bool {
        self.image_import_task.is_some()
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn test_new(
        workspace: WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new(workspace, window, cx)
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn set_title_for_test(
        &mut self,
        title: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.title_editor
            .update(cx, |editor, cx| editor.set_text(title, window, cx));
        self.apply_title(cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn set_properties_for_test(
        &mut self,
        properties: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.properties_editor
            .update(cx, |editor, cx| editor.set_text(properties, window, cx));
        self.apply_properties(cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn set_mode_for_test(&mut self, mode: GraphNodeMode, cx: &mut Context<Self>) {
        self.set_mode(mode, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn set_color_for_test(
        &mut self,
        color: Option<GraphPaletteColor>,
        cx: &mut Context<Self>,
    ) {
        self.set_color(color, cx);
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn target_for_test(&self) -> Option<GraphIdentifier> {
        self.node_identifier.clone()
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn availability_for_test(&self, cx: &App) -> Option<String> {
        self.bound_snapshot(cx).1.message()
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn delete_for_test(&mut self, cx: &mut Context<Self>) -> bool {
        let Some(command) = self.delete_command() else {
            return false;
        };
        self.apply_command(command, "Deleted native graph node.", cx)
    }
}

impl EventEmitter<PanelEvent> for GraphPropertiesPanel {}

impl Focusable for GraphPropertiesPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Panel for GraphPropertiesPanel {
    fn persistent_name() -> &'static str {
        "Native Graph Properties"
    }

    fn panel_key() -> &'static str {
        GRAPH_PROPERTIES_PANEL_KEY
    }

    fn position(&self, _window: &Window, _cx: &App) -> DockPosition {
        DockPosition::Right
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        position == DockPosition::Right
    }

    fn set_position(
        &mut self,
        position: DockPosition,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.position_is_valid(position) {
            self.status_message =
                Some("Native graph properties are owned by the right dock.".to_owned());
            cx.notify();
        }
    }

    fn default_size(&self, _window: &Window, _cx: &App) -> Pixels {
        px(360.)
    }

    fn min_size(&self, _window: &Window, _cx: &App) -> Option<Pixels> {
        Some(px(280.))
    }

    fn icon(&self, _window: &Window, _cx: &App) -> Option<IconName> {
        Some(IconName::Settings)
    }

    fn icon_tooltip(&self, _window: &Window, _cx: &App) -> Option<&'static str> {
        Some("Native Graph Properties")
    }

    fn toggle_action(&self) -> Box<dyn gpui::Action> {
        Box::new(ToggleGraphPropertiesPanel)
    }

    fn starts_open(&self, _window: &Window, _cx: &App) -> bool {
        false
    }

    fn set_active(&mut self, active: bool, window: &mut Window, cx: &mut Context<Self>) {
        self.active = active;
        if active && self.node_identifier.is_none() {
            // Dock activation holds the Workspace lease until this callback returns.
            cx.defer_in(window, |panel, _window, cx| {
                if panel.active && panel.node_identifier.is_none() {
                    panel.follow_active_graph(cx);
                }
            });
        }
        cx.notify();
    }

    fn activation_priority(&self) -> u32 {
        9
    }
}

impl Render for GraphPropertiesPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.cancel_stale_image_import(cx);
        self.synchronize_editors(window, cx);
        let (snapshot, availability) = self.bound_snapshot(cx);
        let unavailable_message = availability.message();
        let status_message = self.status_message.clone();

        v_flex()
            .id("sim-graph-properties-panel")
            .debug_selector(|| "SIM-GRAPH-PROPERTIES-PANEL".into())
            .key_context("SimGraphPropertiesPanel ComfyGraphPropertiesPanel")
            .role(Role::Complementary)
            .aria_label("Native graph node properties")
            .track_focus(&self.focus_handle)
            .size_full()
            .overflow_y_scroll()
            .gap_3()
            .p_3()
            .child(
                h_flex()
                    .justify_between()
                    .child(
                        div()
                            .text_ui_lg(cx)
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child("Node Properties"),
                    )
                    .child(
                        Button::new("sim-properties-follow-selection", "Follow Selection")
                            .style(ButtonStyle::Subtle)
                            .on_click(cx.listener(|this, _, _, cx| this.follow_active_graph(cx))),
                    ),
            )
            .when_some(unavailable_message, |this, message| {
                this.child(
                    div()
                        .id("sim-properties-unavailable")
                        .debug_selector(|| "SIM-PROPERTIES-UNAVAILABLE".into())
                        .role(Role::Alert)
                        .aria_label(message.clone())
                        .text_color(Color::Error.color(cx))
                        .child(message),
                )
            })
            .when_some(snapshot, |this, snapshot| {
                let rename_command = GraphCommand::RenameNode {
                    identifier: snapshot.identifier.clone(),
                    title: self.title_editor.read(cx).text(cx),
                };
                let rename_unavailable =
                    self.target_command_unavailable_reason(&rename_command, cx);
                let properties = self.properties_editor.read(cx).text(cx);
                let property_command = serde_json::from_str::<Map<String, Value>>(&properties)
                    .ok()
                    .map(|properties| GraphCommand::SetNodeProperties {
                        identifier: snapshot.identifier.clone(),
                        properties,
                    });
                let property_unavailable = property_command
                    .as_ref()
                    .and_then(|command| self.target_command_unavailable_reason(command, cx))
                    .or_else(|| {
                        property_command.is_none().then(|| {
                            "node properties must be a JSON object before they can be applied"
                                .to_owned()
                        })
                    });
                let delete_unavailable = self
                    .delete_command()
                    .as_ref()
                    .and_then(|command| self.target_command_unavailable_reason(command, cx));
                let descriptors = graph_node_property_descriptors(&snapshot.node);
                let image_upload = load_image_import_widget(&snapshot.node).is_some();
                let image_import_unavailable = self
                    .image_import_target(cx)
                    .err()
                    .map(|error| error.to_string());
                this.when(image_upload, |this| {
                    this.child(
                        h_flex()
                            .gap_2()
                            .child(
                                div().debug_selector(|| "SIM-IMPORT-IMAGE".into()).child(
                                    Button::new("sim-properties-import-image", "Import PNG")
                                        .disabled(
                                            image_import_unavailable.is_some()
                                                || self.image_import_operation.is_some(),
                                        )
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.start_image_import(window, cx)
                                        })),
                                ),
                            )
                            .when(self.image_import_operation.is_some(), |this| {
                                this.child(
                                    Button::new(
                                        "sim-properties-cancel-image-import",
                                        "Cancel Import",
                                    )
                                    .on_click(cx.listener(
                                        |this, _, _, cx| {
                                            this.cancel_image_import("Image import cancelled.", cx)
                                        },
                                    )),
                                )
                            }),
                    )
                    .when_some(image_import_unavailable, |this, message| {
                        this.child(
                            div()
                                .id("sim-properties-image-import-status")
                                .role(Role::Status)
                                .child(message),
                        )
                    })
                })
                .child(format!("Type: {}", snapshot.node.type_identifier))
                .when_some(
                    openrouter_cost_notice(&snapshot.node.type_identifier),
                    |this, notice| {
                        this.child(
                            div()
                                .id("sim-openrouter-cost-notice")
                                .role(Role::Status)
                                .aria_label(notice)
                                .child(notice),
                        )
                    },
                )
                .child(format!(
                    "{} inputs · {} outputs · {} widgets",
                    snapshot.node.inputs.len(),
                    snapshot.node.outputs.len(),
                    snapshot.node.widgets.len()
                ))
                .child(div().font_weight(gpui::FontWeight::SEMIBOLD).child("Title"))
                .child(self.title_editor.clone())
                .child(
                    h_flex().justify_end().child(
                        Button::new("sim-properties-apply-title", "Apply Title")
                            .style(ButtonStyle::Filled)
                            .disabled(rename_unavailable.is_some())
                            .on_click(cx.listener(|this, _, _, cx| this.apply_title(cx))),
                    ),
                )
                .child(div().font_weight(gpui::FontWeight::SEMIBOLD).child("Mode"))
                .child(self.render_mode_controls(&snapshot, cx))
                .child(div().font_weight(gpui::FontWeight::SEMIBOLD).child("Color"))
                .child(self.render_color_controls(&snapshot, cx))
                .child(
                    div()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child("Typed Properties"),
                )
                .child(match descriptors {
                    Ok(descriptors) if descriptors.is_empty() => div()
                        .id("sim-properties-no-typed-properties")
                        .role(Role::Status)
                        .child("This node declares no typed properties.")
                        .into_any_element(),
                    Ok(descriptors) => v_flex()
                        .gap_2()
                        .children(descriptors.into_iter().map(|descriptor| {
                            self.render_property_descriptor(
                                snapshot.identifier.clone(),
                                descriptor,
                                cx,
                            )
                        }))
                        .into_any_element(),
                    Err(error) => div()
                        .id("sim-properties-invalid-typed-properties")
                        .role(Role::Alert)
                        .text_color(Color::Error.color(cx))
                        .child(format!("Invalid typed property metadata: {error}"))
                        .into_any_element(),
                })
                .child(
                    div()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child("Properties (JSON)"),
                )
                .child(div().min_h_32().child(self.properties_editor.clone()))
                .child(
                    h_flex().justify_end().child(
                        Button::new("sim-properties-apply-json", "Apply Properties")
                            .style(ButtonStyle::Filled)
                            .disabled(property_unavailable.is_some())
                            .on_click(cx.listener(|this, _, _, cx| this.apply_properties(cx))),
                    ),
                )
                .child(
                    h_flex().justify_end().child(
                        Button::new("sim-properties-delete", "Delete Node")
                            .style(ButtonStyle::Tinted(TintColor::Error))
                            .disabled(delete_unavailable.is_some())
                            .on_click(
                                cx.listener(|this, _, window, cx| this.confirm_delete(window, cx)),
                            ),
                    ),
                )
            })
            .when_some(status_message, |this, message| {
                this.child(
                    div()
                        .id("sim-properties-status")
                        .role(Role::Status)
                        .aria_label(message.clone())
                        .child(message),
                )
            })
    }
}

fn mode_label(mode: GraphNodeMode) -> &'static str {
    match mode {
        GraphNodeMode::Always => "Always",
        GraphNodeMode::OnEvent => "On Event",
        GraphNodeMode::Never => "Never",
        GraphNodeMode::OnTrigger => "On Trigger",
        GraphNodeMode::Bypass => "Bypass",
    }
}

fn mode_element_id(mode: GraphNodeMode) -> &'static str {
    match mode {
        GraphNodeMode::Always => "sim-properties-mode-always",
        GraphNodeMode::OnEvent => "sim-properties-mode-event",
        GraphNodeMode::Never => "sim-properties-mode-never",
        GraphNodeMode::OnTrigger => "sim-properties-mode-trigger",
        GraphNodeMode::Bypass => "sim-properties-mode-bypass",
    }
}

fn color_element_id(color: GraphPaletteColor) -> &'static str {
    match color {
        GraphPaletteColor::Red => "sim-properties-color-red",
        GraphPaletteColor::Brown => "sim-properties-color-brown",
        GraphPaletteColor::Green => "sim-properties-color-green",
        GraphPaletteColor::Blue => "sim-properties-color-blue",
        GraphPaletteColor::PaleBlue => "sim-properties-color-pale-blue",
        GraphPaletteColor::Cyan => "sim-properties-color-cyan",
        GraphPaletteColor::Purple => "sim-properties-color-purple",
        GraphPaletteColor::Yellow => "sim-properties-color-yellow",
        GraphPaletteColor::Black => "sim-properties-color-black",
    }
}

pub fn open_for_graph_node(
    workspace: &mut Workspace,
    graph: Entity<GraphWorkspaceItem>,
    identifier: GraphIdentifier,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) -> anyhow::Result<()> {
    let panel = workspace
        .panel::<GraphPropertiesPanel>(cx)
        .ok_or_else(|| anyhow!("native graph properties panel is not registered"))?;
    panel.update(cx, |panel, cx| panel.bind_node(graph, identifier, cx));
    workspace.reveal_panel::<GraphPropertiesPanel>(window, cx);
    workspace.focus_panel::<GraphPropertiesPanel>(window, cx);
    Ok(())
}

fn graph_properties_panel_for_action(
    workspace: &mut Workspace,
    cx: &mut Context<Workspace>,
) -> Option<Entity<GraphPropertiesPanel>> {
    let panel = workspace.panel::<GraphPropertiesPanel>(cx);
    if panel.is_none() {
        workspace.show_toast(
            Toast::new(
                NotificationId::named(GRAPH_PROPERTIES_PANEL_UNAVAILABLE_NOTIFICATION_ID.into()),
                "The native graph properties panel is not available",
            )
            .autohide(),
            cx,
        );
    }
    panel
}

pub fn init_graph_properties_panel(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(|workspace, _: &ToggleGraphPropertiesPanel, window, cx| {
            if let Some(panel) = graph_properties_panel_for_action(workspace, cx) {
                let graph = workspace.active_item_as::<GraphWorkspaceItem>(cx);
                panel.update(cx, |panel, cx| panel.follow_graph(graph, cx));
                workspace.toggle_panel_focus::<GraphPropertiesPanel>(window, cx);
            }
        });
    })
    .detach();
}
