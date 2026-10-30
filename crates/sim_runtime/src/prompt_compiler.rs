use crate::{NativeProviderRegistryPin, cache::canonical_json, executor::NativeNodeRegistry};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_nodes::{
    NativeEffectClass, NativeNodeBindingDisposition, NativeNodeDescriptor, NativePortCardinality,
    NativePrimitive, NativePrimitiveType, NativeTypeUnion, NativeValue, NativeValueType,
};
use sim_types::{ApiPrompt, NodeId, PromptId, PromptNode, PromptSubmission};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;
use uuid::Uuid;

pub const MAX_PROMPT_NODES: usize = 100_000;
pub const MAX_PROMPT_INPUTS: usize = 1_000_000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InputBinding {
    Literal {
        value: NativeValue,
    },
    Link {
        source: NodeId,
        output_index: usize,
        lazy: bool,
        mode: NativePortCardinality,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledNode {
    pub id: NodeId,
    pub class_type: String,
    pub descriptor: NativeNodeDescriptor,
    pub inputs: BTreeMap<String, InputBinding>,
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledPlan {
    pub prompt_id: PromptId,
    pub client_id: Option<String>,
    pub prompt_number: Option<f64>,
    pub extra_data: BTreeMap<String, Value>,
    pub unknown: BTreeMap<String, Value>,
    pub nodes: BTreeMap<NodeId, CompiledNode>,
    pub topological_order: Vec<NodeId>,
    pub static_required_nodes: BTreeSet<NodeId>,
    pub output_nodes: Vec<NodeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_execution: Option<NativeProviderPlanIdentity>,
    #[serde(default, flatten)]
    pub persistence_unknown_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeProviderPlanIdentity {
    registry: NativeProviderRegistryPin,
    compiled_plan_sha256: String,
}

impl NativeProviderPlanIdentity {
    pub fn registry(&self) -> &NativeProviderRegistryPin {
        &self.registry
    }

    pub fn compiled_plan_sha256(&self) -> &str {
        &self.compiled_plan_sha256
    }
}

impl CompiledPlan {
    pub fn provider_registry_pin(&self) -> Option<&NativeProviderRegistryPin> {
        self.provider_execution
            .as_ref()
            .map(|identity| identity.registry())
    }

    pub fn requires_provider_registry(&self) -> bool {
        self.nodes
            .values()
            .any(|node| node.descriptor.effect == NativeEffectClass::Provider)
    }

    pub fn validate_provider_execution_identity(&self) -> Result<(), PromptCompileError> {
        match (self.requires_provider_registry(), &self.provider_execution) {
            (false, None) => Ok(()),
            (false, Some(_)) => Err(PromptCompileError::UnexpectedProviderRegistryPin),
            (true, None) => Err(PromptCompileError::ProviderRegistryPinRequired),
            (true, Some(identity)) => {
                identity
                    .registry
                    .validate()
                    .map_err(|_| PromptCompileError::InvalidProviderRegistryPin)?;
                let expected = provider_plan_digest(self, &identity.registry)?;
                if expected != identity.compiled_plan_sha256 {
                    return Err(PromptCompileError::ProviderPlanIdentityMismatch);
                }
                Ok(())
            }
        }
    }

    pub fn validate_integrity(&self) -> Result<(), PromptCompileError> {
        self.validate_provider_execution_identity()?;
        for (node_id, node) in &self.nodes {
            if node.id != *node_id
                || node.class_type != node.descriptor.class_type
                || node.descriptor.validate().is_err()
            {
                return Err(PromptCompileError::InvalidCompiledPlan);
            }
            for input in &node.descriptor.inputs {
                if input.required && !input.hidden && !node.inputs.contains_key(&input.name) {
                    return Err(PromptCompileError::InvalidCompiledPlan);
                }
            }
            validate_compiled_structured_inputs(node)?;
            for (name, binding) in &node.inputs {
                let (input, source_schema) = if name.contains('.') {
                    resolve_compiled_structured_input(node, name)
                        .ok_or(PromptCompileError::InvalidCompiledPlan)?
                } else {
                    (
                        resolve_input_descriptor(&node.descriptor, name)
                            .ok_or(PromptCompileError::InvalidCompiledPlan)?,
                        resolve_input_schema(&node.descriptor, name),
                    )
                };
                match binding {
                    InputBinding::Literal { value } => {
                        value
                            .validate()
                            .map_err(|_| PromptCompileError::InvalidCompiledPlan)?;
                        let cardinality_matches = match input.cardinality {
                            NativePortCardinality::List => {
                                matches!(value, NativeValue::List { values } if values.iter().all(|value| input.accepted_types.accepts(value)))
                            }
                            NativePortCardinality::Mapped => match value {
                                NativeValue::List { values } => values
                                    .iter()
                                    .all(|value| input.accepted_types.accepts(value)),
                                value => input.accepted_types.accepts(value),
                            },
                            NativePortCardinality::Scalar => {
                                !matches!(value, NativeValue::List { .. })
                                    && input.accepted_types.accepts(value)
                            }
                        };
                        let schema_matches = source_schema.as_ref().is_none_or(|schema| {
                            structured_selector_matches(schema, value)
                                || node.descriptor.source_schema.as_ref().is_some_and(
                                    |source_schema| {
                                        sim_nodes::native_value_matches_source_input_schema(
                                            value,
                                            &node.class_type,
                                            source_schema,
                                            schema,
                                        )
                                    },
                                )
                        });
                        if !cardinality_matches || !schema_matches {
                            return Err(PromptCompileError::InvalidCompiledPlan);
                        }
                    }
                    InputBinding::Link {
                        source,
                        output_index,
                        lazy,
                        mode,
                    } => {
                        let source_node = self
                            .nodes
                            .get(source)
                            .ok_or(PromptCompileError::InvalidCompiledPlan)?;
                        let output = source_node
                            .descriptor
                            .outputs
                            .get(*output_index)
                            .ok_or(PromptCompileError::InvalidCompiledPlan)?;
                        if *lazy != input.lazy
                            || *mode != input.cardinality
                            || !type_union_accepts_output(
                                &input.accepted_types,
                                &output.produced_type,
                            )
                            || (output.is_list
                                && input.cardinality == NativePortCardinality::Scalar)
                        {
                            return Err(PromptCompileError::InvalidCompiledPlan);
                        }
                    }
                }
            }
        }
        let output_nodes = self
            .nodes
            .values()
            .filter(|node| node.descriptor.output_node)
            .map(|node| node.id.clone())
            .collect::<Vec<_>>();
        if output_nodes != self.output_nodes
            || topological_order(&self.nodes)? != self.topological_order
            || static_required_nodes(&self.nodes, &self.output_nodes) != self.static_required_nodes
        {
            return Err(PromptCompileError::InvalidCompiledPlan);
        }
        Ok(())
    }

    fn seal_provider_registry(
        &mut self,
        registry: NativeProviderRegistryPin,
    ) -> Result<(), PromptCompileError> {
        registry
            .validate()
            .map_err(|_| PromptCompileError::InvalidProviderRegistryPin)?;
        self.provider_execution = Some(NativeProviderPlanIdentity {
            compiled_plan_sha256: provider_plan_digest(self, &registry)?,
            registry,
        });
        self.validate_provider_execution_identity()
    }
}

fn provider_plan_digest(
    plan: &CompiledPlan,
    registry: &NativeProviderRegistryPin,
) -> Result<String, PromptCompileError> {
    let mut value = serde_json::to_value(plan)
        .map_err(|_| PromptCompileError::ProviderPlanSerializationFailed)?;
    let Value::Object(fields) = &mut value else {
        return Err(PromptCompileError::ProviderPlanSerializationFailed);
    };
    fields.remove("provider_execution");
    let canonical =
        canonical_json(&value).map_err(|_| PromptCompileError::ProviderPlanSerializationFailed)?;
    let mut digest = Sha256::new();
    digest.update(b"zed-native-provider-plan-v1\0");
    digest.update(registry.identity_sha256().as_bytes());
    digest.update([0]);
    digest.update(canonical.as_bytes());
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PromptCompileError {
    #[error("prompt contains {actual} nodes, exceeding the {maximum} node limit")]
    TooManyNodes { actual: usize, maximum: usize },
    #[error("prompt contains more than {maximum} inputs")]
    TooManyInputs { maximum: usize },
    #[error("runtime descriptor `{0}` is invalid")]
    InvalidRuntimeDescriptor(String),
    #[error("runtime descriptor `{0}` is already registered")]
    DuplicateRuntimeDescriptor(String),
    #[error("node {node:?} references unknown class type `{class_type}`")]
    UnknownNodeType { node: NodeId, class_type: String },
    #[error("node {node:?} class type `{class_type}` is unavailable: {reason}")]
    UnavailableNode {
        node: NodeId,
        class_type: String,
        reason: String,
    },
    #[error("node {node:?} is missing required input `{input}`")]
    MissingInput { node: NodeId, input: String },
    #[error("node {node:?} supplied unknown input `{input}`")]
    UnknownInput { node: NodeId, input: String },
    #[error("node {node:?} input `{input}` has an invalid literal")]
    InvalidLiteral { node: NodeId, input: String },
    #[error("node {node:?} requests unsupported hidden input `{input}`")]
    UnsupportedHiddenInput { node: NodeId, input: String },
    #[error("node {node:?} input `{input}` links to unknown node {source_node:?}")]
    UnknownLink {
        node: NodeId,
        input: String,
        source_node: NodeId,
    },
    #[error(
        "node {node:?} input `{input}` links to missing output {output_index} on {source_node:?}"
    )]
    UnknownOutput {
        node: NodeId,
        input: String,
        source_node: NodeId,
        output_index: usize,
    },
    #[error(
        "node {node:?} input `{input}` is incompatible with {source_node:?} output {output_index}"
    )]
    IncompatibleLink {
        node: NodeId,
        input: String,
        source_node: NodeId,
        output_index: usize,
    },
    #[error("prompt has no reachable output node")]
    NoOutputNode,
    #[error("prompt graph contains a cycle involving {0:?}")]
    Cycle(Vec<NodeId>),
    #[error("provider-backed prompt compilation requires a selected provider registry pin")]
    ProviderRegistryPinRequired,
    #[error("provider registry pin is invalid")]
    InvalidProviderRegistryPin,
    #[error("local-only prompt unexpectedly carries a provider registry pin")]
    UnexpectedProviderRegistryPin,
    #[error("provider plan identity does not match the compiled plan")]
    ProviderPlanIdentityMismatch,
    #[error("provider plan identity could not be serialized canonically")]
    ProviderPlanSerializationFailed,
    #[error("compiled prompt plan integrity validation failed")]
    InvalidCompiledPlan,
}

pub struct PromptCompiler<'a> {
    registry: &'a NativeNodeRegistry,
    provider_registry_pin: Option<NativeProviderRegistryPin>,
}

impl<'a> PromptCompiler<'a> {
    pub fn new(registry: &'a NativeNodeRegistry) -> Self {
        Self {
            registry,
            provider_registry_pin: None,
        }
    }

    pub fn with_provider_registry_pin(
        mut self,
        provider_registry_pin: NativeProviderRegistryPin,
    ) -> Result<Self, PromptCompileError> {
        provider_registry_pin
            .validate()
            .map_err(|_| PromptCompileError::InvalidProviderRegistryPin)?;
        self.provider_registry_pin = Some(provider_registry_pin);
        Ok(self)
    }

    pub fn compile(
        &self,
        submission: PromptSubmission,
    ) -> Result<CompiledPlan, PromptCompileError> {
        if submission.prompt.0.len() > MAX_PROMPT_NODES {
            return Err(PromptCompileError::TooManyNodes {
                actual: submission.prompt.0.len(),
                maximum: MAX_PROMPT_NODES,
            });
        }
        submission
            .prompt
            .0
            .values()
            .try_fold(0_usize, |count, node| count.checked_add(node.inputs.len()))
            .filter(|count| *count <= MAX_PROMPT_INPUTS)
            .ok_or(PromptCompileError::TooManyInputs {
                maximum: MAX_PROMPT_INPUTS,
            })?;
        let descriptors = self.resolve_descriptors(&submission.prompt)?;
        let mut nodes = BTreeMap::new();
        for (node_id, prompt_node) in &submission.prompt.0 {
            let descriptor =
                descriptors
                    .get(node_id)
                    .ok_or_else(|| PromptCompileError::UnknownNodeType {
                        node: node_id.clone(),
                        class_type: prompt_node.class_type.clone(),
                    })?;
            let mut inputs = compile_inputs(
                node_id,
                prompt_node,
                descriptor,
                &submission.prompt,
                &descriptors,
            )?;
            inject_hidden_inputs(
                node_id,
                descriptor,
                &submission.prompt,
                &submission.extra_data,
                &mut inputs,
            )?;
            nodes.insert(
                node_id.clone(),
                CompiledNode {
                    id: node_id.clone(),
                    class_type: prompt_node.class_type.clone(),
                    descriptor: (*descriptor).clone(),
                    inputs,
                    unknown: prompt_node.unknown.clone(),
                },
            );
        }
        let output_nodes = nodes
            .values()
            .filter(|node| node.descriptor.output_node)
            .map(|node| node.id.clone())
            .collect::<Vec<_>>();
        if output_nodes.is_empty() {
            return Err(PromptCompileError::NoOutputNode);
        }
        let topological_order = topological_order(&nodes)?;
        let static_required_nodes = static_required_nodes(&nodes, &output_nodes);
        let provider_required = nodes.values().any(|node| {
            node.descriptor.effect == NativeEffectClass::Provider
                || self.registry.binding_declared_disposition(&node.class_type)
                    == Some(NativeNodeBindingDisposition::ProviderRequired)
        });
        let mut plan = CompiledPlan {
            prompt_id: submission
                .prompt_id
                .unwrap_or_else(|| PromptId(Uuid::new_v4())),
            client_id: submission.client_id,
            prompt_number: submission.number,
            extra_data: submission.extra_data,
            unknown: submission.unknown,
            nodes,
            topological_order,
            static_required_nodes,
            output_nodes,
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        };
        if provider_required {
            let pin = self
                .provider_registry_pin
                .clone()
                .ok_or(PromptCompileError::ProviderRegistryPinRequired)?;
            plan.seal_provider_registry(pin)?;
        }
        plan.validate_integrity()?;
        Ok(plan)
    }

    fn resolve_descriptors<'b>(
        &'b self,
        prompt: &ApiPrompt,
    ) -> Result<BTreeMap<NodeId, &'a NativeNodeDescriptor>, PromptCompileError> {
        let mut descriptors = BTreeMap::new();
        for (node_id, node) in &prompt.0 {
            let descriptor = self.registry.descriptor(&node.class_type).ok_or_else(|| {
                PromptCompileError::UnknownNodeType {
                    node: node_id.clone(),
                    class_type: node.class_type.clone(),
                }
            })?;
            if let Some(reason) = self.registry.unavailable_reason(&node.class_type) {
                return Err(PromptCompileError::UnavailableNode {
                    node: node_id.clone(),
                    class_type: node.class_type.clone(),
                    reason: reason.to_owned(),
                });
            }
            descriptors.insert(node_id.clone(), descriptor);
        }
        Ok(descriptors)
    }
}

fn compile_inputs(
    node_id: &NodeId,
    prompt_node: &PromptNode,
    descriptor: &NativeNodeDescriptor,
    prompt: &ApiPrompt,
    descriptors: &BTreeMap<NodeId, &NativeNodeDescriptor>,
) -> Result<BTreeMap<String, InputBinding>, PromptCompileError> {
    for input in &descriptor.inputs {
        if input.required && !input.hidden && !prompt_node.inputs.contains_key(&input.name) {
            return Err(PromptCompileError::MissingInput {
                node: node_id.clone(),
                input: input.name.clone(),
            });
        }
    }
    for (dynamic_index, dynamic) in descriptor.dynamic_inputs.iter().enumerate() {
        for name in required_dynamic_input_names(descriptor, dynamic_index, dynamic)? {
            if dynamic.input.required
                && !dynamic.input.hidden
                && !prompt_node.inputs.contains_key(&name)
            {
                return Err(PromptCompileError::MissingInput {
                    node: node_id.clone(),
                    input: name,
                });
            }
        }
    }
    validate_required_structured_inputs(node_id, prompt_node, descriptor)?;
    let mut compiled = BTreeMap::new();
    for (name, value) in &prompt_node.inputs {
        let (input, source_schema) = if name.contains('.') {
            resolve_active_structured_input(node_id, descriptor, prompt_node, name)?.ok_or_else(
                || PromptCompileError::UnknownInput {
                    node: node_id.clone(),
                    input: name.clone(),
                },
            )?
        } else {
            let input = resolve_input_descriptor(descriptor, name).ok_or_else(|| {
                PromptCompileError::UnknownInput {
                    node: node_id.clone(),
                    input: name.clone(),
                }
            })?;
            let source_schema = resolve_input_schema(descriptor, name);
            (input, source_schema)
        };
        if input.hidden {
            continue;
        }
        let binding = if let Some((source, output_index)) = decode_link(value) {
            if !prompt.0.contains_key(&source) {
                return Err(PromptCompileError::UnknownLink {
                    node: node_id.clone(),
                    input: name.clone(),
                    source_node: source,
                });
            }
            let source_descriptor = descriptors.get(&source).copied().ok_or_else(|| {
                PromptCompileError::UnknownLink {
                    node: node_id.clone(),
                    input: name.clone(),
                    source_node: source.clone(),
                }
            })?;
            let output = source_descriptor.outputs.get(output_index).ok_or_else(|| {
                PromptCompileError::UnknownOutput {
                    node: node_id.clone(),
                    input: name.clone(),
                    source_node: source.clone(),
                    output_index,
                }
            })?;
            let list_compatible =
                !output.is_list || input.cardinality != NativePortCardinality::Scalar;
            if !type_union_accepts_output(&input.accepted_types, &output.produced_type)
                || !list_compatible
            {
                return Err(PromptCompileError::IncompatibleLink {
                    node: node_id.clone(),
                    input: name.clone(),
                    source_node: source,
                    output_index,
                });
            }
            InputBinding::Link {
                source,
                output_index,
                lazy: input.lazy,
                mode: input.cardinality,
            }
        } else {
            let native_value = native_literal(value, &input.accepted_types, input.cardinality);
            if !input.allows_literal || native_value.is_none() {
                return Err(PromptCompileError::InvalidLiteral {
                    node: node_id.clone(),
                    input: name.clone(),
                });
            }
            let value = native_value.ok_or_else(|| PromptCompileError::InvalidLiteral {
                node: node_id.clone(),
                input: name.clone(),
            })?;
            let schema_matches = source_schema.as_ref().is_none_or(|schema| {
                structured_selector_matches(schema, &value)
                    || descriptor
                        .source_schema
                        .as_ref()
                        .is_some_and(|source_schema| {
                            sim_nodes::native_value_matches_source_input_schema(
                                &value,
                                &descriptor.class_type,
                                source_schema,
                                schema,
                            )
                        })
            });
            if !schema_matches {
                return Err(PromptCompileError::InvalidLiteral {
                    node: node_id.clone(),
                    input: name.clone(),
                });
            }
            InputBinding::Literal { value }
        };
        compiled.insert(name.clone(), binding);
    }
    Ok(compiled)
}

fn validate_required_structured_inputs(
    node_id: &NodeId,
    prompt_node: &PromptNode,
    descriptor: &NativeNodeDescriptor,
) -> Result<(), PromptCompileError> {
    let Some(source_schema) = &descriptor.source_schema else {
        return Ok(());
    };
    for schema in &source_schema.inputs {
        if schema
            .structured_options()
            .map_err(|_| {
                PromptCompileError::InvalidRuntimeDescriptor(descriptor.class_type.clone())
            })?
            .is_empty()
        {
            continue;
        }
        validate_required_structured_fields(
            node_id,
            prompt_node,
            descriptor,
            &schema.name,
            schema,
        )?;
    }
    Ok(())
}

fn validate_compiled_structured_inputs(node: &CompiledNode) -> Result<(), PromptCompileError> {
    let Some(source_schema) = &node.descriptor.source_schema else {
        return Ok(());
    };
    for schema in &source_schema.inputs {
        let options = schema
            .structured_options()
            .map_err(|_| PromptCompileError::InvalidCompiledPlan)?;
        if options.is_empty() {
            continue;
        }
        let mut active_fields = BTreeSet::new();
        validate_compiled_structured_fields(node, &schema.name, schema, &mut active_fields)?;
        let prefix = format!("{}.", schema.name);
        if node
            .inputs
            .keys()
            .any(|name| name.starts_with(&prefix) && !active_fields.contains(name))
        {
            return Err(PromptCompileError::InvalidCompiledPlan);
        }
    }
    Ok(())
}

fn validate_compiled_structured_fields(
    node: &CompiledNode,
    prefix: &str,
    schema: &sim_nodes::NativeInputSchemaMetadata,
    active_fields: &mut BTreeSet<String>,
) -> Result<(), PromptCompileError> {
    let selector = match node.inputs.get(prefix) {
        Some(InputBinding::Literal {
            value:
                NativeValue::PreservedUnknown {
                    type_name,
                    value: Value::String(selector),
                },
        }) if type_name == "COMFY_DYNAMICCOMBO_V3" => selector,
        _ => return Err(PromptCompileError::InvalidCompiledPlan),
    };
    let options = schema
        .structured_options()
        .map_err(|_| PromptCompileError::InvalidCompiledPlan)?;
    let option = options
        .iter()
        .find(|option| option.selector == *selector)
        .ok_or(PromptCompileError::InvalidCompiledPlan)?;
    for field in &option.fields {
        if field.path.is_empty() {
            return Err(PromptCompileError::InvalidCompiledPlan);
        }
        let name = format!("{prefix}.{}", field.path.join("."));
        active_fields.insert(name.clone());
        if field.required && !node.inputs.contains_key(&name) {
            return Err(PromptCompileError::InvalidCompiledPlan);
        }
        if node.inputs.contains_key(&name)
            && !field
                .schema
                .structured_options()
                .map_err(|_| PromptCompileError::InvalidCompiledPlan)?
                .is_empty()
        {
            validate_compiled_structured_fields(node, &name, &field.schema, active_fields)?;
        }
    }
    Ok(())
}

fn resolve_compiled_structured_input(
    node: &CompiledNode,
    name: &str,
) -> Option<(
    sim_nodes::NativeInputDescriptor,
    Option<sim_nodes::NativeInputSchemaMetadata>,
)> {
    let mut parts = name.split('.');
    let root = parts.next()?;
    let mut schema = resolve_input_schema(&node.descriptor, root)?;
    let mut prefix = root.to_owned();
    let parts = parts.collect::<Vec<_>>();
    let mut remaining = parts.as_slice();
    let mut selected_field = None;
    while !remaining.is_empty() {
        let selector = match node.inputs.get(&prefix)? {
            InputBinding::Literal {
                value:
                    NativeValue::PreservedUnknown {
                        type_name,
                        value: Value::String(selector),
                    },
            } if type_name == "COMFY_DYNAMICCOMBO_V3" => selector,
            _ => return None,
        };
        let option = schema
            .structured_options()
            .ok()?
            .into_iter()
            .find(|option| option.selector == *selector)?;
        let field = option
            .fields
            .iter()
            .find(|field| {
                field
                    .path
                    .iter()
                    .map(String::as_str)
                    .eq(remaining.iter().copied())
            })
            .or_else(|| {
                option
                    .fields
                    .iter()
                    .find(|field| field.path.as_slice() == [remaining[0]])
            })?
            .clone();
        schema = field.schema.clone();
        remaining = &remaining[field.path.len()..];
        prefix.push('.');
        prefix.push_str(&field.path.join("."));
        selected_field = Some(field);
    }
    let field = selected_field?;
    let accepted_types = sim_nodes::native_value_types_for_input_schema(&schema).ok()?;
    let allows_literal = accepted_types
        .members()
        .iter()
        .any(|member| !matches!(member, NativeValueType::Handle(_)));
    Some((
        sim_nodes::NativeInputDescriptor {
            name: name.to_owned(),
            accepted_types,
            required: field.required,
            hidden: false,
            lazy: field.lazy,
            cardinality: resolve_input_descriptor(&node.descriptor, root)?.cardinality,
            allows_literal,
        },
        Some(schema),
    ))
}

fn validate_required_structured_fields(
    node_id: &NodeId,
    prompt_node: &PromptNode,
    descriptor: &NativeNodeDescriptor,
    prefix: &str,
    schema: &sim_nodes::NativeInputSchemaMetadata,
) -> Result<(), PromptCompileError> {
    let selector = prompt_node
        .inputs
        .get(prefix)
        .and_then(Value::as_str)
        .ok_or_else(|| PromptCompileError::InvalidLiteral {
            node: node_id.clone(),
            input: prefix.to_owned(),
        })?;
    let options = schema
        .structured_options()
        .map_err(|_| PromptCompileError::InvalidRuntimeDescriptor(descriptor.class_type.clone()))?;
    let option = options
        .iter()
        .find(|option| option.selector == selector)
        .ok_or_else(|| PromptCompileError::InvalidLiteral {
            node: node_id.clone(),
            input: prefix.to_owned(),
        })?;
    for field in &option.fields {
        if field.path.is_empty() {
            return Err(PromptCompileError::InvalidRuntimeDescriptor(
                descriptor.class_type.clone(),
            ));
        }
        let name = format!("{prefix}.{}", field.path.join("."));
        if field.required && !prompt_node.inputs.contains_key(&name) {
            return Err(PromptCompileError::MissingInput {
                node: node_id.clone(),
                input: name,
            });
        }
        if prompt_node.inputs.contains_key(&name)
            && !field
                .schema
                .structured_options()
                .map_err(|_| {
                    PromptCompileError::InvalidRuntimeDescriptor(descriptor.class_type.clone())
                })?
                .is_empty()
        {
            validate_required_structured_fields(
                node_id,
                prompt_node,
                descriptor,
                &name,
                &field.schema,
            )?;
        }
    }
    Ok(())
}

fn resolve_active_structured_input(
    node_id: &NodeId,
    descriptor: &NativeNodeDescriptor,
    prompt_node: &PromptNode,
    name: &str,
) -> Result<
    Option<(
        sim_nodes::NativeInputDescriptor,
        Option<sim_nodes::NativeInputSchemaMetadata>,
    )>,
    PromptCompileError,
> {
    let mut parts = name.split('.');
    let Some(root) = parts.next() else {
        return Ok(None);
    };
    let Some(mut schema) = resolve_input_schema(descriptor, root) else {
        return Ok(None);
    };
    let mut prefix = root.to_owned();
    let parts = parts.collect::<Vec<_>>();
    let mut remaining = parts.as_slice();
    let mut field = None;
    while !remaining.is_empty() {
        let selector = prompt_node
            .inputs
            .get(&prefix)
            .and_then(Value::as_str)
            .ok_or_else(|| PromptCompileError::InvalidLiteral {
                node: node_id.clone(),
                input: prefix.clone(),
            })?;
        let options = schema.structured_options().map_err(|_| {
            PromptCompileError::InvalidRuntimeDescriptor(descriptor.class_type.clone())
        })?;
        let option = options.iter().find(|option| option.selector == selector);
        let Some(next) = option.and_then(|option| {
            option
                .fields
                .iter()
                .find(|field| {
                    field
                        .path
                        .iter()
                        .map(String::as_str)
                        .eq(remaining.iter().copied())
                })
                .or_else(|| {
                    option
                        .fields
                        .iter()
                        .find(|field| field.path.as_slice() == [remaining[0]])
                })
        }) else {
            return Ok(None);
        };
        schema = next.schema.clone();
        remaining = &remaining[next.path.len()..];
        prefix.push('.');
        prefix.push_str(&next.path.join("."));
        field = Some(next.clone());
    }
    let Some(field) = field else {
        return Ok(None);
    };
    let accepted_types = sim_nodes::native_value_types_for_input_schema(&schema)
        .map_err(|_| PromptCompileError::InvalidRuntimeDescriptor(descriptor.class_type.clone()))?;
    let allows_literal = accepted_types
        .members()
        .iter()
        .any(|member| !matches!(member, NativeValueType::Handle(_)));
    Ok(Some((
        sim_nodes::NativeInputDescriptor {
            name: name.to_owned(),
            accepted_types,
            required: field.required,
            hidden: false,
            lazy: field.lazy,
            cardinality: NativePortCardinality::Scalar,
            allows_literal,
        },
        Some(schema),
    )))
}

fn structured_selector_matches(
    schema: &sim_nodes::NativeInputSchemaMetadata,
    value: &NativeValue,
) -> bool {
    let NativeValue::PreservedUnknown {
        type_name,
        value: Value::String(selector),
    } = value
    else {
        return false;
    };
    type_name == "COMFY_DYNAMICCOMBO_V3"
        && schema
            .structured_options()
            .is_ok_and(|options| options.iter().any(|option| option.selector == *selector))
}

fn inject_hidden_inputs(
    node_id: &NodeId,
    descriptor: &NativeNodeDescriptor,
    prompt: &ApiPrompt,
    extra_data: &BTreeMap<String, Value>,
    inputs: &mut BTreeMap<String, InputBinding>,
) -> Result<(), PromptCompileError> {
    for input in &descriptor.inputs {
        if !input.hidden {
            continue;
        }
        let value = match input.name.as_str() {
            "prompt" => serde_json::to_value(prompt),
            "extra_pnginfo" => Ok(extra_data
                .get("extra_pnginfo")
                .cloned()
                .unwrap_or(Value::Null)),
            "unique_id" => Ok(Value::String(node_id.0.clone())),
            _ if !input.required => continue,
            _ => {
                return Err(PromptCompileError::UnsupportedHiddenInput {
                    node: node_id.clone(),
                    input: input.name.clone(),
                });
            }
        }
        .map_err(|_| PromptCompileError::InvalidLiteral {
            node: node_id.clone(),
            input: input.name.clone(),
        })?;
        let value = native_literal(&value, &input.accepted_types, NativePortCardinality::Scalar)
            .ok_or_else(|| PromptCompileError::InvalidLiteral {
                node: node_id.clone(),
                input: input.name.clone(),
            })?;
        if resolve_input_schema(descriptor, &input.name)
            .as_ref()
            .is_some_and(|schema| {
                !descriptor
                    .source_schema
                    .as_ref()
                    .is_some_and(|source_schema| {
                        sim_nodes::native_value_matches_source_input_schema(
                            &value,
                            &descriptor.class_type,
                            source_schema,
                            schema,
                        )
                    })
            })
        {
            return Err(PromptCompileError::InvalidLiteral {
                node: node_id.clone(),
                input: input.name.clone(),
            });
        }
        inputs.insert(input.name.clone(), InputBinding::Literal { value });
    }
    Ok(())
}

pub(crate) fn resolve_input_descriptor(
    descriptor: &NativeNodeDescriptor,
    name: &str,
) -> Option<sim_nodes::NativeInputDescriptor> {
    if name.contains('.') {
        let field = resolve_structured_input_field_any(descriptor, name)?;
        let accepted_types = sim_nodes::native_value_types_for_input_schema(&field.schema).ok()?;
        let allows_literal = accepted_types
            .members()
            .iter()
            .any(|member| !matches!(member, NativeValueType::Handle(_)));
        return Some(sim_nodes::NativeInputDescriptor {
            name: name.to_owned(),
            accepted_types,
            required: field.required,
            hidden: false,
            lazy: field.lazy,
            cardinality: resolve_input_descriptor(descriptor, name.split('.').next()?)?.cardinality,
            allows_literal,
        });
    }
    if let Some(input) = descriptor.inputs.iter().find(|input| input.name == name) {
        return Some(input.clone());
    }
    for (dynamic_index, dynamic) in descriptor.dynamic_inputs.iter().enumerate() {
        if dynamic.name_template == "{name}"
            && descriptor
                .source_schema
                .as_ref()
                .and_then(|schema| schema.dynamic_inputs.get(dynamic_index))
                .is_some_and(|schema| schema.names.iter().any(|candidate| candidate == name))
        {
            let mut input = dynamic.input.clone();
            input.name = name.to_owned();
            return Some(input);
        }
        let Some((prefix, suffix)) = dynamic.name_template.split_once("{index}") else {
            continue;
        };
        let Some(index) = name
            .strip_prefix(prefix)
            .and_then(|value| value.strip_suffix(suffix))
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        let Some(end) = dynamic.start_index.checked_add(dynamic.maximum_count) else {
            continue;
        };
        if index >= dynamic.start_index && index < end {
            let mut input = dynamic.input.clone();
            input.name = name.to_owned();
            return Some(input);
        }
    }
    None
}

fn resolve_input_schema(
    descriptor: &NativeNodeDescriptor,
    name: &str,
) -> Option<sim_nodes::NativeInputSchemaMetadata> {
    if name.contains('.') {
        return resolve_structured_input_field_any(descriptor, name).map(|field| field.schema);
    }
    let source_schema = descriptor.source_schema.as_ref()?;
    if let Some((index, _)) = descriptor
        .inputs
        .iter()
        .enumerate()
        .find(|(_, input)| input.name == name)
    {
        return source_schema.inputs.get(index).cloned();
    }
    for (index, dynamic) in descriptor.dynamic_inputs.iter().enumerate() {
        if dynamic.name_template == "{name}" {
            let dynamic_schema = source_schema.dynamic_inputs.get(index)?;
            if dynamic_schema
                .names
                .iter()
                .any(|candidate| candidate == name)
            {
                let mut schema = (*dynamic_schema.input).clone();
                schema.name = name.to_owned();
                return Some(schema);
            }
            continue;
        }
        let Some((prefix, suffix)) = dynamic.name_template.split_once("{index}") else {
            continue;
        };
        let Some(value) = name
            .strip_prefix(prefix)
            .and_then(|value| value.strip_suffix(suffix))
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        let end = dynamic.start_index.checked_add(dynamic.maximum_count)?;
        if value >= dynamic.start_index && value < end {
            let mut schema = (*source_schema.dynamic_inputs.get(index)?.input).clone();
            schema.name = name.to_owned();
            return Some(schema);
        }
    }
    None
}

fn resolve_structured_input_field_any(
    descriptor: &NativeNodeDescriptor,
    name: &str,
) -> Option<sim_nodes::NativeStructuredInputField> {
    let mut parts = name.split('.');
    let root = parts.next()?;
    let mut schemas = vec![resolve_input_schema(descriptor, root)?];
    let parts = parts.collect::<Vec<_>>();
    let mut remaining = parts.as_slice();
    let mut selected_field = None;
    while !remaining.is_empty() {
        let mut fields = Vec::new();
        for schema in schemas {
            for option in schema.structured_options().ok()? {
                fields.extend(option.fields);
            }
        }
        let mut matches = fields
            .iter()
            .filter(|field| {
                field
                    .path
                    .iter()
                    .map(String::as_str)
                    .eq(remaining.iter().copied())
            })
            .collect::<Vec<_>>();
        if matches.is_empty() {
            matches = fields
                .iter()
                .filter(|field| field.path.as_slice() == [remaining[0]])
                .collect();
        }
        let first = (*matches.first()?).clone();
        if matches.iter().any(|field| {
            field.schema != first.schema
                || field.required != first.required
                || field.lazy != first.lazy
        }) {
            return None;
        }
        remaining = &remaining[first.path.len()..];
        schemas = vec![first.schema.clone()];
        selected_field = Some(first);
    }
    selected_field
}

fn required_dynamic_input_names(
    descriptor: &NativeNodeDescriptor,
    dynamic_index: usize,
    dynamic: &sim_nodes::NativeDynamicInputDescriptor,
) -> Result<Vec<String>, PromptCompileError> {
    if dynamic.name_template == "{name}" {
        let names = descriptor
            .source_schema
            .as_ref()
            .and_then(|schema| schema.dynamic_inputs.get(dynamic_index))
            .map(|schema| schema.names.clone())
            .ok_or_else(|| {
                PromptCompileError::InvalidRuntimeDescriptor(descriptor.class_type.clone())
            })?;
        return Ok(names
            .into_iter()
            .take(dynamic.minimum_count as usize)
            .collect());
    }
    (0..dynamic.minimum_count)
        .map(|offset| {
            dynamic
                .start_index
                .checked_add(offset)
                .map(|index| dynamic.name_template.replace("{index}", &index.to_string()))
                .ok_or_else(|| {
                    PromptCompileError::InvalidRuntimeDescriptor(descriptor.class_type.clone())
                })
        })
        .collect()
}

fn type_union_accepts_output(union: &NativeTypeUnion, output: &NativeValueType) -> bool {
    union.members().iter().any(|expected| {
        expected == &NativeValueType::Any
            || expected == output
            || matches!(
                (expected, output),
                (
                    NativeValueType::Primitive(NativePrimitiveType::Number),
                    NativeValueType::Primitive(NativePrimitiveType::Integer)
                )
            )
    })
}

fn native_literal(
    value: &Value,
    accepted_types: &NativeTypeUnion,
    cardinality: NativePortCardinality,
) -> Option<NativeValue> {
    match cardinality {
        NativePortCardinality::List => value.as_array().and_then(|values| {
            values
                .iter()
                .map(|value| native_scalar_literal(value, accepted_types))
                .collect::<Option<Vec<_>>>()
                .map(|values| NativeValue::List { values })
        }),
        NativePortCardinality::Mapped => value.as_array().map_or_else(
            || native_scalar_literal(value, accepted_types),
            |values| {
                values
                    .iter()
                    .map(|value| native_scalar_literal(value, accepted_types))
                    .collect::<Option<Vec<_>>>()
                    .map(|values| NativeValue::List { values })
            },
        ),
        NativePortCardinality::Scalar => native_scalar_literal(value, accepted_types),
    }
}

fn native_scalar_literal(value: &Value, accepted_types: &NativeTypeUnion) -> Option<NativeValue> {
    let primitive = match value {
        Value::Null => Some(NativePrimitive::Null),
        Value::Bool(value) => Some(NativePrimitive::Boolean(*value)),
        Value::Number(value) => value
            .as_i64()
            .map(NativePrimitive::Integer)
            .or_else(|| value.as_u64().map(NativePrimitive::UnsignedInteger))
            .or_else(|| {
                value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .map(NativePrimitive::Number)
            }),
        Value::String(value) => Some(NativePrimitive::String(value.clone())),
        Value::Array(_) | Value::Object(_) => None,
    };
    if let Some(primitive) = primitive {
        let native = NativeValue::Primitive { value: primitive };
        if accepted_types.accepts(&native)
            || accepted_types.members() == [NativeValueType::Any]
            || matches!(
                &native,
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(_) | NativePrimitive::UnsignedInteger(_)
                }
            ) && accepted_types
                .members()
                .contains(&NativeValueType::Primitive(NativePrimitiveType::Number))
        {
            return Some(native);
        }
    }
    if accepted_types.members() == [NativeValueType::Any]
        || accepted_types
            .members()
            .contains(&NativeValueType::PreservedUnknown)
        || accepted_types
            .members()
            .iter()
            .any(|value_type| matches!(value_type, NativeValueType::NamedPreservedUnknown(_)))
    {
        let type_name = accepted_types
            .members()
            .iter()
            .find_map(|value_type| match value_type {
                NativeValueType::NamedPreservedUnknown(type_name) => Some(type_name.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "zed.json@1".to_owned());
        Some(NativeValue::PreservedUnknown {
            type_name,
            value: value.clone(),
        })
    } else {
        None
    }
}

fn decode_link(value: &Value) -> Option<(NodeId, usize)> {
    let values = value.as_array()?;
    if values.len() != 2 {
        return None;
    }
    let source = values.first()?.as_str()?;
    let output_index = usize::try_from(values.get(1)?.as_u64()?).ok()?;
    Some((NodeId(source.to_owned()), output_index))
}

fn topological_order(
    nodes: &BTreeMap<NodeId, CompiledNode>,
) -> Result<Vec<NodeId>, PromptCompileError> {
    let mut incoming = nodes
        .keys()
        .map(|identifier| (identifier.clone(), 0_usize))
        .collect::<BTreeMap<_, _>>();
    let mut outgoing = BTreeMap::<NodeId, BTreeSet<NodeId>>::new();
    for node in nodes.values() {
        let dependencies = node
            .inputs
            .values()
            .filter_map(|input| match input {
                InputBinding::Link { source, .. } => Some(source),
                InputBinding::Literal { .. } => None,
            })
            .collect::<BTreeSet<_>>();
        if let Some(count) = incoming.get_mut(&node.id) {
            *count = dependencies.len();
        }
        for dependency in dependencies {
            outgoing
                .entry(dependency.clone())
                .or_default()
                .insert(node.id.clone());
        }
    }
    let mut ready = incoming
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(identifier, _)| identifier.clone())
        .collect::<BTreeSet<_>>();
    let mut ordered = Vec::with_capacity(nodes.len());
    while let Some(identifier) = ready.pop_first() {
        ordered.push(identifier.clone());
        if let Some(dependents) = outgoing.get(&identifier) {
            for dependent in dependents {
                if let Some(count) = incoming.get_mut(dependent) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        ready.insert(dependent.clone());
                    }
                }
            }
        }
    }
    if ordered.len() != nodes.len() {
        return Err(PromptCompileError::Cycle(
            incoming
                .into_iter()
                .filter(|(_, count)| *count > 0)
                .map(|(identifier, _)| identifier)
                .collect(),
        ));
    }
    Ok(ordered)
}

fn static_required_nodes(
    nodes: &BTreeMap<NodeId, CompiledNode>,
    output_nodes: &[NodeId],
) -> BTreeSet<NodeId> {
    let mut required = BTreeSet::new();
    let mut pending = output_nodes.to_vec();
    while let Some(identifier) = pending.pop() {
        if !required.insert(identifier.clone()) {
            continue;
        }
        if let Some(node) = nodes.get(&identifier) {
            pending.extend(node.inputs.values().filter_map(|input| match input {
                InputBinding::Link {
                    source,
                    lazy: false,
                    ..
                } => Some(source.clone()),
                InputBinding::Literal { .. } | InputBinding::Link { lazy: true, .. } => None,
            }));
        }
    }
    required
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;
    use sim_nodes::{
        LEGACY_NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCachePolicy,
        NativeDynamicInputDescriptor, NativeEffectClass, NativeInputDescriptor,
        NativeNodeContractError, NativeOutputDescriptor, NativePrimitiveType,
    };

    fn descriptor(
        class_type: &str,
        output_node: bool,
    ) -> Result<NativeNodeDescriptor, NativeNodeContractError> {
        Ok(NativeNodeDescriptor {
            schema_version: LEGACY_NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: None,
            inputs: Vec::new(),
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "value".to_owned(),
                produced_type: NativeValueType::Primitive(NativePrimitiveType::Number),
                is_list: false,
            }],
            output_node,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        })
    }

    fn input(
        name: &str,
        value_type: NativeValueType,
        lazy: bool,
        cardinality: NativePortCardinality,
        allows_literal: bool,
    ) -> Result<NativeInputDescriptor, NativeNodeContractError> {
        Ok(NativeInputDescriptor {
            name: name.to_owned(),
            accepted_types: NativeTypeUnion::new([value_type])?,
            required: true,
            hidden: false,
            lazy,
            cardinality,
            allows_literal,
        })
    }

    fn submission(nodes: BTreeMap<NodeId, PromptNode>) -> PromptSubmission {
        PromptSubmission {
            prompt: ApiPrompt(nodes),
            prompt_id: Some(PromptId(Uuid::nil())),
            client_id: Some("client".to_owned()),
            number: Some(7.0),
            extra_data: BTreeMap::from([("workflow".to_owned(), json!({"version": 1}))]),
            unknown: BTreeMap::from([("future".to_owned(), json!(true))]),
        }
    }

    #[test]
    fn val_domain_004_prompt_graph_and_lazy_dependencies_compile_deterministically()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut source = descriptor("Source", false)?;
        source.outputs[0].is_list = true;
        let mut choose = descriptor("Choose", true)?;
        choose.inputs.push(input(
            "condition",
            NativeValueType::Primitive(NativePrimitiveType::Boolean),
            false,
            NativePortCardinality::Scalar,
            true,
        )?);
        choose.inputs.push(input(
            "value",
            NativeValueType::Primitive(NativePrimitiveType::Number),
            true,
            NativePortCardinality::Mapped,
            false,
        )?);
        let mut registry = NativeNodeRegistry::default();
        registry.register_descriptor(source)?;
        registry.register_descriptor(choose)?;
        let plan = PromptCompiler::new(&registry).compile(submission(BTreeMap::from([
            (
                NodeId::from("001"),
                PromptNode {
                    class_type: "Source".to_owned(),
                    inputs: BTreeMap::new(),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                NodeId::from("output"),
                PromptNode {
                    class_type: "Choose".to_owned(),
                    inputs: BTreeMap::from([
                        ("condition".to_owned(), json!(true)),
                        ("value".to_owned(), json!(["001", 0])),
                    ]),
                    unknown: BTreeMap::from([("future_node_field".to_owned(), json!(9))]),
                },
            ),
        ])))?;
        assert_eq!(
            plan.topological_order,
            [NodeId::from("001"), NodeId::from("output")]
        );
        assert_eq!(
            plan.static_required_nodes,
            BTreeSet::from([NodeId::from("output")])
        );
        assert_eq!(plan.unknown["future"], true);
        assert_eq!(
            plan.nodes[&NodeId::from("output")].unknown["future_node_field"],
            9
        );
        Ok(())
    }

    #[test]
    fn openrouter_named_autogrow_image_links_compile_with_exact_active_option()
    -> Result<(), Box<dyn std::error::Error>> {
        let registry = crate::generated_native_node_registry_projection(None)?;
        let provider = registry
            .descriptor("OpenRouterLLMNode")
            .ok_or("OpenRouter descriptor is absent")?
            .clone();
        let mut source = descriptor("ImageSource", false)?;
        source.outputs[0].produced_type =
            sim_nodes::native_source_type_projection("IMAGE")?.value_type()?;
        let provider_id = NodeId::from("provider");
        let image_id = NodeId::from("image");
        let provider_node = PromptNode {
            class_type: "OpenRouterLLMNode".to_owned(),
            inputs: BTreeMap::from([
                ("prompt".to_owned(), json!("hello")),
                ("model".to_owned(), json!("anthropic/claude-opus-4.7")),
                ("model.reasoning_effort".to_owned(), json!("off")),
                ("model.images.image_1".to_owned(), json!(["image", 0])),
                ("seed".to_owned(), json!(0)),
            ]),
            unknown: BTreeMap::new(),
        };
        let prompt = ApiPrompt(BTreeMap::from([
            (provider_id.clone(), provider_node.clone()),
            (
                image_id.clone(),
                PromptNode {
                    class_type: "ImageSource".to_owned(),
                    inputs: BTreeMap::new(),
                    unknown: BTreeMap::new(),
                },
            ),
        ]));
        let descriptors = BTreeMap::from([
            (provider_id.clone(), &provider),
            (image_id.clone(), &source),
        ]);
        let inputs = compile_inputs(
            &provider_id,
            &provider_node,
            &provider,
            &prompt,
            &descriptors,
        )?;
        assert!(matches!(
            inputs.get("model.images.image_1"),
            Some(InputBinding::Link { source, output_index: 0, .. }) if source == &image_id
        ));
        let compiled = CompiledNode {
            id: provider_id.clone(),
            class_type: provider.class_type.clone(),
            descriptor: provider.clone(),
            inputs,
            unknown: BTreeMap::new(),
        };
        validate_compiled_structured_inputs(&compiled)?;
        assert!(resolve_compiled_structured_input(&compiled, "model.images.image_1").is_some());

        let mut absent_image = provider_node.clone();
        absent_image.inputs.remove("model.images.image_1");
        compile_inputs(
            &provider_id,
            &absent_image,
            &provider,
            &prompt,
            &descriptors,
        )?;
        let mut inactive_video = provider_node.clone();
        inactive_video
            .inputs
            .insert("model.videos.video_1".to_owned(), json!(["image", 0]));
        assert!(matches!(
            compile_inputs(
                &provider_id,
                &inactive_video,
                &provider,
                &prompt,
                &descriptors
            ),
            Err(PromptCompileError::UnknownInput { .. })
        ));
        let mut beyond_limit = provider_node;
        beyond_limit.inputs.remove("model.images.image_1");
        beyond_limit
            .inputs
            .insert("model.images.image_21".to_owned(), json!(["image", 0]));
        assert!(matches!(
            compile_inputs(
                &provider_id,
                &beyond_limit,
                &provider,
                &prompt,
                &descriptors
            ),
            Err(PromptCompileError::UnknownInput { .. })
        ));
        Ok(())
    }

    #[test]
    fn source_declared_dotted_multitype_links_compile_as_real_dependencies()
    -> Result<(), Box<dyn std::error::Error>> {
        let image_type = sim_nodes::native_source_type_projection("IMAGE")?.value_type()?;
        let mut source = descriptor("ImageSource", false)?;
        source.schema_version = sim_nodes::NATIVE_NODE_CONTRACT_SCHEMA_VERSION;
        source.outputs[0].produced_type = image_type;
        source.source_schema = Some(sim_nodes::NativeDescriptorSchemaMetadata::compatibility(
            sim_nodes::NativeSchemaProvenance::SourceV3,
            std::iter::empty(),
            std::iter::empty(),
            [("value".to_owned(), "IMAGE".to_owned())],
        ));

        let expression = json!({
            "arguments": [
                {"kind": "attribute", "name": "ResizeType.MATCH_SIZE"},
                {"kind": "list", "items": [
                    {
                        "arguments": [
                            {"kind": "literal", "value": "match"},
                            {"kind": "list", "items": [
                                {"kind": "attribute", "name": "io.Image"},
                                {"kind": "attribute", "name": "io.Mask"}
                            ]}
                        ],
                        "keywords": [
                            {"name": "lazy", "value": {"kind": "literal", "value": true}}
                        ],
                        "kind": "call",
                        "name": "io.MultiType.Input"
                    },
                    {
                        "arguments": [{"kind": "literal", "value": "crop"}],
                        "keywords": [
                            {"name": "options", "value": {"kind": "list", "items": [
                                {"kind": "literal", "value": "disabled"},
                                {"kind": "literal", "value": "center"}
                            ]}}
                        ],
                        "kind": "call",
                        "name": "io.Combo.Input"
                    }
                ]}
            ],
            "keywords": [],
            "kind": "call",
            "name": "io.DynamicCombo.Option"
        });
        let expression = serde_json::to_string(&expression)?;
        let mut sink = descriptor("StructuredSink", true)?;
        sink.schema_version = sim_nodes::NATIVE_NODE_CONTRACT_SCHEMA_VERSION;
        sink.inputs.push(input(
            "resize_type",
            NativeValueType::NamedPreservedUnknown("COMFY_DYNAMICCOMBO_V3".to_owned()),
            false,
            NativePortCardinality::Scalar,
            true,
        )?);
        sink.outputs[0].produced_type = NativeValueType::Primitive(NativePrimitiveType::Boolean);
        let mut sink_schema = sim_nodes::NativeDescriptorSchemaMetadata::compatibility(
            sim_nodes::NativeSchemaProvenance::SourceV3,
            [("resize_type".to_owned(), "COMFY_DYNAMICCOMBO_V3".to_owned())],
            std::iter::empty(),
            [("value".to_owned(), "BOOLEAN".to_owned())],
        );
        sink_schema.inputs[0].choices = vec![sim_nodes::NativeSchemaValue::PreservedExpression {
            sha256: format!("{:x}", Sha256::digest(expression.as_bytes())),
            source: expression,
        }];
        sink.source_schema = Some(sink_schema);

        let mut registry = NativeNodeRegistry::default();
        registry.register_descriptor(source)?;
        registry.register_descriptor(sink)?;
        let nodes = BTreeMap::from([
            (
                NodeId::from("image"),
                PromptNode {
                    class_type: "ImageSource".to_owned(),
                    inputs: BTreeMap::new(),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                NodeId::from("output"),
                PromptNode {
                    class_type: "StructuredSink".to_owned(),
                    inputs: BTreeMap::from([
                        ("resize_type".to_owned(), json!("match size")),
                        ("resize_type.crop".to_owned(), json!("center")),
                        ("resize_type.match".to_owned(), json!(["image", 0])),
                    ]),
                    unknown: BTreeMap::new(),
                },
            ),
        ]);
        let plan = PromptCompiler::new(&registry).compile(submission(nodes.clone()))?;
        assert_eq!(
            plan.nodes[&NodeId::from("output")].inputs["resize_type.match"],
            InputBinding::Link {
                source: NodeId::from("image"),
                output_index: 0,
                lazy: true,
                mode: NativePortCardinality::Scalar,
            }
        );
        assert_eq!(
            plan.topological_order,
            [NodeId::from("image"), NodeId::from("output")]
        );
        assert!(!plan.static_required_nodes.contains(&NodeId::from("image")));
        plan.validate_integrity()?;
        let restored: CompiledPlan = serde_json::from_slice(&serde_json::to_vec(&plan)?)?;
        assert_eq!(restored, plan);
        restored.validate_integrity()?;

        let mut tampered_plan = plan;
        tampered_plan
            .nodes
            .get_mut(&NodeId::from("output"))
            .ok_or("missing compiled sink")?
            .inputs
            .remove("resize_type.crop");
        assert_eq!(
            tampered_plan.validate_integrity(),
            Err(PromptCompileError::InvalidCompiledPlan)
        );

        let mut missing = nodes.clone();
        missing
            .get_mut(&NodeId::from("output"))
            .ok_or("missing sink")?
            .inputs
            .remove("resize_type.crop");
        assert!(matches!(
            PromptCompiler::new(&registry).compile(submission(missing)),
            Err(PromptCompileError::MissingInput { input, .. })
                if input == "resize_type.crop"
        ));

        let mut inactive = nodes.clone();
        inactive
            .get_mut(&NodeId::from("output"))
            .ok_or("missing sink")?
            .inputs
            .insert("resize_type.width".to_owned(), json!(512));
        assert!(matches!(
            PromptCompiler::new(&registry).compile(submission(inactive)),
            Err(PromptCompileError::UnknownInput { input, .. })
                if input == "resize_type.width"
        ));

        let mut handle_shaped_json = nodes;
        handle_shaped_json
            .get_mut(&NodeId::from("output"))
            .ok_or("missing sink")?
            .inputs = BTreeMap::from([(
            "resize_type".to_owned(),
            json!({
                "resize_type": "match size",
                "match": ["image", 0],
                "crop": "center"
            }),
        )]);
        assert!(matches!(
            PromptCompiler::new(&registry).compile(submission(handle_shaped_json)),
            Err(PromptCompileError::InvalidLiteral { input, .. })
                if input == "resize_type"
        ));
        Ok(())
    }

    #[test]
    fn val_domain_004_invalid_graphs_fail_with_node_addressable_errors()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut output = descriptor("Output", true)?;
        output.inputs.push(input(
            "value",
            NativeValueType::Primitive(NativePrimitiveType::Number),
            false,
            NativePortCardinality::Scalar,
            false,
        )?);
        let mut registry = NativeNodeRegistry::default();
        registry.register_descriptor(output)?;
        let error = PromptCompiler::new(&registry)
            .compile(submission(BTreeMap::from([(
                NodeId::from("out"),
                PromptNode {
                    class_type: "Output".to_owned(),
                    inputs: BTreeMap::from([("value".to_owned(), json!(["missing", 0]))]),
                    unknown: BTreeMap::new(),
                },
            )])))
            .expect_err("unknown link must fail");
        assert!(matches!(error, PromptCompileError::UnknownLink { .. }));
        Ok(())
    }

    #[test]
    fn val_domain_004_hidden_inputs_are_host_injected() -> Result<(), Box<dyn std::error::Error>> {
        let mut output = descriptor("Output", true)?;
        for (name, value_type) in [
            ("prompt", NativeValueType::Any),
            ("extra_pnginfo", NativeValueType::Any),
            (
                "unique_id",
                NativeValueType::Primitive(NativePrimitiveType::String),
            ),
        ] {
            let mut descriptor = input(
                name,
                value_type,
                false,
                NativePortCardinality::Scalar,
                false,
            )?;
            descriptor.hidden = true;
            output.inputs.push(descriptor);
        }
        let mut registry = NativeNodeRegistry::default();
        registry.register_descriptor(output)?;
        let mut host_submission = submission(BTreeMap::from([(
            NodeId::from("out"),
            PromptNode {
                class_type: "Output".to_owned(),
                inputs: BTreeMap::from([
                    ("prompt".to_owned(), json!("untrusted")),
                    ("extra_pnginfo".to_owned(), json!("untrusted")),
                    ("unique_id".to_owned(), json!("untrusted")),
                ]),
                unknown: BTreeMap::new(),
            },
        )]));
        host_submission.extra_data.insert(
            "extra_pnginfo".to_owned(),
            json!({"workflow": {"id": "fixture"}}),
        );
        let plan = PromptCompiler::new(&registry).compile(host_submission)?;
        let inputs = &plan.nodes[&NodeId::from("out")].inputs;
        assert_eq!(
            inputs["unique_id"],
            InputBinding::Literal {
                value: NativeValue::Primitive {
                    value: NativePrimitive::String("out".to_owned())
                }
            }
        );
        assert!(matches!(
            &inputs["extra_pnginfo"],
            InputBinding::Literal {
                value: NativeValue::PreservedUnknown { value, .. }
            } if value == &json!({"workflow": {"id": "fixture"}})
        ));
        let InputBinding::Literal { value: prompt } = &inputs["prompt"] else {
            return Err("prompt hidden input was not injected".into());
        };
        assert!(matches!(
            prompt,
            NativeValue::PreservedUnknown { value, .. }
                if value.get("out").is_some()
        ));

        let mut provider = descriptor("Provider", true)?;
        let mut secret = input(
            "auth_token_sim_org",
            NativeValueType::Primitive(NativePrimitiveType::String),
            false,
            NativePortCardinality::Scalar,
            false,
        )?;
        secret.hidden = true;
        provider.inputs.push(secret);
        let mut provider_registry = NativeNodeRegistry::default();
        provider_registry.register_descriptor(provider)?;
        let provider_error = PromptCompiler::new(&provider_registry)
            .compile(submission(BTreeMap::from([(
                NodeId::from("provider"),
                PromptNode {
                    class_type: "Provider".to_owned(),
                    inputs: BTreeMap::from([(
                        "auth_token_sim_org".to_owned(),
                        json!("untrusted-secret"),
                    )]),
                    unknown: BTreeMap::new(),
                },
            )])))
            .expect_err("raw prompt data must not inject an authorization secret");
        assert!(matches!(
            provider_error,
            PromptCompileError::UnsupportedHiddenInput { .. }
        ));
        Ok(())
    }

    #[test]
    fn integer_literals_preserve_u64_and_signed_values_exactly()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut output = descriptor("Output", true)?;
        output.inputs.push(input(
            "seed",
            NativeValueType::Primitive(NativePrimitiveType::Integer),
            false,
            NativePortCardinality::Scalar,
            true,
        )?);
        let mut registry = NativeNodeRegistry::default();
        registry.register_descriptor(output)?;
        for (literal, expected) in [
            (json!(u64::MAX), NativePrimitive::UnsignedInteger(u64::MAX)),
            (json!(-7), NativePrimitive::Integer(-7)),
        ] {
            let plan = PromptCompiler::new(&registry).compile(submission(BTreeMap::from([(
                NodeId::from("out"),
                PromptNode {
                    class_type: "Output".to_owned(),
                    inputs: BTreeMap::from([("seed".to_owned(), literal)]),
                    unknown: BTreeMap::new(),
                },
            )])))?;
            assert_eq!(
                plan.nodes[&NodeId::from("out")].inputs["seed"],
                InputBinding::Literal {
                    value: NativeValue::Primitive { value: expected }
                }
            );
        }
        Ok(())
    }

    #[test]
    fn source_custom_asset_combo_compiles_and_rejects_forged_plan_metadata()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_schema = sim_nodes::built_in_source_schema("LoadLatent")?.bind_execution_ports(
            &["latent".to_owned()],
            &[],
            &["latent".to_owned()],
        )?;
        let source_input = source_schema.inputs.first().ok_or("source input missing")?;
        let source_output = source_schema
            .outputs
            .first()
            .ok_or("source output missing")?;
        let latent_type = sim_nodes::native_value_type_for_output_schema(source_output)?;
        let mut load = descriptor("LoadLatent", false)?;
        load.schema_version = sim_nodes::NATIVE_NODE_CONTRACT_SCHEMA_VERSION;
        load.inputs = vec![NativeInputDescriptor {
            name: "latent".to_owned(),
            accepted_types: sim_nodes::native_value_types_for_input_schema(source_input)?,
            required: true,
            hidden: false,
            lazy: false,
            cardinality: NativePortCardinality::Mapped,
            allows_literal: true,
        }];
        load.outputs = vec![NativeOutputDescriptor {
            name: "latent".to_owned(),
            produced_type: latent_type.clone(),
            is_list: false,
        }];
        load.effect = NativeEffectClass::ReadsArtifact;
        load.source_schema = Some(source_schema);
        let mut sink = descriptor("Sink", true)?;
        sink.inputs = vec![input(
            "value",
            latent_type,
            false,
            NativePortCardinality::Mapped,
            false,
        )?];
        let mut registry = NativeNodeRegistry::default();
        registry.register_descriptor(load)?;
        registry.register_descriptor(sink)?;
        let compile = |filename| {
            PromptCompiler::new(&registry).compile(submission(BTreeMap::from([
                (
                    NodeId::from("load"),
                    PromptNode {
                        class_type: "LoadLatent".to_owned(),
                        inputs: BTreeMap::from([("latent".to_owned(), filename)]),
                        unknown: BTreeMap::new(),
                    },
                ),
                (
                    NodeId::from("out"),
                    PromptNode {
                        class_type: "Sink".to_owned(),
                        inputs: BTreeMap::from([("value".to_owned(), json!(["load", 0]))]),
                        unknown: BTreeMap::new(),
                    },
                ),
            ])))
        };
        for value in [
            json!("existing.latent"),
            json!("existing.txt"),
            json!(["one.latent", "two.txt"]),
        ] {
            let plan = compile(value)?;
            plan.validate_integrity()?;
            let load = plan
                .nodes
                .get(&NodeId::from("load"))
                .ok_or("load node missing")?;
            let InputBinding::Literal { value } =
                load.inputs.get("latent").ok_or("latent binding missing")?
            else {
                return Err("filename was compiled as a link".into());
            };
            let check = |value: &NativeValue| matches!(value, NativeValue::PreservedUnknown { type_name, value } if type_name == "COMBO" && value.is_string());
            match value {
                NativeValue::List { values } => assert!(values.iter().all(check)),
                value => assert!(check(value)),
            }
        }
        for value in [
            json!(17),
            json!(false),
            json!(["one.latent", 17, "two.txt"]),
        ] {
            assert!(matches!(
                compile(value),
                Err(PromptCompileError::InvalidLiteral { .. })
            ));
        }
        let mut forged = compile(json!("existing.latent"))?;
        let metadata = forged
            .nodes
            .get_mut(&NodeId::from("load"))
            .and_then(|node| node.descriptor.source_schema.as_mut())
            .ok_or("source schema missing")?;
        metadata.node.definition_sha256 = Some("0".repeat(64));
        assert!(matches!(
            forged.validate_integrity(),
            Err(PromptCompileError::InvalidCompiledPlan)
        ));
        Ok(())
    }

    #[test]
    fn prompt_schema_constraints_fail_before_execution() -> Result<(), Box<dyn std::error::Error>> {
        let mut bounded = descriptor("Bounded", true)?;
        bounded.schema_version = sim_nodes::NATIVE_NODE_CONTRACT_SCHEMA_VERSION;
        bounded.inputs = vec![
            input(
                "seed",
                NativeValueType::Primitive(NativePrimitiveType::Integer),
                false,
                NativePortCardinality::Scalar,
                true,
            )?,
            input(
                "weights",
                NativeValueType::Primitive(NativePrimitiveType::Number),
                false,
                NativePortCardinality::List,
                true,
            )?,
        ];
        let mut source_schema = sim_nodes::NativeDescriptorSchemaMetadata::compatibility(
            sim_nodes::NativeSchemaProvenance::SourceV3,
            [
                ("seed".to_owned(), "INT".to_owned()),
                ("weights".to_owned(), "FLOAT".to_owned()),
            ],
            std::iter::empty(),
            [("value".to_owned(), "FLOAT".to_owned())],
        );
        source_schema.inputs[0].default =
            Some(sim_nodes::NativeSchemaValue::UnsignedInteger { value: 3 });
        source_schema.inputs[0].minimum =
            Some(sim_nodes::NativeSchemaValue::UnsignedInteger { value: 1 });
        source_schema.inputs[0].maximum =
            Some(sim_nodes::NativeSchemaValue::UnsignedInteger { value: 5 });
        source_schema.inputs[0].choices = [1, 3, 5]
            .into_iter()
            .map(|value| sim_nodes::NativeSchemaValue::UnsignedInteger { value })
            .collect();
        source_schema.inputs[1].minimum = Some(sim_nodes::NativeSchemaValue::FiniteDecimal {
            value: "0.0".to_owned(),
        });
        source_schema.inputs[1].maximum = Some(sim_nodes::NativeSchemaValue::FiniteDecimal {
            value: "1.0".to_owned(),
        });
        bounded.source_schema = Some(source_schema);
        let mut registry = NativeNodeRegistry::default();
        registry.register_descriptor(bounded)?;

        let compile = |seed, weights| {
            PromptCompiler::new(&registry).compile(submission(BTreeMap::from([(
                NodeId::from("out"),
                PromptNode {
                    class_type: "Bounded".to_owned(),
                    inputs: BTreeMap::from([
                        ("seed".to_owned(), seed),
                        ("weights".to_owned(), weights),
                    ]),
                    unknown: BTreeMap::new(),
                },
            )])))
        };
        compile(json!(3), json!([0.0, 0.5, 1.0]))?;
        for (seed, weights) in [
            (json!(0), json!([0.5])),
            (json!(4), json!([0.5])),
            (json!(6), json!([0.5])),
            (json!(3), json!([0.5, 1.1])),
        ] {
            assert!(matches!(
                compile(seed, weights),
                Err(PromptCompileError::InvalidLiteral { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn dynamic_input_resolution_checks_every_template() -> Result<(), Box<dyn std::error::Error>> {
        let mut dynamic = descriptor("Dynamic", true)?;
        let dynamic_input = input(
            "value",
            NativeValueType::Primitive(NativePrimitiveType::Number),
            false,
            NativePortCardinality::Scalar,
            true,
        )?;
        dynamic.dynamic_inputs = vec![
            NativeDynamicInputDescriptor {
                name_template: "first_{index}".to_owned(),
                start_index: 0,
                minimum_count: 0,
                maximum_count: 2,
                input: dynamic_input.clone(),
            },
            NativeDynamicInputDescriptor {
                name_template: "second_{index}".to_owned(),
                start_index: 4,
                minimum_count: 0,
                maximum_count: 2,
                input: dynamic_input,
            },
            NativeDynamicInputDescriptor {
                name_template: "{name}".to_owned(),
                start_index: 0,
                minimum_count: 1,
                maximum_count: 2,
                input: input(
                    "named_value",
                    NativeValueType::Primitive(NativePrimitiveType::Number),
                    false,
                    NativePortCardinality::Scalar,
                    true,
                )?,
            },
        ];
        let mut named_schema = sim_nodes::NativeDynamicSchemaMetadata::compatibility(
            "{name}",
            0,
            1,
            2,
            sim_nodes::NativeInputSchemaMetadata::compatibility("named_value", "FLOAT"),
        );
        named_schema.names = vec!["left".to_owned(), "right".to_owned()];
        dynamic.schema_version = sim_nodes::NATIVE_NODE_CONTRACT_SCHEMA_VERSION;
        dynamic.source_schema = Some(sim_nodes::NativeDescriptorSchemaMetadata::compatibility(
            sim_nodes::NativeSchemaProvenance::SourceV3,
            std::iter::empty(),
            [
                sim_nodes::NativeDynamicSchemaMetadata::compatibility(
                    "first_{index}",
                    0,
                    0,
                    2,
                    sim_nodes::NativeInputSchemaMetadata::compatibility("value", "FLOAT"),
                ),
                sim_nodes::NativeDynamicSchemaMetadata::compatibility(
                    "second_{index}",
                    4,
                    0,
                    2,
                    sim_nodes::NativeInputSchemaMetadata::compatibility("value", "FLOAT"),
                ),
                named_schema,
            ],
            [("value".to_owned(), "FLOAT".to_owned())],
        ));
        dynamic.validate()?;
        assert_eq!(
            resolve_input_descriptor(&dynamic, "second_5").map(|input| input.name),
            Some("second_5".to_owned())
        );
        assert_eq!(
            resolve_input_descriptor(&dynamic, "right").map(|input| input.name),
            Some("right".to_owned())
        );
        assert!(resolve_input_descriptor(&dynamic, "unknown").is_none());
        Ok(())
    }

    #[test]
    fn provider_plans_require_and_seal_the_selected_registry_pin()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut provider = descriptor("ProviderOutput", true)?;
        provider.effect = NativeEffectClass::Provider;
        let mut registry = NativeNodeRegistry::default();
        registry.register_descriptor(provider)?;
        let provider_submission = || {
            submission(BTreeMap::from([(
                NodeId::from("provider"),
                PromptNode {
                    class_type: "ProviderOutput".to_owned(),
                    inputs: BTreeMap::new(),
                    unknown: BTreeMap::new(),
                },
            )]))
        };
        assert_eq!(
            PromptCompiler::new(&registry)
                .compile(provider_submission())
                .expect_err("provider plans require a host-selected registry"),
            PromptCompileError::ProviderRegistryPinRequired
        );

        let pin = NativeProviderRegistryPin::checked(
            9,
            "a".repeat(64),
            vec!["b".repeat(64), "c".repeat(64)],
        )?;
        let plan = PromptCompiler::new(&registry)
            .with_provider_registry_pin(pin.clone())?
            .compile(provider_submission())?;
        assert_eq!(plan.provider_registry_pin(), Some(&pin));
        plan.validate_provider_execution_identity()?;

        let mut tampered = plan;
        tampered
            .extra_data
            .insert("tampered".to_owned(), json!(true));
        assert_eq!(
            tampered
                .validate_provider_execution_identity()
                .expect_err("execution-relevant plan drift invalidates the seal"),
            PromptCompileError::ProviderPlanIdentityMismatch
        );

        let local = descriptor("LocalOutput", true)?;
        let mut local_registry = NativeNodeRegistry::default();
        local_registry.register_descriptor(local)?;
        let local_plan = PromptCompiler::new(&local_registry)
            .with_provider_registry_pin(pin)?
            .compile(submission(BTreeMap::from([(
                NodeId::from("local"),
                PromptNode {
                    class_type: "LocalOutput".to_owned(),
                    inputs: BTreeMap::new(),
                    unknown: BTreeMap::new(),
                },
            )])))?;
        assert!(local_plan.provider_execution.is_none());
        Ok(())
    }

    pub(crate) fn val_domain_004_prompt_case_results()
    -> Result<Vec<(&'static str, bool)>, Box<dyn std::error::Error>> {
        val_domain_004_prompt_graph_and_lazy_dependencies_compile_deterministically()?;
        val_domain_004_invalid_graphs_fail_with_node_addressable_errors()?;
        val_domain_004_hidden_inputs_are_host_injected()?;
        integer_literals_preserve_u64_and_signed_values_exactly()?;
        dynamic_input_resolution_checks_every_template()?;
        Ok(vec![
            ("prompt_graph_lazy_demand_closure", true),
            ("prompt_node_addressable_validation", true),
            ("prompt_host_hidden_input_injection", true),
            ("prompt_integer_literal_signed_unsigned_exactness", true),
            ("prompt_dynamic_input_template_resolution", true),
        ])
    }
}
