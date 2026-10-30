use crate::{
    ComponentHost, ComponentHostError, InstalledVerifiedPlugin, InvocationInputs,
    InvocationMediaContent, capabilities::artifact_value_identity, require_exact_port_type,
};
use futures::future::BoxFuture;
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use sim_nodes::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeDynamicInputDescriptor, NativeHandleKind,
    NativeHandleStoreIdentity, NativeHandleType, NativeInputDescriptor, NativeNode,
    NativeNodeContext, NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle, NativeOutputDescriptor,
    NativeOutputEffectRequest, NativeOutputNamespace, NativeOutputShape, NativePortCardinality,
    NativePrimitive, NativeSourceValueClass, NativeStoredPayload, NativeStructuredValue,
    NativeTypeUnion, NativeValue, NativeValueType, native_plugin_source_type_projection,
};
use sim_plugin_sdk::{
    ArtifactValue, CachePolicy, DeterminismPolicy, EffectPolicy, ModelValue, PluginNode,
    PluginPort, PluginValue, PluginValueRepresentation, PortCardinality, PortDirection,
    PortPresence, ProviderBindingClaim, ScalarValue, TensorValue, TypeRegistry, ValueFamily,
};
use sim_runtime::{
    NativeNodeRegistry, NativeNodeRegistryError, NativeProviderBindingActivation,
    NativeProviderBindingActivationSet, ProviderTransportEncoding, ProviderTransportPort,
    ProviderTransportRequest, ProviderTransportResponse, ProviderTransportValue,
};
use sim_types::CancellationToken;
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;
#[cfg(test)]
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum PluginRegistryAdapterError {
    #[error(transparent)]
    ComponentHost(#[from] ComponentHostError),
    #[error(transparent)]
    Registry(#[from] NativeNodeRegistryError),
    #[error("plugin node `{node}` has an invalid port contract: {message}")]
    InvalidPort { node: String, message: String },
}

pub fn registry_with_installed_plugins(
    base: &NativeNodeRegistry,
    component_host: &ComponentHost,
) -> Result<NativeNodeRegistry, PluginRegistryAdapterError> {
    registry_with_plugins(
        base,
        component_host,
        component_host.installed_plugins()?,
        component_host.verified_generation()?,
    )
}

pub(crate) fn registry_with_plugins(
    base: &NativeNodeRegistry,
    component_host: &ComponentHost,
    plugins: Vec<InstalledVerifiedPlugin>,
    generation: crate::VerifiedComponentGeneration,
) -> Result<NativeNodeRegistry, PluginRegistryAdapterError> {
    let type_registry =
        TypeRegistry::built_in().map_err(|error| PluginRegistryAdapterError::InvalidPort {
            node: "type-registry".to_owned(),
            message: error.to_string(),
        })?;
    let mut ordinary_bindings = Vec::new();
    let mut provider_activation_sets = Vec::new();
    for plugin in plugins {
        let provider_binding_set = plugin.manifest().provider_binding.as_ref();
        let claims = provider_binding_set
            .into_iter()
            .flat_map(|set| &set.bindings)
            .map(|claim| (claim.node_id.as_str(), claim))
            .collect::<BTreeMap<_, _>>();
        let mut used_claims = BTreeMap::new();
        let mut provider_bindings = Vec::new();
        for node in &plugin.manifest().nodes {
            let descriptor = native_descriptor(node, &type_registry)?;
            let declared_disposition = base.binding_declared_disposition(&node.id);
            let implementation_version = if declared_disposition
                == Some(sim_nodes::NativeNodeBindingDisposition::ProviderRequired)
            {
                base.descriptor(&node.id)
                    .ok_or_else(|| provider_binding_error(&node.id, "descriptor is missing"))?
                    .implementation_version
                    .clone()
            } else {
                descriptor.implementation_version.clone()
            };
            let provider_binding = if declared_disposition
                == Some(sim_nodes::NativeNodeBindingDisposition::ProviderRequired)
            {
                Some(
                    claims
                        .get(node.id.as_str())
                        .copied()
                        .ok_or_else(|| {
                            provider_binding_error(&node.id, "signed provider claim is missing")
                        })?
                        .clone(),
                )
            } else {
                None
            };
            let implementation: Arc<dyn NativeNode> = Arc::new(PluginNativeNode {
                component_host: component_host.clone(),
                plugin: plugin.clone(),
                node: node.clone(),
                type_registry: type_registry.clone(),
                implementation_version,
                provider_binding,
                provider_descriptor: (declared_disposition
                    == Some(sim_nodes::NativeNodeBindingDisposition::ProviderRequired))
                .then(|| {
                    base.descriptor(&node.id)
                        .cloned()
                        .ok_or_else(|| provider_binding_error(&node.id, "descriptor is missing"))
                })
                .transpose()?,
            });
            match declared_disposition {
                Some(sim_nodes::NativeNodeBindingDisposition::ProviderRequired) => {
                    let claim = claims.get(node.id.as_str()).copied().ok_or_else(|| {
                        provider_binding_error(&node.id, "signed provider claim is missing")
                    })?;
                    validate_provider_node_projection(base, node, claim)?;
                    if used_claims.insert(node.id.as_str(), claim).is_some() {
                        return Err(provider_binding_error(
                            &node.id,
                            "signed provider claim is duplicated",
                        ));
                    }
                    provider_bindings.push(NativeProviderBindingActivation::new(
                        claim.clone(),
                        implementation,
                    ));
                }
                Some(_) => {
                    return Err(provider_binding_error(
                        &node.id,
                        "plugin node collides with a non-provider native binding",
                    ));
                }
                None => {
                    if claims.contains_key(node.id.as_str()) {
                        return Err(provider_binding_error(
                            &node.id,
                            "provider claim targets a non-provider plugin node",
                        ));
                    }
                    ordinary_bindings.push((descriptor, implementation, native_presentation(node)));
                }
            }
        }
        if used_claims.len() != claims.len() {
            return Err(provider_binding_error(
                plugin.binding().signed_plugin_identifier(),
                "signed provider claim set contains an undeclared component node",
            ));
        }
        if let Some(binding_set) = provider_binding_set {
            let deployment = generation
                .components()
                .iter()
                .find(|deployment| deployment.extension_id() == plugin.extension_id())
                .ok_or_else(|| {
                    provider_binding_error(
                        plugin.extension_id(),
                        "verified component deployment is missing",
                    )
                })?;
            provider_activation_sets.push(NativeProviderBindingActivationSet::checked(
                generation.profile_id(),
                generation.generation(),
                generation.snapshot_sha256(),
                deployment.component_sha256(),
                deployment.authorization_generation(),
                binding_set.clone(),
                provider_bindings,
            )?);
        } else if !provider_bindings.is_empty() {
            return Err(provider_binding_error(
                plugin.binding().signed_plugin_identifier(),
                "provider bindings require a signed claim set",
            ));
        }
    }
    let mut registry = base.clone();
    registry.register_bound_batch_with_presentations(ordinary_bindings)?;
    for activation in provider_activation_sets {
        registry.activate_provider_binding_set(activation)?;
    }
    Ok(registry)
}

fn provider_binding_error(
    node: impl Into<String>,
    message: impl Into<String>,
) -> PluginRegistryAdapterError {
    PluginRegistryAdapterError::InvalidPort {
        node: node.into(),
        message: message.into(),
    }
}

fn validate_provider_node_projection(
    base: &NativeNodeRegistry,
    node: &PluginNode,
    claim: &ProviderBindingClaim,
) -> Result<(), PluginRegistryAdapterError> {
    let type_registry = TypeRegistry::built_in().map_err(|error| {
        provider_binding_error(&node.id, format!("type registry is invalid: {error}"))
    })?;
    let descriptor = base
        .descriptor(&node.id)
        .ok_or_else(|| provider_binding_error(&node.id, "descriptor is missing"))?;
    let presentation = base
        .presentation(&node.id)
        .ok_or_else(|| provider_binding_error(&node.id, "presentation is missing"))?;
    let output_names = node
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
        .map(|port| port.name.as_str())
        .collect::<Vec<_>>();
    let expected_digest = base
        .provider_binding_contract_sha256(
            &node.id,
            &claim.transport_schema.to_string(),
            &claim.materializer_schema.to_string(),
        )?
        .ok_or_else(|| provider_binding_error(&node.id, "provider contract is missing"))?;
    if node.determinism != DeterminismPolicy::External
        || node.cache != CachePolicy::Never
        || node.effects != EffectPolicy::Provider
        || node.display_name != presentation.display_name
        || node.category != presentation.category
        || output_names
            != presentation
                .output_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        || claim.contract_sha256 != expected_digest
        || !provider_ports_match_descriptor(node, descriptor, &type_registry)?
    {
        return Err(provider_binding_error(
            &node.id,
            "signed provider node does not match the canonical native contract",
        ));
    }
    Ok(())
}

fn provider_ports_match_descriptor(
    node: &PluginNode,
    descriptor: &NativeNodeDescriptor,
    registry: &TypeRegistry,
) -> Result<bool, PluginRegistryAdapterError> {
    let input_ports = node
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Input)
        .collect::<Vec<_>>();
    let output_ports = node
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
        .collect::<Vec<_>>();
    let projected_inputs = provider_input_descriptors(node, descriptor)?;
    if input_ports.len() != projected_inputs.len() + descriptor.dynamic_inputs.len()
        || output_ports.len() != descriptor.outputs.len()
    {
        return Ok(false);
    }
    for input in &projected_inputs {
        let Some(port) = input_ports.iter().find(|port| port.id == input.name) else {
            return Ok(false);
        };
        if !plugin_port_matches_input(port, input, PortCardinality::Singular, registry)?
            || port.presence
                != if input.hidden {
                    PortPresence::Hidden
                } else if input.required {
                    PortPresence::Required
                } else {
                    PortPresence::Optional
                }
        {
            return Ok(false);
        }
    }
    for dynamic in &descriptor.dynamic_inputs {
        let group_name = descriptor
            .source_schema
            .as_ref()
            .and_then(|schema| {
                schema.dynamic_inputs.iter().find_map(|source| {
                    if source.identity != dynamic.name_template
                        || source.input.name != dynamic.input.name
                    {
                        return None;
                    }
                    source.extra.iter().find_map(|field| match &field.value {
                        sim_nodes::NativeSchemaValue::String { value }
                            if field.name == "autogrow_group" =>
                        {
                            Some(value.as_str())
                        }
                        _ => None,
                    })
                })
            })
            .unwrap_or(&dynamic.input.name);
        let Some(port) = input_ports.iter().find(|port| port.id == group_name) else {
            return Ok(false);
        };
        if !plugin_port_matches_input(port, &dynamic.input, PortCardinality::List, registry)?
            || (dynamic.minimum_count == 0) != (port.presence == PortPresence::Optional)
        {
            return Ok(false);
        }
    }
    for (port, output) in output_ports.iter().zip(&descriptor.outputs) {
        let value_type = native_value_type(
            port,
            registry.family(&port.type_id).map_err(|error| {
                provider_binding_error(&node.id, format!("output type is invalid: {error}"))
            })?,
        )
        .map_err(|error| provider_binding_error(&node.id, error.to_string()))?;
        if port.id != output.name
            || port.name != output.name
            || port.cardinality
                != if output.is_list {
                    PortCardinality::List
                } else {
                    PortCardinality::Singular
                }
            || port.presence != PortPresence::Required
            || value_type != output.produced_type
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn provider_input_descriptors(
    node: &PluginNode,
    descriptor: &NativeNodeDescriptor,
) -> Result<Vec<NativeInputDescriptor>, PluginRegistryAdapterError> {
    let mut inputs = descriptor.inputs.clone();
    let Some(selector_name) = provider_structured_selector(&node.id) else {
        return Ok(inputs);
    };
    let Some(model) = descriptor.source_schema.as_ref().and_then(|schema| {
        schema
            .inputs
            .iter()
            .find(|input| input.name == selector_name)
    }) else {
        return Ok(inputs);
    };
    let options = model
        .structured_options()
        .map_err(|error| provider_binding_error(&node.id, error.to_string()))?;
    let option_count = options.len();
    let mut fields: BTreeMap<String, (NativeInputDescriptor, usize)> = BTreeMap::new();
    for option in options {
        for field in option.fields {
            if field.path != [field.schema.name.clone()]
                || inputs.iter().any(|input| input.name == field.schema.name)
            {
                return Err(provider_binding_error(
                    &node.id,
                    "unsupported provider field path",
                ));
            }
            let accepted_types = sim_nodes::native_value_types_for_input_schema(&field.schema)
                .map_err(|error| provider_binding_error(&node.id, error.to_string()))?;
            let allows_literal = !accepted_types
                .members()
                .iter()
                .any(|value_type| matches!(value_type, NativeValueType::Handle(_)));
            let input = NativeInputDescriptor {
                name: field.schema.name.clone(),
                accepted_types,
                required: field.required,
                hidden: false,
                lazy: field.lazy,
                cardinality: NativePortCardinality::Scalar,
                allows_literal,
            };
            if let Some((existing, count)) = fields.get_mut(&input.name) {
                if existing != &input {
                    return Err(provider_binding_error(
                        &node.id,
                        "conflicting provider field contracts",
                    ));
                }
                *count += 1;
            } else {
                fields.insert(input.name.clone(), (input, 1));
            }
        }
    }
    for (mut input, count) in fields.into_values() {
        input.required &= count == option_count;
        inputs.push(input);
    }
    Ok(inputs)
}

fn plugin_port_matches_input(
    port: &PluginPort,
    input: &NativeInputDescriptor,
    cardinality: PortCardinality,
    registry: &TypeRegistry,
) -> Result<bool, PluginRegistryAdapterError> {
    let family = registry
        .family(&port.type_id)
        .map_err(|error| provider_binding_error(&port.id, error.to_string()))?;
    let value_type = native_value_type(port, family)
        .map_err(|error| provider_binding_error(&port.id, error.to_string()))?;
    Ok(port.cardinality == cardinality
        && port.hidden == input.hidden
        && port.lazy == input.lazy
        && input.accepted_types.members() == [value_type])
}

fn native_presentation(node: &PluginNode) -> NativeNodePresentation {
    NativeNodePresentation {
        display_name: node.display_name.clone(),
        category: node.category.clone(),
        description: String::new(),
        output_names: node
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
            .map(|port| port.name.clone())
            .collect(),
        search_aliases: Vec::new(),
        is_deprecated: false,
        is_experimental: false,
    }
}

struct PluginNativeNode {
    component_host: ComponentHost,
    plugin: InstalledVerifiedPlugin,
    node: PluginNode,
    type_registry: TypeRegistry,
    implementation_version: String,
    provider_binding: Option<ProviderBindingClaim>,
    provider_descriptor: Option<NativeNodeDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ImportedHandle {
    Exact(NativeOpaqueHandle),
    Ambiguous,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ImportedHandles {
    store_identity: NativeHandleStoreIdentity,
    handles: BTreeMap<(String, String), ImportedHandle>,
}

impl ImportedHandles {
    fn new(store_identity: NativeHandleStoreIdentity) -> Self {
        Self {
            store_identity,
            handles: BTreeMap::new(),
        }
    }
}

impl NativeNode for PluginNativeNode {
    fn class_type(&self) -> &str {
        &self.node.id
    }

    fn implementation_version(&self) -> &str {
        &self.implementation_version
    }

    fn implementation_namespace(&self) -> &str {
        self.plugin.binding().signed_plugin_identifier()
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            context.validate().map_err(plugin_failure)?;
            let profile_id = self.plugin.authorization().capabilities().profile_id();
            let (mut invocation_inputs, imported_handles) = invocation_inputs(
                &self.node,
                inputs,
                &self.type_registry,
                profile_id,
                &context,
                self.provider_descriptor.as_ref(),
            )?;
            let (outputs, result_effects) = if let Some(provider_binding) = &self.provider_binding {
                sim_runtime::validate_native_provider_schemas(
                    &provider_binding.transport_schema,
                    &provider_binding.materializer_schema,
                )
                .map_err(plugin_failure)?;
                context.provider_execution().map_err(plugin_failure)?;
                prepare_beeble_video_media(
                    &self.node,
                    &mut invocation_inputs,
                    &imported_handles,
                    &context,
                )
                .await?;
                let request = provider_transport_request(
                    &self.node,
                    &invocation_inputs,
                    &imported_handles,
                    &context,
                )?;
                let mut result = self
                    .component_host
                    .execute_provider(
                        &self.plugin,
                        &self.node.id,
                        invocation_inputs,
                        request.to_bytes().map_err(plugin_failure)?,
                        context.clone(),
                    )
                    .await
                    .map_err(plugin_failure)?;
                if !result.outputs.is_empty()
                    || !result.output_presence.is_empty()
                    || !result.effects.outputs.is_empty()
                    || !result.effects.logs.is_empty()
                    || !result.effects.ui_state.is_empty()
                    || !result.effects.routes.is_empty()
                {
                    return Err(unmaterialized_plugin_output("provider response"));
                }
                let materialized = result
                    .take_materialized_provider_response(
                        self.plugin.binding().signed_plugin_identifier(),
                        &self.node.id,
                    )
                    .map_err(plugin_failure)?;
                let mut resolved = result.take_resolved_provider_results().into_iter();
                let response = resolved
                    .next()
                    .ok_or_else(|| unmaterialized_plugin_output("provider receipt"))?;
                if resolved.next().is_some() {
                    return Err(unmaterialized_plugin_output("provider receipt count"));
                }
                let response = match materialized {
                    Some(materialized) => materialized,
                    None => ProviderTransportResponse::from_bytes(response.response())
                        .map_err(plugin_failure)?,
                };
                let outputs = provider_transport_outputs(
                    &self.node,
                    &response,
                    &self.type_registry,
                    self.plugin.binding().signed_plugin_identifier(),
                    &context,
                )?;
                if let Err(error) = context.cancellation.check() {
                    rollback_native_provider_outputs(&context, &outputs)?;
                    return Err(plugin_failure(error));
                }
                (outputs, result.effects)
            } else {
                let result = self
                    .component_host
                    .execute_plugin(
                        &self.plugin,
                        &self.node.id,
                        invocation_inputs,
                        context.clone(),
                    )
                    .await
                    .map_err(plugin_failure)?;
                let outputs = invocation_outputs(
                    &self.node,
                    &result.outputs,
                    &result.output_presence,
                    &self.type_registry,
                    profile_id,
                    &context,
                    &imported_handles,
                )?;
                (outputs, result.effects)
            };
            let ui = (!result_effects.ui_state.is_empty()
                || !result_effects.logs.is_empty()
                || !result_effects.routes.is_empty())
            .then(|| {
                serde_json::json!({
                    "plugin_ui": &result_effects.ui_state,
                    "plugin_logs": &result_effects.logs,
                    "plugin_routes": &result_effects.routes,
                })
            });
            let effect_service = context.prepared_effects().map_err(plugin_failure)?;
            let requests = result_effects
                .outputs
                .iter()
                .enumerate()
                .map(|(index, proposal)| {
                    let namespace = match proposal.namespace.as_str() {
                        "output" | "outputs" => NativeOutputNamespace::Output,
                        "temp" | "temporary" => NativeOutputNamespace::Temporary,
                        _ => return Err(unmaterialized_plugin_effect("output namespace")),
                    };
                    let (filename_prefix, extension) = proposal
                        .name
                        .rsplit_once('.')
                        .ok_or_else(|| unmaterialized_plugin_effect("output filename"))?;
                    NativeOutputEffectRequest::checked(
                        namespace,
                        filename_prefix,
                        extension,
                        u32::try_from(index).map_err(plugin_failure)?,
                        NativeOutputShape::File,
                        Arc::from(proposal.bytes.clone()),
                        effect_service.maximum_output_bytes(),
                    )
                    .map_err(plugin_failure)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut effects = Vec::with_capacity(requests.len());
            for request in requests {
                match effect_service.prepare_output(request, &context.cancellation) {
                    Ok(effect) => effects.push(effect),
                    Err(error) => {
                        for effect in effects.iter().rev() {
                            effect_service
                                .rollback_prepared(effect)
                                .map_err(plugin_failure)?;
                        }
                        return Err(plugin_failure(error));
                    }
                }
            }
            Ok(NativeNodeOutcome::Values {
                outputs,
                ui,
                effects,
            })
        })
    }
}

fn native_descriptor(
    node: &PluginNode,
    registry: &TypeRegistry,
) -> Result<sim_nodes::NativeNodeDescriptor, PluginRegistryAdapterError> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for port in &node.ports {
        let family = registry.family(&port.type_id).map_err(|error| {
            PluginRegistryAdapterError::InvalidPort {
                node: node.id.clone(),
                message: error.to_string(),
            }
        })?;
        let value_type = native_value_type(port, family).map_err(|error| {
            PluginRegistryAdapterError::InvalidPort {
                node: node.id.clone(),
                message: error.to_string(),
            }
        })?;
        match port.direction {
            PortDirection::Input => inputs.push(NativeInputDescriptor {
                name: port.id.clone(),
                accepted_types: NativeTypeUnion::new([value_type.clone()]).map_err(|error| {
                    PluginRegistryAdapterError::InvalidPort {
                        node: node.id.clone(),
                        message: error.to_string(),
                    }
                })?,
                required: port.presence == PortPresence::Required,
                hidden: port.hidden,
                lazy: port.lazy,
                cardinality: if port.cardinality == PortCardinality::List {
                    NativePortCardinality::List
                } else {
                    NativePortCardinality::Scalar
                },
                allows_literal: !matches!(value_type, NativeValueType::Handle(_)),
            }),
            PortDirection::Output => outputs.push(NativeOutputDescriptor {
                name: port.name.clone(),
                produced_type: if port.presence == PortPresence::Optional {
                    NativeValueType::Any
                } else {
                    value_type
                },
                is_list: port.cardinality == PortCardinality::List,
            }),
        }
    }
    let source_schema = sim_nodes::NativeDescriptorSchemaMetadata::compatibility(
        sim_nodes::NativeSchemaProvenance::Plugin,
        node.ports
            .iter()
            .filter(|port| port.direction == PortDirection::Input)
            .map(|port| (port.id.clone(), port.type_id.to_string())),
        std::iter::empty(),
        node.ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
            .map(|port| (port.name.clone(), port.type_id.to_string())),
    );
    let descriptor = sim_nodes::NativeNodeDescriptor {
        schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
        class_type: node.id.clone(),
        implementation_version: node.version.to_string(),
        source_schema: Some(source_schema),
        inputs,
        dynamic_inputs: Vec::new(),
        outputs,
        output_node: node.effects != EffectPolicy::Pure,
        effect: match node.effects {
            EffectPolicy::Pure => sim_nodes::NativeEffectClass::Pure,
            EffectPolicy::Transactional => sim_nodes::NativeEffectClass::WritesArtifact,
            EffectPolicy::Provider => sim_nodes::NativeEffectClass::Provider,
        },
        cache: if node.cache == CachePolicy::Never {
            sim_nodes::NativeCachePolicy::Never
        } else {
            sim_nodes::NativeCachePolicy::InputIdentity
        },
    };
    descriptor
        .validate()
        .map_err(|error| PluginRegistryAdapterError::InvalidPort {
            node: node.id.clone(),
            message: error.to_string(),
        })?;
    Ok(descriptor)
}

fn native_value_type(
    port: &PluginPort,
    family: ValueFamily,
) -> Result<NativeValueType, sim_nodes::NativeNodeContractError> {
    let projection =
        native_plugin_source_type_projection(port.type_id.name()).map_err(|error| {
            sim_nodes::NativeNodeContractError::InvalidSourceSchema(error.to_string())
        })?;
    let value_type = if projection.value_class() == NativeSourceValueClass::SchemaScalar {
        NativeValueType::NamedPreservedUnknown(projection.source_type().to_owned())
    } else {
        projection.value_type().map_err(|error| {
            sim_nodes::NativeNodeContractError::InvalidSourceSchema(error.to_string())
        })?
    };
    let representation_matches = match family {
        ValueFamily::Scalar => !matches!(value_type, NativeValueType::Handle(_)),
        ValueFamily::Tensor | ValueFamily::Artifact | ValueFamily::Model => {
            matches!(value_type, NativeValueType::Handle(_))
        }
    };
    if !representation_matches {
        return Err(sim_nodes::NativeNodeContractError::InvalidSourceSchema(
            format!(
                "plugin type `{}` disagrees with canonical source value class",
                port.type_id
            ),
        ));
    }
    Ok(value_type)
}

fn native_handle_type(
    port: &PluginPort,
) -> Result<NativeHandleType, sim_nodes::NativeNodeContractError> {
    native_plugin_source_type_projection(port.type_id.name())
        .map_err(|error| {
            sim_nodes::NativeNodeContractError::InvalidSourceSchema(error.to_string())
        })?
        .handle_type()
        .map_err(|error| {
            sim_nodes::NativeNodeContractError::InvalidSourceSchema(error.to_string())
        })?
        .ok_or_else(|| {
            sim_nodes::NativeNodeContractError::InvalidSourceSchema(format!(
                "plugin type `{}` is not handle-backed",
                port.type_id
            ))
        })
}

fn invocation_inputs(
    node: &PluginNode,
    mut values: BTreeMap<String, NativeValue>,
    registry: &TypeRegistry,
    profile_id: &str,
    context: &NativeNodeContext,
    provider_descriptor: Option<&NativeNodeDescriptor>,
) -> Result<(InvocationInputs, ImportedHandles), NativeNodeFailure> {
    let result = (|| {
        let mut inputs = InvocationInputs::default();
        let mut imported_handles = ImportedHandles::new(context.handle_store().identity());
        let prepare_claude_images = provider_descriptor.is_some() && node.id == "ClaudeNode";
        if let Some(selector_name) =
            provider_descriptor.and_then(|_| provider_structured_selector(&node.id))
        {
            prepare_provider_structured_inputs(&mut values, provider_descriptor, selector_name)?;
        }
        let mut provider_images = Vec::new();
        let mut beeble_images = Vec::new();
        let mut hitpaw_images = Vec::new();
        let mut bfl_ultra_images = Vec::new();
        for port in node
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Input)
        {
            let dynamic = provider_descriptor.and_then(|descriptor| {
                descriptor
                    .dynamic_inputs
                    .iter()
                    .find(|dynamic| {
                        dynamic.input.name == port.id
                            || descriptor.source_schema.as_ref().is_some_and(|schema| {
                                schema.dynamic_inputs.iter().any(|source| {
                                    source.identity == dynamic.name_template
                                        && source.input.name == dynamic.input.name
                                        && source.extra.iter().any(|field| {
                                            field.name == "autogrow_group"
                                                && matches!(&field.value, sim_nodes::NativeSchemaValue::String { value } if value == &port.id)
                                        })
                                })
                            })
                    })
            });
            let value = if let Some(dynamic) = dynamic {
                let source = provider_descriptor
                    .and_then(|descriptor| descriptor.source_schema.as_ref())
                    .and_then(|schema| {
                        schema.dynamic_inputs.iter().find(|source| {
                            source.identity == dynamic.name_template
                                && source.input.name == dynamic.input.name
                        })
                    });
                let dynamic_values = take_dynamic_values(&mut values, dynamic, source)?;
                (!dynamic_values.is_empty()).then_some(NativeValue::List {
                    values: dynamic_values,
                })
            } else {
                values.remove(&port.id)
            };
            let Some(value) = value else {
                inputs.set_absent(&port.id);
                continue;
            };
            let values = if port.cardinality == PortCardinality::List {
                match value {
                    NativeValue::List { values } => values,
                    _ => return Err(invalid_value(&port.id)),
                }
            } else {
                if matches!(value, NativeValue::List { .. }) {
                    return Err(invalid_value(&port.id));
                }
                vec![value]
            };
            if prepare_claude_images && port.id == "images" && values.len() > 20 {
                return Err(invalid_value("images"));
            }
            let mut plugin_values = Vec::with_capacity(values.len());
            for value in values {
                let (plugin_value, imported_handle) =
                    plugin_value(port, value, registry, profile_id, context)?;
                if prepare_claude_images && port.id == "images" {
                    let (_, handle) = imported_handle
                        .as_ref()
                        .ok_or_else(|| invalid_value("images"))?;
                    provider_images.push((plugin_value.clone(), handle.clone()));
                }
                if let Some(kind) =
                    provider_descriptor.and_then(|_| BeeblePngInput::for_port(&node.id, &port.id))
                {
                    let (_, handle) = imported_handle
                        .as_ref()
                        .ok_or_else(|| invalid_value(&port.id))?;
                    beeble_images.push((
                        port.id.clone(),
                        kind,
                        plugin_value.clone(),
                        handle.clone(),
                    ));
                }
                if provider_descriptor.is_some() && hitpaw_uses_first_image(&node.id, &port.id) {
                    let (_, handle) = imported_handle
                        .as_ref()
                        .ok_or_else(|| invalid_value(&port.id))?;
                    hitpaw_images.push((plugin_value.clone(), handle.clone()));
                }
                if provider_descriptor.is_some() && bfl_ultra_uses_first_image(&node.id, &port.id) {
                    let (_, handle) = imported_handle
                        .as_ref()
                        .ok_or_else(|| invalid_value(&port.id))?;
                    bfl_ultra_images.push((plugin_value.clone(), handle.clone()));
                }
                if let Some((key, handle)) = imported_handle {
                    remember_imported_handle(&mut imported_handles, key, handle);
                }
                plugin_values.push(plugin_value);
            }
            inputs.set_present(&port.id, plugin_values);
        }
        if let Some((unknown, _)) = values.into_iter().next() {
            return Err(invalid_value(&unknown));
        }
        if prepare_claude_images {
            prepare_claude_input_media(&mut inputs, &provider_images, context)?;
        }
        prepare_beeble_input_media(&mut inputs, &beeble_images, context)?;
        prepare_hitpaw_input_media(&mut inputs, &hitpaw_images, context)?;
        prepare_bfl_ultra_input_media(&mut inputs, &bfl_ultra_images, context)?;
        Ok((inputs, imported_handles))
    })();
    if context.cancellation.is_cancelled() {
        return Err(NativeNodeFailure {
            code: "plugin_input_preparation_cancelled".to_owned(),
            message: "Plugin input preparation was cancelled".to_owned(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: false,
        });
    }
    result
}

#[derive(Clone, Copy)]
enum BeeblePngInput {
    SourceImage,
    ReferenceImage,
    Mask,
}

impl BeeblePngInput {
    fn for_port(node_id: &str, port_id: &str) -> Option<Self> {
        match (node_id, port_id) {
            ("BeebleSwitchXImageEdit", "image") => Some(Self::SourceImage),
            ("BeebleSwitchXImageEdit" | "BeebleSwitchXVideoEdit", "reference_image") => {
                Some(Self::ReferenceImage)
            }
            ("BeebleSwitchXImageEdit", "alpha_keyframe" | "alpha_mask")
            | ("BeebleSwitchXVideoEdit", "alpha_keyframe") => Some(Self::Mask),
            _ => None,
        }
    }
}

pub(crate) fn beeble_uses_first_image(node_id: &str, port_id: &str) -> bool {
    matches!(
        BeeblePngInput::for_port(node_id, port_id),
        Some(BeeblePngInput::SourceImage | BeeblePngInput::ReferenceImage)
    )
}

pub(crate) fn hitpaw_uses_first_image(node_id: &str, port_id: &str) -> bool {
    node_id == "HitPawGeneralImageEnhance" && port_id == "image"
}

pub(crate) fn bfl_ultra_uses_first_image(node_id: &str, port_id: &str) -> bool {
    node_id == "FluxProUltraImageNode" && port_id == "image_prompt"
}

fn prepare_bfl_ultra_input_media(
    inputs: &mut InvocationInputs,
    values: &[(PluginValue, NativeOpaqueHandle)],
    context: &NativeNodeContext,
) -> Result<(), NativeNodeFailure> {
    let expected =
        NativeHandleType::new(NativeHandleKind::Image, "IMAGE").map_err(plugin_failure)?;
    for (value_index, (value, handle)) in values.iter().enumerate() {
        let payload = context
            .handle_store()
            .resolve(handle, &expected, &context.cancellation)
            .map_err(plugin_failure)?;
        let NativeStoredPayload::Tensor(tensor) = payload.as_ref() else {
            return Err(invalid_value("image_prompt"));
        };
        let bytes = context.prepare_provider_image_png(
            tensor
                .image()
                .ok_or_else(|| invalid_value("image_prompt"))?,
            0,
            Some(2048 * 2048),
        )?;
        let source_digest =
            hex_digest(&Sha256::digest(value.abi_bytes().map_err(plugin_failure)?).into());
        let content_digest = hex_digest(&Sha256::digest(&bytes).into());
        inputs
            .register_media_batch_item(
                "image_prompt",
                u32::try_from(value_index).map_err(plugin_failure)?,
                0,
                InvocationMediaContent::new("image/png", source_digest, content_digest, bytes),
            )
            .map_err(plugin_failure)?;
    }
    Ok(())
}

fn prepare_hitpaw_input_media(
    inputs: &mut InvocationInputs,
    values: &[(PluginValue, NativeOpaqueHandle)],
    context: &NativeNodeContext,
) -> Result<(), NativeNodeFailure> {
    let expected =
        NativeHandleType::new(NativeHandleKind::Image, "IMAGE").map_err(plugin_failure)?;
    for (value_index, (value, handle)) in values.iter().enumerate() {
        let payload = context
            .handle_store()
            .resolve(handle, &expected, &context.cancellation)
            .map_err(plugin_failure)?;
        let NativeStoredPayload::Tensor(tensor) = payload.as_ref() else {
            return Err(invalid_value("image"));
        };
        let bytes = context.prepare_provider_image_png(
            tensor.image().ok_or_else(|| invalid_value("image"))?,
            0,
            None,
        )?;
        let source_digest =
            hex_digest(&Sha256::digest(value.abi_bytes().map_err(plugin_failure)?).into());
        let content_digest = hex_digest(&Sha256::digest(&bytes).into());
        inputs
            .register_media_batch_item(
                "image",
                u32::try_from(value_index).map_err(plugin_failure)?,
                0,
                InvocationMediaContent::new("image/png", source_digest, content_digest, bytes),
            )
            .map_err(plugin_failure)?;
    }
    Ok(())
}

fn prepare_beeble_input_media(
    inputs: &mut InvocationInputs,
    values: &[(String, BeeblePngInput, PluginValue, NativeOpaqueHandle)],
    context: &NativeNodeContext,
) -> Result<(), NativeNodeFailure> {
    for (port_id, kind, value, handle) in values {
        let expected = match kind {
            BeeblePngInput::Mask => NativeHandleType::new(NativeHandleKind::Mask, "MASK"),
            BeeblePngInput::SourceImage | BeeblePngInput::ReferenceImage => {
                NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
            }
        }
        .map_err(plugin_failure)?;
        let payload = context
            .handle_store()
            .resolve(handle, &expected, &context.cancellation)
            .map_err(plugin_failure)?;
        let NativeStoredPayload::Tensor(tensor) = payload.as_ref() else {
            return Err(invalid_value(port_id));
        };
        let bytes = match kind {
            BeeblePngInput::SourceImage => context.prepare_provider_source_image_png(
                tensor.image().ok_or_else(|| invalid_value(port_id))?,
                2_770_000,
            )?,
            BeeblePngInput::ReferenceImage => context.prepare_provider_image_png(
                tensor.image().ok_or_else(|| invalid_value(port_id))?,
                0,
                Some(2_770_000),
            )?,
            BeeblePngInput::Mask => {
                context.prepare_provider_mask_png(tensor.tensor(), 2_770_000)?
            }
        };
        let source_digest =
            hex_digest(&Sha256::digest(value.abi_bytes().map_err(plugin_failure)?).into());
        let content_digest = hex_digest(&Sha256::digest(&bytes).into());
        inputs
            .register_media_batch_item(
                port_id,
                0,
                0,
                InvocationMediaContent::new("image/png", source_digest, content_digest, bytes),
            )
            .map_err(plugin_failure)?;
    }
    Ok(())
}

async fn prepare_beeble_video_media(
    node: &PluginNode,
    inputs: &mut InvocationInputs,
    imported_handles: &ImportedHandles,
    context: &NativeNodeContext,
) -> Result<(), NativeNodeFailure> {
    if node.id != "BeebleSwitchXVideoEdit" {
        return Ok(());
    }
    let source_values = inputs
        .values
        .get("video")
        .and_then(Option::as_ref)
        .ok_or_else(|| invalid_value("video"))?;
    let [source_value] = source_values.as_slice() else {
        return Err(invalid_value("video"));
    };
    let resolve = |value: &PluginValue, port: &str, kind: NativeHandleKind, source_type: &str| {
        let digest = value_digest(value).ok_or_else(|| invalid_value(port))?;
        let key = (value.type_id().to_string(), digest.to_owned());
        let Some(ImportedHandle::Exact(handle)) = imported_handles.handles.get(&key) else {
            return Err(invalid_value(port));
        };
        validate_value_digest(handle, value, port)?;
        let expected = NativeHandleType::new(kind, source_type).map_err(plugin_failure)?;
        context
            .handle_store()
            .resolve(handle, &expected, &context.cancellation)
            .map_err(plugin_failure)
    };
    let source = resolve(source_value, "video", NativeHandleKind::Video, "VIDEO")?;
    let NativeStoredPayload::Video(video) = source.as_ref() else {
        return Err(invalid_value("video"));
    };
    if video.frame_count() > 240 {
        return Err(invalid_value("video frame count exceeds 240"));
    }
    let Some(Some(mask_values)) = inputs.values.get("alpha_mask") else {
        return Ok(());
    };
    let [mask_value] = mask_values.as_slice() else {
        return Err(invalid_value("alpha_mask"));
    };
    let mask = resolve(mask_value, "alpha_mask", NativeHandleKind::Mask, "MASK")?;
    let NativeStoredPayload::Tensor(mask) = mask.as_ref() else {
        return Err(invalid_value("alpha_mask"));
    };
    let bytes = context
        .prepare_provider_mask_video(
            mask.tensor(),
            video.frame_rate(),
            video.frame_count(),
            2_770_000,
        )
        .await?;
    let source_digest =
        hex_digest(&Sha256::digest(mask_value.abi_bytes().map_err(plugin_failure)?).into());
    let content_digest = hex_digest(&Sha256::digest(&bytes).into());
    inputs
        .register_media_batch_item(
            "alpha_mask",
            0,
            0,
            InvocationMediaContent::new("video/mp4", source_digest, content_digest, bytes),
        )
        .map_err(plugin_failure)
}

fn provider_structured_selector(node_id: &str) -> Option<&'static str> {
    match node_id {
        "ClaudeNode" => Some("model"),
        "BeebleSwitchXImageEdit" | "BeebleSwitchXVideoEdit" => Some("alpha_mode"),
        _ => None,
    }
}

fn prepare_provider_structured_inputs(
    values: &mut BTreeMap<String, NativeValue>,
    descriptor: Option<&NativeNodeDescriptor>,
    selector_name: &str,
) -> Result<(), NativeNodeFailure> {
    let Some(value) = values.get(selector_name) else {
        return Ok(());
    };
    let fields = if let Some(structured) =
        NativeStructuredValue::from_native_value(value).map_err(plugin_failure)?
    {
        if structured.type_name() != "COMFY_DYNAMICCOMBO_V3" {
            return Err(invalid_value(selector_name));
        }
        structured.fields().clone()
    } else if let NativeValue::PreservedUnknown {
        type_name,
        value: Value::Object(fields),
    } = value
    {
        if type_name != "COMFY_DYNAMICCOMBO_V3" {
            return Err(invalid_value(selector_name));
        }
        fields
            .iter()
            .map(|(name, value)| {
                scalar_to_native(scalar_from_json(value.clone())?, name)
                    .map(|value| (name.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?
    } else {
        return Ok(());
    };
    let Some(NativeValue::Primitive {
        value: NativePrimitive::String(selector),
    }) = fields.get(selector_name)
    else {
        return Err(invalid_value(selector_name));
    };
    let schema = descriptor
        .and_then(|descriptor| descriptor.source_schema.as_ref())
        .and_then(|schema| {
            schema
                .inputs
                .iter()
                .find(|input| input.name == selector_name)
        })
        .ok_or_else(|| invalid_value(selector_name))?;
    let options = schema.structured_options().map_err(plugin_failure)?;
    for field in options.iter().flat_map(|option| &option.fields) {
        if values.contains_key(&field.schema.name) {
            return Err(invalid_value(&field.schema.name));
        }
    }
    let option = options
        .into_iter()
        .find(|option| option.selector == *selector)
        .ok_or_else(|| invalid_value(selector_name))?;
    if fields.len() != option.fields.len() + 1 {
        return Err(invalid_value(selector_name));
    }
    let mut projected = BTreeMap::from([(
        selector_name.to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::String(selector.clone()),
        },
    )]);
    for field in option.fields {
        if field.path != [field.schema.name.clone()] || values.contains_key(&field.schema.name) {
            return Err(invalid_value(&field.schema.name));
        }
        let value = fields
            .get(&field.schema.name)
            .ok_or_else(|| invalid_value(&field.schema.name))?;
        if !sim_nodes::native_value_types_for_input_schema(&field.schema)
            .map_err(plugin_failure)?
            .accepts(value)
            || !sim_nodes::native_value_matches_input_schema(value, &field.schema)
        {
            return Err(invalid_value(&field.schema.name));
        }
        projected.insert(field.schema.name, value.clone());
    }
    values.extend(projected);
    Ok(())
}

fn prepare_claude_input_media(
    inputs: &mut InvocationInputs,
    values: &[(PluginValue, NativeOpaqueHandle)],
    context: &NativeNodeContext,
) -> Result<(), NativeNodeFailure> {
    let mut images = Vec::with_capacity(values.len());
    let mut total_images = 0_u64;
    for (value, handle) in values {
        let expected =
            NativeHandleType::new(NativeHandleKind::Image, "IMAGE").map_err(plugin_failure)?;
        let payload = context
            .handle_store()
            .resolve(handle, &expected, &context.cancellation)
            .map_err(plugin_failure)?;
        let NativeStoredPayload::Tensor(tensor) = payload.as_ref() else {
            return Err(invalid_value("images"));
        };
        let image = tensor.image().ok_or_else(|| invalid_value("images"))?;
        let (batch, _, _, _) = image.dimensions().map_err(plugin_failure)?;
        total_images = total_images
            .checked_add(batch)
            .filter(|count| *count <= 20)
            .ok_or_else(|| invalid_value("images"))?;
        let source_digest =
            hex_digest(&Sha256::digest(value.abi_bytes().map_err(plugin_failure)?).into());
        images.push((tensor.clone(), batch, source_digest));
    }
    for (value_index, (tensor, batch, source_digest)) in images.into_iter().enumerate() {
        let image = tensor.image().ok_or_else(|| invalid_value("images"))?;
        for batch_index in 0..batch {
            let bytes =
                context.prepare_provider_image_png(image, batch_index, Some(1_568 * 1_568))?;
            let content_digest = hex_digest(&Sha256::digest(&bytes).into());
            inputs
                .register_media_batch_item(
                    "images",
                    u32::try_from(value_index).map_err(plugin_failure)?,
                    u32::try_from(batch_index).map_err(plugin_failure)?,
                    InvocationMediaContent::new(
                        "image/png",
                        source_digest.clone(),
                        content_digest,
                        bytes,
                    ),
                )
                .map_err(plugin_failure)?;
        }
    }
    Ok(())
}

fn take_dynamic_values(
    values: &mut BTreeMap<String, NativeValue>,
    dynamic: &NativeDynamicInputDescriptor,
    source: Option<&sim_nodes::NativeDynamicSchemaMetadata>,
) -> Result<Vec<NativeValue>, NativeNodeFailure> {
    if dynamic.name_template == "{name}" && source.is_none() {
        return Err(invalid_value(&dynamic.input.name));
    }
    let mut matches = values
        .keys()
        .filter_map(|name| {
            dynamic_input_order(name, dynamic, source).map(|order| (order, name.clone()))
        })
        .collect::<Vec<_>>();
    matches.sort_by_key(|matched| matched.0);
    let count = u32::try_from(matches.len()).map_err(plugin_failure)?;
    if count < dynamic.minimum_count || count > dynamic.maximum_count {
        return Err(invalid_value(&dynamic.input.name));
    }
    matches
        .into_iter()
        .map(|(_, name)| {
            values
                .remove(&name)
                .ok_or_else(|| invalid_value(&dynamic.input.name))
        })
        .collect()
}

fn dynamic_input_order(
    name: &str,
    dynamic: &NativeDynamicInputDescriptor,
    source: Option<&sim_nodes::NativeDynamicSchemaMetadata>,
) -> Option<u32> {
    if dynamic.name_template == "{name}" {
        return source?
            .names
            .iter()
            .position(|candidate| candidate == name)
            .and_then(|index| u32::try_from(index).ok());
    }
    let (prefix, suffix) = dynamic.name_template.split_once("{index}")?;
    let number = name
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse::<u32>()
        .ok()?;
    let end = dynamic.start_index.checked_add(dynamic.maximum_count)?;
    (number >= dynamic.start_index && number < end).then_some(number)
}

fn remember_imported_handle(
    imported_handles: &mut ImportedHandles,
    key: (String, String),
    handle: NativeOpaqueHandle,
) {
    imported_handles
        .handles
        .entry(key)
        .and_modify(|imported| {
            if matches!(imported, ImportedHandle::Exact(existing) if existing != &handle) {
                *imported = ImportedHandle::Ambiguous;
            }
        })
        .or_insert(ImportedHandle::Exact(handle));
}

fn plugin_value(
    port: &PluginPort,
    value: NativeValue,
    registry: &TypeRegistry,
    profile_id: &str,
    context: &NativeNodeContext,
) -> Result<(PluginValue, Option<((String, String), NativeOpaqueHandle)>), NativeNodeFailure> {
    let family = registry.family(&port.type_id).map_err(plugin_failure)?;
    match family {
        ValueFamily::Scalar => Ok((
            PluginValue::scalar(
                port.type_id.clone(),
                scalar_from_native(value, &port.type_id.to_string())?,
                registry,
            )
            .map_err(plugin_failure)?,
            None,
        )),
        ValueFamily::Tensor | ValueFamily::Artifact | ValueFamily::Model => {
            let NativeValue::Handle { value: handle } = value else {
                return Err(invalid_value(&port.id));
            };
            let expected_type = native_handle_type(port).map_err(plugin_failure)?;
            let stored = context
                .handle_store()
                .resolve(&handle, &expected_type, &context.cancellation)
                .map_err(plugin_failure)?;
            let provider_semantic_digest = match stored.as_ref() {
                NativeStoredPayload::Provider(payload) => {
                    Some(payload.semantic_digest_sha256().to_owned())
                }
                _ => None,
            };
            let stored = plugin_value_from_stored(port, family, stored.as_ref(), registry)?;
            if stored.type_id() != &port.type_id || stored.family() != family {
                return Err(invalid_value(&port.id));
            }
            if let Some(expected_digest) = provider_semantic_digest {
                if value_digest(&stored) != Some(expected_digest.as_str()) {
                    return Err(invalid_value(&port.id));
                }
            } else {
                validate_value_digest(&handle, &stored, &port.id)?;
            }
            if let PluginValueRepresentation::Artifact(artifact) = stored.representation() {
                artifact_value_identity(profile_id, artifact).map_err(plugin_failure)?;
            }
            let digest = value_digest(&stored)
                .ok_or_else(|| invalid_value(&port.id))?
                .to_owned();
            Ok((stored, Some(((port.type_id.to_string(), digest), handle))))
        }
    }
}

fn plugin_value_from_stored(
    port: &PluginPort,
    family: ValueFamily,
    stored: &NativeStoredPayload,
    registry: &TypeRegistry,
) -> Result<PluginValue, NativeNodeFailure> {
    match stored {
        NativeStoredPayload::Provider(stored) => {
            let expected_type = native_handle_type(port).map_err(plugin_failure)?;
            if expected_type.kind != NativeHandleKind::ProviderTask
                || stored.handle_type() != &expected_type
            {
                return Err(invalid_value(&port.id));
            }
            let value = PluginValue::from_abi_bytes(stored.abi_bytes(), registry)
                .map_err(plugin_failure)?;
            if value.type_id() != &port.type_id || value.family() != family {
                return Err(invalid_value(&port.id));
            }
            Ok(value)
        }
        NativeStoredPayload::Tensor(stored) => {
            if family != ValueFamily::Tensor {
                return Err(invalid_value(&port.id));
            }
            let projection = stored.projection();
            let value = TensorValue::new(
                projection.descriptor().clone(),
                projection.resident_bytes(),
                projection.content_digest(),
            )
            .map_err(plugin_failure)?;
            PluginValue::tensor(port.type_id.clone(), value, registry).map_err(plugin_failure)
        }
        NativeStoredPayload::Model(stored) => {
            if family != ValueFamily::Model {
                return Err(invalid_value(&port.id));
            }
            let identity = stored.model_payload().identity();
            let value = ModelValue::new(
                identity.identifier(),
                identity.format(),
                stored.digest_sha256(),
            )
            .map_err(plugin_failure)?;
            PluginValue::model(port.type_id.clone(), value, registry).map_err(plugin_failure)
        }
        NativeStoredPayload::Control(stored) => {
            if family != ValueFamily::Model {
                return Err(invalid_value(&port.id));
            }
            let value = ModelValue::new(
                stored.digest_sha256(),
                "zed-native-control-v1",
                stored.digest_sha256(),
            )
            .map_err(plugin_failure)?;
            PluginValue::model(port.type_id.clone(), value, registry).map_err(plugin_failure)
        }
        NativeStoredPayload::Guider(stored) => {
            if family != ValueFamily::Model {
                return Err(invalid_value(&port.id));
            }
            let identity = stored.model().identity();
            let value = ModelValue::new(
                identity.identifier(),
                "zed-native-guider-v1",
                stored.semantic_digest_sha256(),
            )
            .map_err(plugin_failure)?;
            PluginValue::model(port.type_id.clone(), value, registry).map_err(plugin_failure)
        }
        NativeStoredPayload::Sampler(stored) => {
            if family != ValueFamily::Model {
                return Err(invalid_value(&port.id));
            }
            let value = ModelValue::new(
                stored.identity().as_str(),
                "zed-native-sampler-v1",
                stored.semantic_digest_sha256(),
            )
            .map_err(plugin_failure)?;
            PluginValue::model(port.type_id.clone(), value, registry).map_err(plugin_failure)
        }
        NativeStoredPayload::Audio(stored) => {
            if family != ValueFamily::Tensor {
                return Err(invalid_value(&port.id));
            }
            let value = TensorValue::new(
                stored.waveform().descriptor().clone(),
                stored.resident_bytes(),
                hex_digest(stored.semantic_digest_sha256()),
            )
            .map_err(plugin_failure)?;
            PluginValue::tensor(port.type_id.clone(), value, registry).map_err(plugin_failure)
        }
        NativeStoredPayload::Video(stored) => {
            if family != ValueFamily::Tensor {
                return Err(invalid_value(&port.id));
            }
            let components = stored.components().ok_or_else(|| invalid_value(&port.id))?;
            let value = TensorValue::new(
                components.frames().descriptor().clone(),
                stored.resident_bytes(),
                hex_digest(stored.semantic_digest_sha256()),
            )
            .map_err(plugin_failure)?;
            PluginValue::tensor(port.type_id.clone(), value, registry).map_err(plugin_failure)
        }
        NativeStoredPayload::Artifact(stored) => artifact_plugin_value(
            port,
            stored.source_type_id(),
            stored.bytes().len(),
            hex_digest(stored.semantic_digest_sha256()),
            registry,
        ),
        NativeStoredPayload::File3D(stored) => artifact_plugin_value(
            port,
            stored.source_type_id(),
            stored.bytes().len(),
            hex_digest(stored.semantic_digest_sha256()),
            registry,
        ),
        NativeStoredPayload::Camera(stored) => artifact_plugin_value(
            port,
            stored.source_type_id(),
            usize::try_from(stored.resident_bytes()).map_err(plugin_failure)?,
            hex_digest(stored.semantic_digest_sha256()),
            registry,
        ),
        NativeStoredPayload::Latent(_)
        | NativeStoredPayload::Conditioning(_)
        | NativeStoredPayload::Noise(_)
        | NativeStoredPayload::BoundingBox(_)
        | NativeStoredPayload::FaceLandmarks(_)
        | NativeStoredPayload::PoseKeypoint(_)
        | NativeStoredPayload::Sam3TrackData(_)
        | NativeStoredPayload::Tracks(_)
        | NativeStoredPayload::AudioEncoderOutput(_)
        | NativeStoredPayload::ClipVisionOutput(_)
        | NativeStoredPayload::IcLoraParameters(_)
        | NativeStoredPayload::LossMap(_)
        | NativeStoredPayload::Splat(_)
        | NativeStoredPayload::Mesh(_)
        | NativeStoredPayload::Voxel(_) => Err(unmaterialized_plugin_input(&port.id)),
    }
}

fn artifact_plugin_value(
    port: &PluginPort,
    source_type_id: &str,
    byte_length: usize,
    digest: String,
    registry: &TypeRegistry,
) -> Result<PluginValue, NativeNodeFailure> {
    let value = ArtifactValue::new(
        "native-provider-input",
        source_type_id,
        u64::try_from(byte_length).map_err(plugin_failure)?,
        digest,
    )
    .map_err(plugin_failure)?;
    PluginValue::artifact(port.type_id.clone(), value, registry).map_err(plugin_failure)
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn scalar_from_native(
    value: NativeValue,
    expected_unknown_type: &str,
) -> Result<ScalarValue, NativeNodeFailure> {
    match value {
        NativeValue::Primitive { value } => match value {
            NativePrimitive::Null => Ok(ScalarValue::Null),
            NativePrimitive::Boolean(value) => Ok(ScalarValue::Boolean(value)),
            NativePrimitive::Integer(value) => Ok(ScalarValue::Integer(value)),
            NativePrimitive::UnsignedInteger(value) => i64::try_from(value)
                .map(ScalarValue::Integer)
                .map_err(|_| invalid_value("unsigned-integer")),
            NativePrimitive::Number(value) => Ok(ScalarValue::Float(value)),
            NativePrimitive::String(value) => Ok(ScalarValue::String(value)),
        },
        NativeValue::PreservedUnknown { type_name, value }
            if type_name == expected_unknown_type =>
        {
            scalar_from_json(value)
        }
        _ => Err(invalid_value(expected_unknown_type)),
    }
}

fn scalar_from_json(value: Value) -> Result<ScalarValue, NativeNodeFailure> {
    match value {
        Value::Null => Ok(ScalarValue::Null),
        Value::Bool(value) => Ok(ScalarValue::Boolean(value)),
        Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                Ok(ScalarValue::Integer(integer))
            } else {
                value
                    .as_f64()
                    .map(ScalarValue::Float)
                    .ok_or_else(|| invalid_value("number"))
            }
        }
        Value::String(value) => Ok(ScalarValue::String(value)),
        Value::Array(values) => values
            .into_iter()
            .map(scalar_from_json)
            .collect::<Result<Vec<_>, _>>()
            .map(ScalarValue::List),
        Value::Object(values) => {
            let mut values = values.into_iter().collect::<Vec<_>>();
            values.sort_by(|left, right| left.0.cmp(&right.0));
            values
                .into_iter()
                .map(|(key, value)| Ok((key, scalar_from_json(value)?)))
                .collect::<Result<Vec<_>, NativeNodeFailure>>()
                .map(ScalarValue::Record)
        }
    }
}

fn provider_transport_request(
    node: &PluginNode,
    inputs: &InvocationInputs,
    imported_handles: &ImportedHandles,
    context: &NativeNodeContext,
) -> Result<ProviderTransportRequest, NativeNodeFailure> {
    let mut ports = Vec::with_capacity(inputs.values().len());
    for (port_id, values) in inputs.values() {
        let present = values.is_some();
        let values = values.as_deref().unwrap_or_default();
        let mut transport_values = Vec::with_capacity(values.len());
        for value in values {
            let transport_value = if value.family() == ValueFamily::Scalar
                || (node.id == "ClaudeNode" && port_id == "images")
                || BeeblePngInput::for_port(&node.id, port_id).is_some()
                || hitpaw_uses_first_image(&node.id, port_id)
                || bfl_ultra_uses_first_image(&node.id, port_id)
                || (node.id == "BeebleSwitchXVideoEdit" && port_id == "alpha_mask")
            {
                // Encoded provider inputs consume admitted PNG attachments; the request
                // retains tensor identity without duplicating the source pixels.
                ProviderTransportValue::checked(
                    value.type_id().to_string(),
                    value.family(),
                    value.abi_bytes().map_err(plugin_failure)?,
                )
                .map_err(plugin_failure)?
            } else {
                let digest =
                    value_digest(value).ok_or_else(|| unmaterialized_plugin_input(port_id))?;
                let key = (value.type_id().to_string(), digest.to_owned());
                let handle = match imported_handles.handles.get(&key) {
                    Some(ImportedHandle::Exact(handle)) => handle,
                    _ => return Err(unmaterialized_plugin_input(port_id)),
                };
                let expected_type = native_plugin_source_type_projection(value.type_id().name())
                    .map_err(plugin_failure)?
                    .handle_type()
                    .map_err(plugin_failure)?
                    .ok_or_else(|| unmaterialized_plugin_input(port_id))?;
                let payload = context
                    .handle_store()
                    .resolve(handle, &expected_type, &context.cancellation)
                    .map_err(plugin_failure)?;
                ProviderTransportValue::from_native_payload(
                    value.type_id().to_string(),
                    value.family(),
                    payload.as_ref(),
                )
                .map_err(plugin_failure)?
            };
            transport_values.push(transport_value);
        }
        ports.push(
            ProviderTransportPort::checked(port_id, present, transport_values)
                .map_err(plugin_failure)?,
        );
    }
    ProviderTransportRequest::checked(&node.id, ports).map_err(plugin_failure)
}

fn provider_transport_outputs(
    node: &PluginNode,
    response: &ProviderTransportResponse,
    registry: &TypeRegistry,
    signed_namespace: &str,
    context: &NativeNodeContext,
) -> Result<Vec<NativeValue>, NativeNodeFailure> {
    if response.class_type() != node.id {
        return Err(unmaterialized_plugin_output("provider class"));
    }
    let expected_ports = node
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
        .collect::<Vec<_>>();
    if response.ports().len() != expected_ports.len() {
        return Err(unmaterialized_plugin_output("provider output count"));
    }
    let mut values_by_port = BTreeMap::new();
    let mut pending_payloads = Vec::new();
    for transport_port in response.ports() {
        let port = expected_ports
            .iter()
            .copied()
            .find(|port| port.id == transport_port.port_id())
            .ok_or_else(|| unmaterialized_plugin_output("provider output port"))?;
        validate_provider_output_cardinality(
            port,
            transport_port.present(),
            transport_port.values().len(),
        )?;
        let mut port_values = Vec::with_capacity(transport_port.values().len());
        for transport_value in transport_port.values() {
            if transport_value.type_id() != port.type_id.to_string()
                || transport_value.family()
                    != registry.family(&port.type_id).map_err(plugin_failure)?
            {
                return Err(unmaterialized_plugin_output("provider output metadata"));
            }
            match transport_value.encoding() {
                ProviderTransportEncoding::PluginValue => {
                    if transport_value.family() != ValueFamily::Scalar {
                        return Err(unmaterialized_plugin_output("provider output encoding"));
                    }
                    let value = PluginValue::from_abi_bytes(transport_value.abi_bytes(), registry)
                        .map_err(plugin_failure)?;
                    require_exact_port_type(port, &value).map_err(plugin_failure)?;
                    port_values.push(scalar_native_value(value, &port.type_id.to_string())?);
                }
                ProviderTransportEncoding::NativePayload => {
                    if transport_value.family() == ValueFamily::Scalar {
                        return Err(unmaterialized_plugin_output("provider output encoding"));
                    }
                    let expected_type = native_handle_type(port).map_err(plugin_failure)?;
                    let payload = transport_value
                        .materialize_native_payload(&expected_type, signed_namespace, context)
                        .map_err(plugin_failure)?;
                    let pending_index = pending_payloads.len();
                    pending_payloads.push(payload);
                    port_values.push(PendingProviderValue::Handle(pending_index));
                }
            }
        }
        values_by_port.insert(port.id.clone(), (transport_port.present(), port_values));
    }
    if expected_ports
        .iter()
        .any(|port| !values_by_port.contains_key(&port.id))
    {
        return Err(unmaterialized_plugin_output("provider output port"));
    }
    let handles = publish_provider_payloads(context, pending_payloads)?;
    let mut output_values = Vec::with_capacity(expected_ports.len());
    for port in expected_ports {
        let (present, values) = values_by_port
            .remove(&port.id)
            .ok_or_else(|| unmaterialized_plugin_output("provider output port"))?;
        if !present {
            output_values.push(NativeValue::Primitive {
                value: NativePrimitive::Null,
            });
            continue;
        }
        let values = values
            .into_iter()
            .map(|value| match value {
                PendingProviderValue::Native(value) => Ok(value),
                PendingProviderValue::Handle(index) => handles
                    .get(index)
                    .cloned()
                    .map(|value| NativeValue::Handle { value })
                    .ok_or_else(|| unmaterialized_plugin_output("provider output handle")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if port.cardinality == PortCardinality::List {
            output_values.push(NativeValue::List { values });
        } else {
            output_values.push(
                values
                    .into_iter()
                    .next()
                    .ok_or_else(|| unmaterialized_plugin_output("provider scalar output"))?,
            );
        }
    }
    Ok(output_values)
}

pub struct PreparedNativeProviderInvocation {
    pub inputs: InvocationInputs,
    pub request: ProviderTransportRequest,
}

pub async fn prepare_native_provider_invocation(
    node: &PluginNode,
    descriptor: &NativeNodeDescriptor,
    inputs: BTreeMap<String, NativeValue>,
    profile_id: &str,
    context: &NativeNodeContext,
) -> Result<PreparedNativeProviderInvocation, NativeNodeFailure> {
    let registry = TypeRegistry::built_in().map_err(plugin_failure)?;
    let (mut inputs, imported_handles) = invocation_inputs(
        node,
        inputs,
        &registry,
        profile_id,
        context,
        Some(descriptor),
    )?;
    prepare_beeble_video_media(node, &mut inputs, &imported_handles, context).await?;
    let request = provider_transport_request(node, &inputs, &imported_handles, context)?;
    Ok(PreparedNativeProviderInvocation { inputs, request })
}

pub fn materialize_native_provider_response(
    node: &PluginNode,
    response: &ProviderTransportResponse,
    signed_namespace: &str,
    context: &NativeNodeContext,
) -> Result<Vec<NativeValue>, NativeNodeFailure> {
    let registry = TypeRegistry::built_in().map_err(plugin_failure)?;
    provider_transport_outputs(node, response, &registry, signed_namespace, context)
}

enum PendingProviderValue {
    Native(NativeValue),
    Handle(usize),
}

fn scalar_native_value(
    value: PluginValue,
    type_name: &str,
) -> Result<PendingProviderValue, NativeNodeFailure> {
    let PluginValueRepresentation::Scalar(value) = value.representation() else {
        return Err(unmaterialized_plugin_output("provider scalar output"));
    };
    scalar_to_native(value.clone(), type_name).map(PendingProviderValue::Native)
}

fn publish_provider_payloads(
    context: &NativeNodeContext,
    payloads: Vec<NativeStoredPayload>,
) -> Result<Vec<NativeOpaqueHandle>, NativeNodeFailure> {
    let mut handles = Vec::with_capacity(payloads.len());
    for payload in payloads {
        if let Err(error) = context.cancellation.check() {
            rollback_provider_handles(context, &handles)?;
            return Err(plugin_failure(error));
        }
        match context
            .handle_store()
            .publish(payload, &context.cancellation)
        {
            Ok(handle) => handles.push(handle),
            Err(error) => {
                rollback_provider_handles(context, &handles)?;
                return Err(plugin_failure(error));
            }
        }
    }
    if let Err(error) = context.cancellation.check() {
        rollback_provider_handles(context, &handles)?;
        return Err(plugin_failure(error));
    }
    Ok(handles)
}

fn rollback_provider_handles(
    context: &NativeNodeContext,
    handles: &[NativeOpaqueHandle],
) -> Result<(), NativeNodeFailure> {
    let cleanup = CancellationToken::default();
    for handle in handles.iter().rev() {
        context
            .handle_store()
            .revoke(handle, &cleanup)
            .map_err(plugin_failure)?;
    }
    Ok(())
}

pub fn rollback_native_provider_outputs(
    context: &NativeNodeContext,
    values: &[NativeValue],
) -> Result<(), NativeNodeFailure> {
    let mut handles = Vec::new();
    for value in values {
        collect_provider_output_handles(value, &mut handles);
    }
    rollback_provider_handles(context, &handles)
}

fn collect_provider_output_handles(value: &NativeValue, handles: &mut Vec<NativeOpaqueHandle>) {
    match value {
        NativeValue::Handle { value } => handles.push(value.clone()),
        NativeValue::List { values } => {
            for value in values {
                collect_provider_output_handles(value, handles);
            }
        }
        NativeValue::Primitive { .. } | NativeValue::PreservedUnknown { .. } => {}
    }
}

fn validate_provider_output_cardinality(
    port: &PluginPort,
    present: bool,
    value_count: usize,
) -> Result<(), NativeNodeFailure> {
    if (!present && value_count != 0)
        || (port.presence == PortPresence::Required && !present)
        || (port.cardinality == PortCardinality::Singular
            && ((present && value_count != 1) || value_count > 1))
    {
        return Err(unmaterialized_plugin_output(&port.id));
    }
    Ok(())
}

fn invocation_outputs(
    node: &PluginNode,
    outputs: &BTreeMap<String, Vec<PluginValue>>,
    presence: &BTreeMap<String, bool>,
    registry: &TypeRegistry,
    profile_id: &str,
    context: &NativeNodeContext,
    imported_handles: &ImportedHandles,
) -> Result<Vec<NativeValue>, NativeNodeFailure> {
    let result = node
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
        .map(|port| {
            if !presence.get(&port.id).copied().unwrap_or(false) {
                return Ok(NativeValue::Primitive {
                    value: NativePrimitive::Null,
                });
            }
            let values = outputs.get(&port.id).cloned().unwrap_or_default();
            if port.cardinality == PortCardinality::List {
                values
                    .into_iter()
                    .map(|value| runtime_value(value, port, registry, profile_id, imported_handles))
                    .collect::<Result<Vec<_>, _>>()
                    .map(|values| NativeValue::List { values })
            } else {
                let mut values = values.into_iter();
                let value = values.next().ok_or_else(|| invalid_value(&port.id))?;
                if values.next().is_some() {
                    return Err(invalid_value(&port.id));
                }
                runtime_value(value, port, registry, profile_id, imported_handles)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if context.cancellation.is_cancelled() {
        return Err(plugin_failure("plugin output projection was cancelled"));
    }
    Ok(result)
}

fn runtime_value(
    value: PluginValue,
    port: &PluginPort,
    registry: &TypeRegistry,
    profile_id: &str,
    imported_handles: &ImportedHandles,
) -> Result<NativeValue, NativeNodeFailure> {
    if value.type_id() != &port.type_id {
        return Err(invalid_value(&port.id));
    }
    match value.representation() {
        PluginValueRepresentation::Scalar(value) => {
            scalar_to_native(value.clone(), &port.type_id.to_string())
        }
        PluginValueRepresentation::Artifact(artifact) => {
            artifact_value_identity(profile_id, artifact).map_err(plugin_failure)?;
            imported_runtime_value(value, port, registry, imported_handles)
        }
        PluginValueRepresentation::Tensor(_) | PluginValueRepresentation::Model(_) => {
            imported_runtime_value(value, port, registry, imported_handles)
        }
    }
}

fn imported_runtime_value(
    value: PluginValue,
    port: &PluginPort,
    registry: &TypeRegistry,
    imported_handles: &ImportedHandles,
) -> Result<NativeValue, NativeNodeFailure> {
    let family = registry.family(&port.type_id).map_err(plugin_failure)?;
    if family == ValueFamily::Scalar || value.family() != family {
        return Err(invalid_value(&port.id));
    }
    let digest = value_digest(&value).ok_or_else(|| invalid_value(&port.id))?;
    let key = (port.type_id.to_string(), digest.to_owned());
    let ImportedHandle::Exact(handle) = imported_handles
        .handles
        .get(&key)
        .ok_or_else(|| unmaterialized_plugin_output(&port.id))?
    else {
        return Err(unmaterialized_plugin_output(&port.id));
    };
    let expected = native_handle_type(port).map_err(plugin_failure)?;
    if handle.handle_type() != &expected
        || handle.store_identity() != imported_handles.store_identity
        || (expected.kind != NativeHandleKind::ProviderTask
            && handle.digest_sha256() != Some(digest))
    {
        return Err(invalid_value(&port.id));
    }
    Ok(NativeValue::Handle {
        value: handle.clone(),
    })
}

fn validate_value_digest(
    handle: &sim_nodes::NativeOpaqueHandle,
    value: &PluginValue,
    port: &str,
) -> Result<(), NativeNodeFailure> {
    if value_digest(value) != handle.digest_sha256() {
        return Err(invalid_value(port));
    }
    Ok(())
}

fn value_digest(value: &PluginValue) -> Option<&str> {
    match value.representation() {
        PluginValueRepresentation::Tensor(value) => Some(value.digest()),
        PluginValueRepresentation::Artifact(value) => Some(value.digest()),
        PluginValueRepresentation::Model(value) => Some(value.digest()),
        PluginValueRepresentation::Scalar(_) => None,
    }
}

fn scalar_to_native(value: ScalarValue, type_name: &str) -> Result<NativeValue, NativeNodeFailure> {
    let value = match value {
        ScalarValue::Null => {
            return Ok(NativeValue::Primitive {
                value: NativePrimitive::Null,
            });
        }
        ScalarValue::Boolean(value) => {
            return Ok(NativeValue::Primitive {
                value: NativePrimitive::Boolean(value),
            });
        }
        ScalarValue::Integer(value) => {
            return Ok(NativeValue::Primitive {
                value: NativePrimitive::Integer(value),
            });
        }
        ScalarValue::Float(value) => {
            return Ok(NativeValue::Primitive {
                value: NativePrimitive::Number(value),
            });
        }
        ScalarValue::String(value) => {
            return Ok(NativeValue::Primitive {
                value: NativePrimitive::String(value),
            });
        }
        ScalarValue::Bytes(values) => Value::Array(
            values
                .into_iter()
                .map(|value| Value::Number(Number::from(value)))
                .collect(),
        ),
        ScalarValue::List(values) => Value::Array(
            values
                .into_iter()
                .map(scalar_to_json)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ScalarValue::Record(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| Ok((key, scalar_to_json(value)?)))
                .collect::<Result<Map<_, _>, NativeNodeFailure>>()?,
        ),
    };
    Ok(NativeValue::PreservedUnknown {
        type_name: type_name.to_owned(),
        value,
    })
}

fn scalar_to_json(value: ScalarValue) -> Result<Value, NativeNodeFailure> {
    match value {
        ScalarValue::Null => Ok(Value::Null),
        ScalarValue::Boolean(value) => Ok(Value::Bool(value)),
        ScalarValue::Integer(value) => Ok(Value::Number(Number::from(value))),
        ScalarValue::Float(value) => Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| invalid_value("float")),
        ScalarValue::String(value) => Ok(Value::String(value)),
        ScalarValue::Bytes(values) => Ok(Value::Array(
            values
                .into_iter()
                .map(|value| Value::Number(Number::from(value)))
                .collect(),
        )),
        ScalarValue::List(values) => values
            .into_iter()
            .map(scalar_to_json)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        ScalarValue::Record(values) => values
            .into_iter()
            .map(|(key, value)| Ok((key, scalar_to_json(value)?)))
            .collect::<Result<Map<_, _>, NativeNodeFailure>>()
            .map(Value::Object),
    }
}

fn invalid_value(port: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_plugin_value".to_owned(),
        message: format!("plugin port `{port}` has an invalid native value"),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn unmaterialized_plugin_effect(kind: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "unmaterialized_plugin_effect".to_owned(),
        message: format!("plugin {kind} effects are not materializable at the native boundary"),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn unmaterialized_plugin_output(port: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "unmaterialized_plugin_payload".to_owned(),
        message: format!(
            "plugin port `{port}` returned non-scalar metadata without a host-materialized payload"
        ),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn unmaterialized_plugin_input(port: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "unmaterialized_plugin_payload".to_owned(),
        message: format!(
            "plugin port `{port}` has no lossless representation in the plugin SDK value families"
        ),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn plugin_failure(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "plugin_invocation_failed".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_model::ClipVisionOutput;
    use sim_nodes::NativePrimitiveType;
    use sim_plugin_sdk::{
        ApiVersion, ArtifactValue, DType, DeviceId, PortSerialization, StreamId, TensorDescriptor,
    };
    use sim_runtime::{AttemptId, NativeHandleStoreGeneration, PromptId};
    use sim_tensor::{
        CancellationToken, CpuBackend, CpuWorkspaceAuthority, NativeLatentBundle,
        NativeLatentMetadata, Tensor,
    };
    use sim_types::NodeId;
    use std::error::Error;

    fn artifact_value(identifier: &str) -> Result<PluginValue, Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        Ok(PluginValue::artifact(
            registry.resolve("SVG")?.clone(),
            ArtifactValue::new("input", identifier, 3, "2".repeat(64))?,
            &registry,
        )?)
    }

    #[test]
    fn beeble_structured_masks_preserve_native_identity_and_reject_ambiguity()
    -> Result<(), Box<dyn Error>> {
        let registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let mask = NativeValue::Handle {
            value: opaque_handle(
                NativeHandleType::new(NativeHandleKind::Mask, "MASK")?,
                "alpha-mask",
                1,
                &"a".repeat(64),
            )?,
        };
        for node_id in ["BeebleSwitchXImageEdit", "BeebleSwitchXVideoEdit"] {
            let descriptor = registry
                .descriptor(node_id)
                .ok_or("missing Beeble descriptor")?;
            for (selector, field) in [
                ("auto", None),
                ("fill", None),
                ("select", Some("alpha_keyframe")),
                ("custom", Some("alpha_mask")),
            ] {
                let mut fields = BTreeMap::from([(
                    "alpha_mode".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::String(selector.to_owned()),
                    },
                )]);
                if let Some(field) = field {
                    fields.insert(field.to_owned(), mask.clone());
                }
                let structured = NativeStructuredValue::checked("COMFY_DYNAMICCOMBO_V3", fields)?
                    .into_runtime_value()?;
                let mut values = BTreeMap::from([("alpha_mode".to_owned(), structured)]);
                let original = values.clone();
                for field in ["alpha_keyframe", "alpha_mask"] {
                    let mut ambiguous = values.clone();
                    ambiguous.insert(field.to_owned(), mask.clone());
                    let before = ambiguous.clone();
                    assert!(
                        prepare_provider_structured_inputs(
                            &mut ambiguous,
                            Some(descriptor),
                            "alpha_mode"
                        )
                        .is_err()
                    );
                    assert_eq!(ambiguous, before);
                }
                prepare_provider_structured_inputs(&mut values, Some(descriptor), "alpha_mode")?;
                assert_eq!(
                    values.get("alpha_mode"),
                    Some(&NativeValue::Primitive {
                        value: NativePrimitive::String(selector.to_owned()),
                    })
                );
                if let Some(field) = field {
                    assert_eq!(values.get(field), Some(&mask));
                    let mut ambiguous = original.clone();
                    ambiguous.insert(field.to_owned(), mask.clone());
                    let before = ambiguous.clone();
                    assert!(
                        prepare_provider_structured_inputs(
                            &mut ambiguous,
                            Some(descriptor),
                            "alpha_mode"
                        )
                        .is_err()
                    );
                    assert_eq!(ambiguous, before);
                    let mut wrong_type = BTreeMap::from([(
                        "alpha_mode".to_owned(),
                        NativeValue::PreservedUnknown {
                            type_name: "COMFY_DYNAMICCOMBO_V3".to_owned(),
                            value: serde_json::json!({"alpha_mode": selector, field: {"kind":"handle", "identifier":"alpha-mask"}}),
                        },
                    )]);
                    let before = wrong_type.clone();
                    assert!(
                        prepare_provider_structured_inputs(
                            &mut wrong_type,
                            Some(descriptor),
                            "alpha_mode"
                        )
                        .is_err()
                    );
                    assert_eq!(wrong_type, before);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn beeble_mask_import_resolves_the_original_store_and_rejects_stale_handles()
    -> Result<(), Box<dyn Error>> {
        use sim_nodes::{NativeNodeComputeSession, NativeNodeServiceIdentity, NativeNodeServices};
        use sim_tensor::{NativeTensorPayload, NativeTensorRole};
        let registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let descriptor = registry
            .descriptor("BeebleSwitchXImageEdit")
            .ok_or("missing descriptor")?;
        let types = TypeRegistry::built_in()?;
        let mut selector_port =
            input_port(&types, "COMFY_DYNAMICCOMBO_V3", PortSerialization::Inline)?;
        selector_port.id = "alpha_mode".to_owned();
        selector_port.name = selector_port.id.clone();
        let mut mask_port = input_port(&types, "MASK", PortSerialization::Handle)?;
        mask_port.id = "alpha_mask".to_owned();
        mask_port.name = mask_port.id.clone();
        mask_port.presence = PortPresence::Optional;
        let node = PluginNode {
            id: "BeebleSwitchXImageEdit".to_owned(),
            version: ApiVersion {
                major: 1,
                minor: 0,
                patch: 0,
            },
            display_name: "Beeble".to_owned(),
            category: "test/provider".to_owned(),
            ports: vec![selector_port, mask_port],
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        };
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let backend = Arc::new(backend);
        let cancellation = CancellationToken::default();
        let tensor = {
            let execution = backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(16)?,
                &cancellation,
            );
            backend
                .upload_f32(
                    TensorDescriptor::contiguous(
                        vec![1, 2, 2],
                        DType::F32,
                        DeviceId::CPU,
                        StreamId::DEFAULT,
                    )?,
                    &[0.0, 0.25, 0.75, 1.0],
                    &execution,
                )?
                .0
        };
        let payload = NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_tensor(
            NativeTensorRole::Mask,
            tensor,
        )?));
        let generation = NativeHandleStoreGeneration::with_capacities(2, 1024)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x43620));
        let store = generation.handle_store_for_attempt(attempt_id);
        let mask = store.publish(payload, &cancellation)?;
        let scratch = authority.authorize_workspace(512)?;
        let node_id = NodeId::from("beeble-mask");
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x43622),
                attempt_id,
                node_id.clone(),
            )?,
            backend,
            StreamId::DEFAULT,
            &scratch,
        )?;
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x43621)),
            attempt_id,
            node_id,
            cancellation.clone(),
            scratch,
            store,
            NativeNodeServices::checked(None, None, Some(compute))?,
        )?;
        let values =
            |mask: NativeOpaqueHandle| -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
                Ok(BTreeMap::from([(
                    "alpha_mode".to_owned(),
                    NativeStructuredValue::checked(
                        "COMFY_DYNAMICCOMBO_V3",
                        BTreeMap::from([
                            (
                                "alpha_mode".to_owned(),
                                NativeValue::Primitive {
                                    value: NativePrimitive::String("custom".to_owned()),
                                },
                            ),
                            ("alpha_mask".to_owned(), NativeValue::Handle { value: mask }),
                        ]),
                    )?
                    .into_runtime_value()?,
                )]))
            };
        let (inputs, imported) = invocation_inputs(
            &node,
            values(mask.clone())?,
            &types,
            "profile-a",
            &context,
            Some(descriptor),
        )?;
        assert_eq!(imported.handles.len(), 1);
        assert!(
            imported
                .handles
                .values()
                .any(|value| matches!(value, ImportedHandle::Exact(handle) if handle == &mask))
        );
        let value = inputs
            .values
            .get("alpha_mask")
            .and_then(Option::as_ref)
            .and_then(|values| values.first())
            .ok_or("missing imported mask")?;
        assert_eq!(value.type_id(), types.resolve("MASK")?);
        let media = inputs
            .media_batches
            .get("alpha_mask")
            .and_then(|values| values.get(&0))
            .and_then(|frames| frames.get(&0))
            .ok_or("missing encoded alpha mask")?;
        assert!(media.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(
            media.bytes.get(16..24),
            Some([0, 0, 0, 2, 0, 0, 0, 2].as_slice())
        );
        assert_eq!(
            media.content_sha256,
            hex_digest(&Sha256::digest(&media.bytes).into())
        );
        assert_eq!(
            media.source_value_sha256,
            hex_digest(&Sha256::digest(value.abi_bytes()?).into())
        );
        let mut video_node = node.clone();
        video_node.id = "BeebleSwitchXVideoEdit".to_owned();
        let (video_inputs, _) = invocation_inputs(
            &video_node,
            values(mask.clone())?,
            &types,
            "profile-a",
            &context,
            registry.descriptor("BeebleSwitchXVideoEdit"),
        )?;
        assert!(
            video_inputs.media_batches.is_empty(),
            "custom video alpha must not become a PNG"
        );
        assert_eq!(generation.len(), 1);
        let resident_bytes = generation.resident_bytes();
        let stale = opaque_handle(
            NativeHandleType::new(NativeHandleKind::Mask, "MASK")?,
            "stale-mask",
            1,
            &"a".repeat(64),
        )?;
        assert!(
            invocation_inputs(
                &node,
                values(stale)?,
                &types,
                "profile-a",
                &context,
                Some(descriptor)
            )
            .is_err()
        );
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), resident_bytes);
        cancellation.cancel();
        let failure = invocation_inputs(
            &node,
            values(mask)?,
            &types,
            "profile-a",
            &context,
            Some(descriptor),
        )
        .err()
        .ok_or("cancelled mask import succeeded")?;
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(generation.len(), 1);
        Ok(())
    }

    #[test]
    fn beeble_image_attachments_preserve_source_and_reference_preparation()
    -> Result<(), Box<dyn Error>> {
        use sim_nodes::{NativeNodeComputeSession, NativeNodeServiceIdentity, NativeNodeServices};
        use sim_tensor::{ImageTensor, NativeTensorPayload, NativeTensorRole};
        let registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let descriptor = registry
            .descriptor("BeebleSwitchXImageEdit")
            .ok_or("missing descriptor")?;
        let types = TypeRegistry::built_in()?;
        let mut ports = Vec::new();
        for name in ["image", "reference_image"] {
            let mut port = input_port(&types, "IMAGE", PortSerialization::Handle)?;
            port.id = name.to_owned();
            port.name = name.to_owned();
            ports.push(port);
        }
        let node = PluginNode {
            id: "BeebleSwitchXImageEdit".to_owned(),
            version: ApiVersion {
                major: 1,
                minor: 0,
                patch: 0,
            },
            display_name: "Beeble".to_owned(),
            category: "test/provider".to_owned(),
            ports,
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        };
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let backend = Arc::new(backend);
        let cancellation = CancellationToken::default();
        let scratch = authority.authorize_workspace(1024 * 1024)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x43630));
        let node_id = NodeId::from("beeble-images");
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x43631),
                attempt_id,
                node_id.clone(),
            )?,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let generation = NativeHandleStoreGeneration::with_capacities(2, 1024 * 1024)?;
        let store = generation.handle_store_for_attempt(attempt_id);
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x43632)),
            attempt_id,
            node_id,
            cancellation.clone(),
            scratch,
            store.clone(),
            NativeNodeServices::checked(None, None, Some(compute))?,
        )?;
        let execution = context.compute_session()?.execution_context(&context)?;
        let mut pixels = vec![0.25; 4096 * 3];
        pixels.extend(vec![0.75; 4096 * 3]);
        let image = ImageTensor::from_f32(&backend, &execution, 2, 1, 4096, 3, &pixels)?;
        let handle = store.publish(
            NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                NativeTensorRole::Image,
                image,
            )?)),
            &cancellation,
        )?;
        let resident = generation.resident_bytes();
        let values = BTreeMap::from([
            (
                "image".to_owned(),
                NativeValue::Handle {
                    value: handle.clone(),
                },
            ),
            (
                "reference_image".to_owned(),
                NativeValue::Handle { value: handle },
            ),
        ]);
        let prepared = futures::executor::block_on(prepare_native_provider_invocation(
            &node,
            descriptor,
            values,
            "profile-a",
            &context,
        ))?;
        assert!(prepared.request.to_bytes()?.len() < 4096);
        for (port_id, width, height, color_type) in [
            ("image", 4096_u32, 1_u32, 2_u8),
            ("reference_image", 3, 4096, 0),
        ] {
            let frames = prepared
                .inputs
                .media_batches
                .get(port_id)
                .and_then(|values| values.get(&0))
                .ok_or("missing image attachment")?;
            assert_eq!(frames.len(), 1);
            let media = frames.get(&0).ok_or("missing first image frame")?;
            assert_eq!(
                media.bytes.get(16..20),
                Some(width.to_be_bytes().as_slice())
            );
            assert_eq!(
                media.bytes.get(20..24),
                Some(height.to_be_bytes().as_slice())
            );
            assert_eq!(media.bytes.get(25), Some(&color_type));
            assert_eq!(
                media.content_sha256,
                hex_digest(&Sha256::digest(&media.bytes).into())
            );
            let value = prepared
                .inputs
                .values
                .get(port_id)
                .and_then(Option::as_ref)
                .and_then(|values| values.first())
                .ok_or("missing source value")?;
            assert_eq!(
                media.source_value_sha256,
                hex_digest(&Sha256::digest(value.abi_bytes()?).into())
            );
        }
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), resident);
        Ok(())
    }

    #[test]
    fn beeble_component_ports_include_conditionally_present_mask_handles()
    -> Result<(), Box<dyn Error>> {
        let types = TypeRegistry::built_in()?;
        let registry = sim_runtime::generated_native_node_registry_projection(None)?;
        for (node_id, feature_id, source_name, source_type, alpha_output_type) in [
            (
                "BeebleSwitchXImageEdit",
                "COMFY-NODE-0020",
                "image",
                "IMAGE",
                "MASK",
            ),
            (
                "BeebleSwitchXVideoEdit",
                "COMFY-NODE-0021",
                "video",
                "VIDEO",
                "VIDEO",
            ),
        ] {
            let mut ports = Vec::new();
            for (name, source_type, presence, serialization) in [
                (
                    source_name,
                    source_type,
                    PortPresence::Required,
                    PortSerialization::Handle,
                ),
                (
                    "prompt",
                    "STRING",
                    PortPresence::Required,
                    PortSerialization::Inline,
                ),
                (
                    "reference_image",
                    "IMAGE",
                    PortPresence::Optional,
                    PortSerialization::Handle,
                ),
                (
                    "alpha_mode",
                    "COMFY_DYNAMICCOMBO_V3",
                    PortPresence::Required,
                    PortSerialization::Inline,
                ),
                (
                    "max_resolution",
                    "STRING",
                    PortPresence::Required,
                    PortSerialization::Inline,
                ),
                (
                    "seed",
                    "INT",
                    PortPresence::Required,
                    PortSerialization::Inline,
                ),
                (
                    "alpha_keyframe",
                    "MASK",
                    PortPresence::Optional,
                    PortSerialization::Handle,
                ),
                (
                    "alpha_mask",
                    "MASK",
                    PortPresence::Optional,
                    PortSerialization::Handle,
                ),
            ] {
                let mut port = input_port(&types, source_type, serialization)?;
                port.id = name.to_owned();
                port.name = name.to_owned();
                port.presence = presence;
                ports.push(port);
            }
            for (index, source_type) in [source_type, alpha_output_type].into_iter().enumerate() {
                let mut port = input_port(&types, source_type, PortSerialization::Handle)?;
                port.id = format!("output_{index}");
                port.name = port.id.clone();
                port.direction = PortDirection::Output;
                ports.push(port);
            }
            let presentation = registry
                .presentation(node_id)
                .ok_or("missing Beeble presentation")?;
            let mut node = PluginNode {
                id: node_id.to_owned(),
                version: ApiVersion {
                    major: 1,
                    minor: 0,
                    patch: 0,
                },
                display_name: presentation.display_name.clone(),
                category: presentation.category.clone(),
                ports,
                determinism: DeterminismPolicy::External,
                cache: CachePolicy::Never,
                effects: EffectPolicy::Provider,
            };
            let claim = ProviderBindingClaim {
                feature_id: feature_id.to_owned(),
                node_id: node_id.to_owned(),
                contract_sha256: registry
                    .provider_binding_contract_sha256(
                        node_id,
                        sim_runtime::NATIVE_PROVIDER_TRANSPORT_SCHEMA,
                        "zed:sim-provider-materializer@1",
                    )?
                    .ok_or("missing Beeble provider contract")?,
                transport_schema: sim_runtime::NATIVE_PROVIDER_TRANSPORT_SCHEMA.parse()?,
                materializer_schema: "zed:sim-provider-materializer@1".parse()?,
            };
            validate_provider_node_projection(&registry, &node, &claim)?;
            let port = node
                .ports
                .iter_mut()
                .find(|port| port.id == "alpha_mask")
                .ok_or("missing alpha mask")?;
            port.presence = PortPresence::Required;
            assert!(validate_provider_node_projection(&registry, &node, &claim).is_err());
        }
        Ok(())
    }

    #[test]
    fn claude_prompt_compilation_preserves_model_fields_and_named_images()
    -> Result<(), Box<dyn Error>> {
        let generated = sim_runtime::generated_native_node_registry_projection(None)?;
        let mut registry = NativeNodeRegistry::default();
        for class_type in ["ClaudeNode", "LoadImage", "PreviewImage"] {
            registry.register_descriptor(
                generated
                    .descriptor(class_type)
                    .ok_or("missing source descriptor")?
                    .clone(),
            )?;
        }
        let compiler = sim_runtime::PromptCompiler::new(&registry).with_provider_registry_pin(
            sim_runtime::NativeProviderRegistryPin::checked(
                1,
                "a".repeat(64),
                vec!["b".repeat(64)],
            )?,
        )?;
        for model in [
            "Opus 4.7",
            "Opus 4.6",
            "Sonnet 4.6",
            "Sonnet 4.5",
            "Haiku 4.5",
        ] {
            let mut inputs = serde_json::json!({
                "prompt": "hello", "model": model, "seed": 1,
                "model.max_tokens": 32768, "model.temperature": 1.0,
                "system_prompt": "Be concise", "image_2": ["image", 0], "image_10": ["image", 0]
            });
            if model != "Haiku 4.5" {
                inputs
                    .as_object_mut()
                    .ok_or("inputs are not an object")?
                    .insert(
                        "model.reasoning_effort".to_owned(),
                        Value::String("high".to_owned()),
                    );
            }
            let submission =
                |inputs: Value| -> Result<sim_types::PromptSubmission, serde_json::Error> {
                    serde_json::from_value(serde_json::json!({"prompt": {
                    "image": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
                        "claude": {"class_type": "ClaudeNode", "inputs": inputs},
                    "preview": {"class_type": "PreviewImage", "inputs": {"images": ["image", 0]}}
                    }}))
                };
            let plan = compiler.compile(submission(inputs.clone())?)?;
            let node = plan
                .nodes
                .get(&NodeId::from("claude"))
                .ok_or("missing compiled Claude")?;
            assert!(node.inputs.contains_key("image_2"));
            assert!(node.inputs.contains_key("image_10"));
            assert!(node.inputs.contains_key("model.max_tokens"));
            assert!(node.inputs.contains_key("model.temperature"));
            assert_eq!(
                node.inputs.contains_key("model.reasoning_effort"),
                model != "Haiku 4.5"
            );
            plan.validate_integrity()?;
            inputs
                .as_object_mut()
                .ok_or("inputs are not an object")?
                .insert("image_21".to_owned(), serde_json::json!(["image", 0]));
            assert!(compiler.compile(submission(inputs)?).is_err());
        }
        Ok(())
    }

    #[test]
    fn claude_component_ports_match_the_canonical_model_and_image_schema()
    -> Result<(), Box<dyn Error>> {
        let types = TypeRegistry::built_in()?;
        let registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let mut ports = Vec::new();
        for (name, source_type, presence) in [
            ("prompt", "STRING", PortPresence::Required),
            ("model", "COMFY_DYNAMICCOMBO_V3", PortPresence::Required),
            ("seed", "INT", PortPresence::Required),
            ("max_tokens", "INT", PortPresence::Required),
            ("temperature", "FLOAT", PortPresence::Required),
            ("reasoning_effort", "STRING", PortPresence::Optional),
            ("system_prompt", "STRING", PortPresence::Optional),
            ("images", "IMAGE", PortPresence::Optional),
        ] {
            let mut port = input_port(&types, source_type, PortSerialization::Inline)?;
            port.id = name.to_owned();
            port.name = name.to_owned();
            port.presence = presence;
            if name == "images" {
                port.cardinality = PortCardinality::List;
                port.serialization = PortSerialization::Handle;
            }
            ports.push(port);
        }
        let mut output = input_port(&types, "STRING", PortSerialization::Inline)?;
        output.id = "output_0".to_owned();
        output.name = "output_0".to_owned();
        output.direction = PortDirection::Output;
        ports.push(output);
        let mut node = PluginNode {
            id: "ClaudeNode".to_owned(),
            version: ApiVersion {
                major: 1,
                minor: 0,
                patch: 0,
            },
            display_name: "Anthropic Claude".to_owned(),
            category: "partner/text/Anthropic".to_owned(),
            ports,
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        };
        let claim = ProviderBindingClaim {
            feature_id: "COMFY-NODE-0056".to_owned(),
            node_id: node.id.clone(),
            contract_sha256: registry
                .provider_binding_contract_sha256(
                    &node.id,
                    sim_runtime::NATIVE_PROVIDER_TRANSPORT_SCHEMA,
                    "zed:sim-provider-materializer@1",
                )?
                .ok_or("missing provider contract")?,
            transport_schema: sim_runtime::NATIVE_PROVIDER_TRANSPORT_SCHEMA.parse()?,
            materializer_schema: "zed:sim-provider-materializer@1".parse()?,
        };
        let fixture: Value = serde_json::from_str(include_str!(
            "../../sim_test_support/fixtures/providers/anthropic/provider_v2_host.fixture"
        ))?;
        let fixture_claim = fixture
            .get("bindings")
            .and_then(Value::as_array)
            .and_then(|bindings| bindings.first())
            .ok_or("missing fixture claim")?;
        assert_eq!(
            fixture_claim.get("contract_sha256").and_then(Value::as_str),
            Some(claim.contract_sha256.as_str())
        );
        assert_eq!(
            fixture_claim
                .get("transport_schema")
                .and_then(Value::as_str),
            Some(sim_runtime::NATIVE_PROVIDER_TRANSPORT_SCHEMA)
        );
        let binding = sim_plugin_sdk::ProviderBindingSet {
            schema_version: 1,
            implementation_namespace: "zed.sim.provider.anthropic".to_owned(),
            bindings_sha256: "0".repeat(64),
            bindings: vec![claim.clone()],
        };
        assert_eq!(
            fixture.get("bindings_sha256").and_then(Value::as_str),
            Some(binding.canonical_bindings_sha256()?.as_str())
        );
        assert!(
            provider_ports_match_descriptor(
                &node,
                registry
                    .descriptor("ClaudeNode")
                    .ok_or("missing Claude descriptor")?,
                &types,
            )?,
            "Claude scalar and image ports differ from the canonical schema"
        );
        validate_provider_node_projection(&registry, &node, &claim)?;
        node.ports
            .iter_mut()
            .find(|port| port.id == "reasoning_effort")
            .ok_or("missing reasoning port")?
            .presence = PortPresence::Required;
        assert!(validate_provider_node_projection(&registry, &node, &claim).is_err());
        Ok(())
    }

    #[test]
    fn named_provider_inputs_preserve_source_order_and_unrelated_values()
    -> Result<(), Box<dyn Error>> {
        let mut dynamic = NativeDynamicInputDescriptor {
            name_template: "{name}".to_owned(),
            start_index: 0,
            minimum_count: 0,
            maximum_count: 20,
            input: NativeInputDescriptor {
                name: "image".to_owned(),
                accepted_types: NativeTypeUnion::new(vec![NativeValueType::Handle(
                    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?,
                )])?,
                required: true,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal: false,
            },
        };
        let mut source = sim_nodes::NativeDynamicSchemaMetadata::compatibility(
            "{name}",
            0,
            0,
            20,
            sim_nodes::NativeInputSchemaMetadata::compatibility("image", "IMAGE"),
        );
        source.names = (1..=20).map(|index| format!("image_{index}")).collect();
        let marker = |value| NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        };
        let original = BTreeMap::from([
            ("image_10".to_owned(), marker(10)),
            ("image_2".to_owned(), marker(2)),
            ("image_1".to_owned(), marker(1)),
            ("image_21".to_owned(), marker(21)),
            ("system_prompt".to_owned(), marker(99)),
        ]);
        let mut values = original.clone();
        assert!(take_dynamic_values(&mut values, &dynamic, None).is_err());
        assert_eq!(values, original);
        dynamic.minimum_count = 4;
        assert!(take_dynamic_values(&mut values, &dynamic, Some(&source)).is_err());
        assert_eq!(values, original);
        dynamic.minimum_count = 0;
        assert_eq!(
            take_dynamic_values(&mut values, &dynamic, Some(&source))?,
            vec![marker(1), marker(2), marker(10)]
        );
        assert_eq!(
            values,
            BTreeMap::from([
                ("image_21".to_owned(), marker(21)),
                ("system_prompt".to_owned(), marker(99)),
            ])
        );
        Ok(())
    }

    #[test]
    fn claude_large_image_preparation_retains_pixels_outside_transport()
    -> Result<(), Box<dyn Error>> {
        use sim_nodes::{NativeNodeComputeSession, NativeNodeServiceIdentity, NativeNodeServices};
        use sim_tensor::{ImageTensor, NativeTensorPayload, NativeTensorRole};
        let types = TypeRegistry::built_in()?;
        let mut port = input_port(&types, "IMAGE", PortSerialization::Handle)?;
        port.id = "images".to_owned();
        port.cardinality = PortCardinality::List;
        let node = PluginNode {
            id: "ClaudeNode".to_owned(),
            version: ApiVersion {
                major: 1,
                minor: 0,
                patch: 0,
            },
            display_name: "Claude".to_owned(),
            category: "test/provider".to_owned(),
            ports: vec![port],
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        };
        let descriptor = native_descriptor(&node, &types)?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(128 * 1024 * 1024)?;
        let backend = Arc::new(backend);
        let scratch = authority.authorize_workspace(64 * 1024 * 1024)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x43610));
        let node_id = NodeId::from("large-claude-image");
        let generation = NativeHandleStoreGeneration::with_capacities(4, 128 * 1024 * 1024)?;
        let store = generation.handle_store_for_attempt(attempt_id);
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x43611),
                attempt_id,
                node_id.clone(),
            )?,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let cancellation = CancellationToken::default();
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x43612)),
            attempt_id,
            node_id,
            cancellation.clone(),
            scratch.clone(),
            store.clone(),
            NativeNodeServices::checked(None, None, Some(compute))?,
        )?;
        let execution = context.compute_session()?.execution_context(&context)?;
        let image = ImageTensor::from_f32(
            &backend,
            &execution,
            1,
            1536,
            1536,
            4,
            &vec![0.5; 1536 * 1536 * 4],
        )?;
        let payload = NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
            NativeTensorRole::Image,
            image,
        )?));
        assert!(payload.resident_bytes()? > sim_runtime::MAX_PROVIDER_TRANSPORT_REQUEST_BYTES);
        let handle = store.publish(payload.clone(), &cancellation)?;
        let duplicate = store.publish(payload, &cancellation)?;
        assert_ne!(handle, duplicate);
        let batch = ImageTensor::from_f32(
            &backend,
            &execution,
            2,
            1,
            1,
            3,
            &[0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
        )?;
        let first_frame = context.prepare_provider_image_png(&batch, 0, Some(1_568 * 1_568))?;
        let second_frame = context.prepare_provider_image_png(&batch, 1, Some(1_568 * 1_568))?;
        assert_ne!(first_frame, second_frame);
        let batch_handle = store.publish(
            NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                NativeTensorRole::Image,
                batch,
            )?)),
            &cancellation,
        )?;
        let oversized = BTreeMap::from([(
            "images".to_owned(),
            NativeValue::List {
                values: vec![
                    NativeValue::Handle {
                        value: handle.clone()
                    };
                    21
                ],
            },
        )]);
        assert!(
            futures::executor::block_on(prepare_native_provider_invocation(
                &node,
                &descriptor,
                oversized,
                "profile-a",
                &context
            ))
            .is_err()
        );
        let values = BTreeMap::from([(
            "images".to_owned(),
            NativeValue::List {
                values: vec![
                    NativeValue::Handle {
                        value: handle.clone(),
                    },
                    NativeValue::Handle { value: duplicate },
                    NativeValue::Handle {
                        value: batch_handle,
                    },
                ],
            },
        )]);
        let prepared = futures::executor::block_on(prepare_native_provider_invocation(
            &node,
            &descriptor,
            values,
            "profile-a",
            &context,
        ))?;
        let encoded = prepared.request.to_bytes()?;
        assert!(encoded.len() < 4096);
        let decoded = ProviderTransportRequest::from_bytes(&encoded)?;
        assert_eq!(decoded, prepared.request);
        assert_eq!(
            prepared
                .inputs
                .media_batches
                .get("images")
                .ok_or("missing images")?
                .len(),
            3
        );
        let batch = prepared
            .inputs
            .media_batches
            .get("images")
            .and_then(|values| values.get(&2))
            .ok_or("missing two-frame batch")?;
        assert_eq!(batch.len(), 2);
        assert_eq!(
            batch.get(&0).ok_or("missing first frame")?.bytes,
            first_frame
        );
        assert_eq!(
            batch.get(&1).ok_or("missing second frame")?.bytes,
            second_frame
        );
        let attached = prepared
            .inputs
            .media_batches
            .get("images")
            .and_then(|values| values.get(&0))
            .ok_or("missing native PNG batch")?;
        assert_eq!(attached.len(), 1);
        let media = attached.get(&0).ok_or("missing native PNG")?;
        assert!(media.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(
            media.content_sha256,
            hex_digest(&Sha256::digest(&media.bytes).into())
        );
        assert_eq!(generation.len(), 3);
        assert_eq!(scratch.in_use_bytes(), 0);
        assert!(
            store
                .resolve(&handle, handle.handle_type(), &cancellation)
                .is_ok()
        );
        Ok(())
    }

    #[test]
    fn provider_transport_outputs_are_exactly_typed_and_guest_independent()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let output = PluginPort {
            id: "result".to_owned(),
            name: "Result".to_owned(),
            direction: PortDirection::Output,
            type_id: registry.resolve("STRING")?.clone(),
            cardinality: PortCardinality::Singular,
            presence: PortPresence::Required,
            hidden: false,
            lazy: false,
            default: None,
            serialization: PortSerialization::Inline,
            accepted_legacy_names: Vec::new(),
        };
        let node = PluginNode {
            id: "provider.echo".to_owned(),
            version: ApiVersion {
                major: 1,
                minor: 0,
                patch: 0,
            },
            display_name: "Provider Echo".to_owned(),
            category: "test/provider".to_owned(),
            ports: vec![output],
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        };
        let value = PluginValue::scalar(
            registry.resolve("STRING")?.clone(),
            ScalarValue::String("provider-output".to_owned()),
            &registry,
        )?;
        let response = ProviderTransportResponse::checked(
            "provider.echo",
            vec![ProviderTransportPort::checked(
                "result",
                true,
                vec![ProviderTransportValue::checked(
                    value.type_id().to_string(),
                    value.family(),
                    value.abi_bytes()?,
                )?],
            )?],
        )?;
        let generation = NativeHandleStoreGeneration::with_capacities(8, 1024 * 1024)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x901));
        let store = generation.handle_store_for_attempt(attempt_id);
        let (_, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x902)),
            attempt_id,
            NodeId::from("provider-output"),
            CancellationToken::default(),
            authority.authorize_workspace(0)?,
            store,
        )?;
        let outputs =
            provider_transport_outputs(&node, &response, &registry, "plugin.fixture", &context)?;
        assert_eq!(
            outputs,
            vec![NativeValue::Primitive {
                value: NativePrimitive::String("provider-output".to_owned())
            }]
        );

        let wrong_class =
            ProviderTransportResponse::checked("provider.other", response.ports().to_vec())?;
        assert!(
            provider_transport_outputs(&node, &wrong_class, &registry, "plugin.fixture", &context,)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn provider_output_batch_reverses_every_publication_when_a_later_value_fails()
    -> Result<(), Box<dyn Error>> {
        let first =
            NativeStoredPayload::Provider(Arc::new(sim_nodes::NativeProviderPayload::checked(
                NativeHandleType::new(NativeHandleKind::ProviderTask, "MODEL_TASK_ID")?,
                "plugin.fixture",
                "a".repeat(64),
                b"provider-task".to_vec(),
            )?));
        let second = first.clone();
        let resident_bytes = first
            .resident_bytes()?
            .checked_mul(2)
            .ok_or("provider fixture residency overflowed")?;
        let generation = NativeHandleStoreGeneration::with_capacities(1, resident_bytes)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x920));
        let store = generation.handle_store_for_attempt(attempt_id);
        let (_, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x921)),
            attempt_id,
            NodeId::from("provider-atomic-output"),
            CancellationToken::default(),
            authority.authorize_workspace(0)?,
            store,
        )?;
        let failure = publish_provider_payloads(&context, vec![first, second])
            .expect_err("the second publication must exceed the store value bound");
        assert_eq!(failure.code, "plugin_invocation_failed");
        assert_eq!(generation.len(), 0);
        assert_eq!(generation.resident_bytes(), 0);
        Ok(())
    }

    fn input_port(
        registry: &TypeRegistry,
        type_name: &str,
        serialization: PortSerialization,
    ) -> Result<PluginPort, Box<dyn Error>> {
        Ok(PluginPort {
            id: format!("{type_name}-input"),
            name: format!("{type_name} input"),
            direction: PortDirection::Input,
            type_id: registry.resolve(type_name)?.clone(),
            cardinality: PortCardinality::Singular,
            presence: PortPresence::Required,
            hidden: false,
            lazy: false,
            default: None,
            serialization,
            accepted_legacy_names: Vec::new(),
        })
    }

    fn opaque_handle(
        handle_type: NativeHandleType,
        identifier: &str,
        generation: u64,
        digest: &str,
    ) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
        opaque_handle_with_identity(
            handle_type,
            test_store_identity()?,
            identifier,
            generation,
            digest,
        )
    }

    fn opaque_handle_with_identity(
        handle_type: NativeHandleType,
        store_identity: NativeHandleStoreIdentity,
        identifier: &str,
        generation: u64,
        digest: &str,
    ) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
        Ok(NativeOpaqueHandle::new(
            handle_type,
            store_identity,
            identifier,
            generation,
            Some(digest.to_owned()),
        )?)
    }

    fn test_store_identity() -> Result<NativeHandleStoreIdentity, Box<dyn Error>> {
        Ok(NativeHandleStoreIdentity::new(
            Uuid::from_u128(0x100),
            Uuid::from_u128(0x200),
        )?)
    }

    fn tensor_value(digest: &str) -> Result<PluginValue, Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        Ok(PluginValue::tensor(
            registry.resolve("IMAGE")?.clone(),
            TensorValue::new(
                TensorDescriptor::contiguous(
                    vec![1],
                    DType::F32,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                4,
                digest,
            )?,
            &registry,
        )?)
    }

    fn model_value(digest: &str) -> Result<PluginValue, Box<dyn Error>> {
        model_value_for("MODEL", "model", "safetensors", digest)
    }

    fn model_value_for(
        type_name: &str,
        identifier: &str,
        format: &str,
        digest: &str,
    ) -> Result<PluginValue, Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        Ok(PluginValue::model(
            registry.resolve(type_name)?.clone(),
            ModelValue::new(identifier, format, digest)?,
            &registry,
        )?)
    }

    fn clip_vision_output_tensor(
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
        shape: Vec<u64>,
        value: f32,
    ) -> Result<Tensor, Box<dyn Error>> {
        let elements = shape
            .iter()
            .try_fold(1_u64, |total, dimension| total.checked_mul(*dimension))
            .ok_or("clip vision output tensor element count overflowed")?;
        let descriptor =
            TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(
                elements
                    .checked_mul(4)
                    .ok_or("clip vision output tensor byte count overflowed")?,
            )?,
            cancellation,
        );
        Ok(backend
            .upload_f32(
                descriptor,
                &vec![value; usize::try_from(elements)?],
                &context,
            )?
            .0)
    }

    #[test]
    fn signed_plugin_families_delegate_to_canonical_source_projections()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let cases = [
            (
                "String",
                NativeValueType::Primitive(NativePrimitiveType::String),
            ),
            (
                "Image",
                NativeValueType::Handle(NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?),
            ),
            (
                "SVG",
                NativeValueType::Handle(NativeHandleType::new(NativeHandleKind::Artifact, "SVG")?),
            ),
            (
                "Model",
                NativeValueType::Handle(NativeHandleType::new(NativeHandleKind::Model, "MODEL")?),
            ),
            (
                "AudioEncoderOutput",
                NativeValueType::Handle(NativeHandleType::new(
                    NativeHandleKind::StructuredCompute,
                    "AUDIO_ENCODER_OUTPUT",
                )?),
            ),
            (
                "File3DGLB",
                NativeValueType::Handle(NativeHandleType::new(
                    NativeHandleKind::ThreeD,
                    "FILE_3D_GLB",
                )?),
            ),
        ];
        for (source_type, expected) in cases {
            let port = input_port(&registry, source_type, PortSerialization::Handle)?;
            assert_eq!(
                native_value_type(&port, registry.family(&port.type_id)?)?,
                expected
            );
        }
        let curve = input_port(&registry, "Curve", PortSerialization::Handle)?;
        assert!(native_value_type(&curve, registry.family(&curve.type_id)?).is_err());
        Ok(())
    }

    #[test]
    fn imported_tensor_artifact_and_model_handles_round_trip_exactly() -> Result<(), Box<dyn Error>>
    {
        let registry = TypeRegistry::built_in()?;
        let digest = "a".repeat(64);
        let cases = [
            (
                input_port(&registry, "IMAGE", PortSerialization::Handle)?,
                tensor_value(&digest)?,
            ),
            (
                input_port(&registry, "SVG", PortSerialization::ArtifactReference)?,
                PluginValue::artifact(
                    registry.resolve("SVG")?.clone(),
                    ArtifactValue::new("input", "fixture.svg", 4, &digest)?,
                    &registry,
                )?,
            ),
            (
                input_port(&registry, "MODEL", PortSerialization::Handle)?,
                model_value(&digest)?,
            ),
        ];
        for (index, (port, value)) in cases.into_iter().enumerate() {
            let handle = opaque_handle(
                native_handle_type(&port)?,
                &format!("imported-{index}"),
                u64::try_from(index)? + 1,
                &digest,
            )?;
            let key = (port.type_id.to_string(), digest.clone());
            let mut imported = ImportedHandles::new(test_store_identity()?);
            remember_imported_handle(&mut imported, key, handle.clone());
            assert_eq!(
                runtime_value(value, &port, &registry, "profile-a", &imported)?,
                NativeValue::Handle { value: handle }
            );
        }
        Ok(())
    }

    #[test]
    fn explicit_model_family_handles_round_trip_without_mutating_imports()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        for (index, type_name) in [
            "Model",
            "Clip",
            "Vae",
            "ControlNet",
            "Guider",
            "Sampler",
            "ClipVision",
        ]
        .into_iter()
        .enumerate()
        {
            let digest = format!("{index:x}").repeat(64);
            let port = input_port(&registry, type_name, PortSerialization::Handle)?;
            let value = model_value_for(
                type_name,
                &format!("{type_name}-identity"),
                "zed-native-test-v1",
                &digest,
            )?;
            let handle = opaque_handle(
                native_handle_type(&port)?,
                &format!("imported-{type_name}"),
                u64::try_from(index)? + 1,
                &digest,
            )?;
            let mut imported = ImportedHandles::new(test_store_identity()?);
            remember_imported_handle(
                &mut imported,
                (port.type_id.to_string(), digest),
                handle.clone(),
            );
            let before = imported.clone();
            assert_eq!(
                runtime_value(value, &port, &registry, "profile-a", &imported)?,
                NativeValue::Handle { value: handle }
            );
            assert_eq!(imported, before);
        }
        Ok(())
    }

    #[test]
    fn clip_vision_output_input_rejection_preserves_the_canonical_store_entry()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let port = input_port(&registry, "ClipVisionOutput", PortSerialization::Handle)?;
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let hidden =
            clip_vision_output_tensor(&backend, &authority, &cancellation, vec![1, 2, 2], 1.0)?;
        let embeds =
            clip_vision_output_tensor(&backend, &authority, &cancellation, vec![1, 2], 2.0)?;
        let output = Arc::new(ClipVisionOutput::checked(
            hidden,
            None,
            embeds,
            None,
            vec![[3, 32, 32]],
        )?);
        let payload = NativeStoredPayload::ClipVisionOutput(output.clone());
        let payload_bytes = payload.resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(1, payload_bytes)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x300));
        let store = generation.handle_store_for_attempt(attempt_id);
        let handle = store.publish(payload, &cancellation)?;
        let expected_type = NativeHandleType::new(
            NativeHandleKind::StructuredCompute,
            ClipVisionOutput::SOURCE_TYPE_ID,
        )?;
        assert_eq!(handle.handle_type(), &expected_type);
        let before_len = generation.len();
        let before_resident_bytes = generation.resident_bytes();
        let context = NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x400)),
            attempt_id,
            NodeId::from("clip-vision-output-plugin-input"),
            cancellation.clone(),
            authority.authorize_workspace(0)?,
            store.clone(),
        )?;

        let failure = plugin_value(
            &port,
            NativeValue::Handle {
                value: handle.clone(),
            },
            &registry,
            "profile-a",
            &context,
        )
        .expect_err("CLIP_VISION_OUTPUT has no lossless plugin SDK representation");
        assert_eq!(failure.code, "unmaterialized_plugin_payload");
        assert_eq!(generation.len(), before_len);
        assert_eq!(generation.resident_bytes(), before_resident_bytes);

        let resolved = store.resolve(&handle, &expected_type, &cancellation)?;
        let NativeStoredPayload::ClipVisionOutput(resolved_output) = resolved.as_ref() else {
            return Err("resolved CLIP_VISION_OUTPUT changed stored payload variant".into());
        };
        assert!(Arc::ptr_eq(resolved_output, &output));
        assert_eq!(
            resolved_output.semantic_digest_sha256(),
            output.semantic_digest_sha256()
        );
        Ok(())
    }

    #[test]
    fn latent_bundle_input_rejection_preserves_the_canonical_store_entry()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let port = input_port(&registry, "LATENT", PortSerialization::Handle)?;
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let samples =
            clip_vision_output_tensor(&backend, &authority, &cancellation, vec![1, 4, 2, 2], 1.0)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let bundle = Arc::new(NativeLatentBundle::single(
            samples,
            None,
            Some(vec![7]),
            NativeLatentMetadata::checked(None, None, Some(8), None)?,
            &context,
        )?);
        let payload = NativeStoredPayload::Latent(bundle.clone());
        let payload_bytes = payload.resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(1, payload_bytes)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x301));
        let store = generation.handle_store_for_attempt(attempt_id);
        let handle = store.publish(payload, &cancellation)?;
        let expected_type = NativeHandleType::new(NativeHandleKind::Latent, "LATENT")?;
        assert_eq!(handle.handle_type(), &expected_type);
        let before_len = generation.len();
        let before_resident_bytes = generation.resident_bytes();
        let node_context = NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x401)),
            attempt_id,
            NodeId::from("latent-bundle-plugin-input"),
            cancellation.clone(),
            authority.authorize_workspace(0)?,
            store.clone(),
        )?;

        let failure = plugin_value(
            &port,
            NativeValue::Handle {
                value: handle.clone(),
            },
            &registry,
            "profile-a",
            &node_context,
        )
        .expect_err("LATENT bundle has no lossless plugin SDK representation");
        assert_eq!(failure.code, "unmaterialized_plugin_payload");
        assert_eq!(generation.len(), before_len);
        assert_eq!(generation.resident_bytes(), before_resident_bytes);

        let resolved = store.resolve(&handle, &expected_type, &cancellation)?;
        let NativeStoredPayload::Latent(resolved_bundle) = resolved.as_ref() else {
            return Err("resolved LATENT changed stored payload variant".into());
        };
        assert!(Arc::ptr_eq(resolved_bundle, &bundle));
        assert_eq!(
            resolved_bundle.semantic_digest_sha256(),
            bundle.semantic_digest_sha256()
        );
        Ok(())
    }

    #[test]
    fn explicit_stored_variants_are_exhaustively_projected_or_rejected()
    -> Result<(), Box<dyn Error>> {
        let source = include_str!("registry_adapter.rs");
        let projection = source
            .split_once("fn plugin_value_from_stored(")
            .and_then(|(_, source)| source.split_once("fn scalar_from_native("))
            .map(|(projection, _)| projection)
            .ok_or("stored payload projection function is missing")?;
        for projected in [
            "NativeStoredPayload::Provider(stored)",
            "NativeStoredPayload::Tensor(stored)",
            "NativeStoredPayload::Model(stored)",
            "NativeStoredPayload::Control(stored)",
            "NativeStoredPayload::Guider(stored)",
            "NativeStoredPayload::Sampler(stored)",
            "NativeStoredPayload::Audio(stored)",
            "NativeStoredPayload::Video(stored)",
            "NativeStoredPayload::Artifact(stored)",
            "NativeStoredPayload::File3D(stored)",
            "NativeStoredPayload::Camera(stored)",
        ] {
            assert!(
                projection.contains(projected),
                "missing projection {projected}"
            );
        }
        for rejected in [
            "NativeStoredPayload::Latent(_)",
            "NativeStoredPayload::Conditioning(_)",
            "NativeStoredPayload::Noise(_)",
            "NativeStoredPayload::BoundingBox(_)",
            "NativeStoredPayload::FaceLandmarks(_)",
            "NativeStoredPayload::PoseKeypoint(_)",
            "NativeStoredPayload::Sam3TrackData(_)",
            "NativeStoredPayload::Tracks(_)",
            "NativeStoredPayload::AudioEncoderOutput(_)",
            "NativeStoredPayload::ClipVisionOutput(_)",
            "NativeStoredPayload::IcLoraParameters(_)",
            "NativeStoredPayload::LossMap(_)",
        ] {
            assert!(
                projection.contains(rejected),
                "missing rejection {rejected}"
            );
        }
        assert!(projection.contains("Err(unmaterialized_plugin_input(&port.id))"));
        assert!(!projection.contains("StructuredCompute"));
        assert!(!projection.contains("serde_json"));
        assert!(!projection.contains("remember_imported_handle"));
        assert!(!projection.contains(".publish("));
        assert!(!projection.contains(".revoke("));
        assert!(projection.contains("stored.model_payload().identity()"));
        assert!(!projection.contains("stored.diffusion()"));

        let failure = unmaterialized_plugin_input("tracks");
        assert_eq!(failure.code, "unmaterialized_plugin_payload");
        assert!(failure.message.contains("no lossless representation"));
        let failure = unmaterialized_plugin_input("conditioning");
        assert_eq!(failure.code, "unmaterialized_plugin_payload");
        assert!(failure.message.contains("no lossless representation"));
        Ok(())
    }

    #[test]
    fn forged_or_unmaterialized_plugin_outputs_fail_without_changing_imports()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let digest = "b".repeat(64);
        let port = input_port(&registry, "IMAGE", PortSerialization::Handle)?;
        let value = tensor_value(&digest)?;
        let mut imported = ImportedHandles::new(test_store_identity()?);
        let wrong_type = opaque_handle(
            NativeHandleType::new(NativeHandleKind::Model, "MODEL")?,
            "wrong-type",
            1,
            &digest,
        )?;
        remember_imported_handle(
            &mut imported,
            (port.type_id.to_string(), digest.clone()),
            wrong_type,
        );
        let before = imported.clone();
        let failure = runtime_value(value.clone(), &port, &registry, "profile-a", &imported)
            .expect_err("a wrong-type imported handle must fail closed");
        assert_eq!(failure.code, "invalid_plugin_value");
        assert_eq!(imported, before);

        let unmaterialized = ImportedHandles::new(test_store_identity()?);
        let failure = runtime_value(
            value.clone(),
            &port,
            &registry,
            "profile-a",
            &unmaterialized,
        )
        .expect_err("metadata-only output must not allocate a native handle");
        assert_eq!(failure.code, "unmaterialized_plugin_payload");
        assert!(unmaterialized.handles.is_empty());

        let first = opaque_handle(native_handle_type(&port)?, "first", 1, &digest)?;
        let second = opaque_handle(native_handle_type(&port)?, "second", 2, &digest)?;
        let mut ambiguous = ImportedHandles::new(test_store_identity()?);
        let key = (port.type_id.to_string(), digest);
        remember_imported_handle(&mut ambiguous, key.clone(), first);
        remember_imported_handle(&mut ambiguous, key, second);
        let failure = runtime_value(value, &port, &registry, "profile-a", &ambiguous)
            .expect_err("equal metadata from distinct handles must not alias");
        assert_eq!(failure.code, "unmaterialized_plugin_payload");

        for (label, handle) in [
            (
                "store",
                opaque_handle_with_identity(
                    native_handle_type(&port)?,
                    NativeHandleStoreIdentity::new(Uuid::from_u128(0x101), Uuid::from_u128(0x200))?,
                    "wrong-store",
                    1,
                    &"b".repeat(64),
                )?,
            ),
            (
                "generation",
                opaque_handle_with_identity(
                    native_handle_type(&port)?,
                    NativeHandleStoreIdentity::new(Uuid::from_u128(0x100), Uuid::from_u128(0x201))?,
                    "wrong-generation",
                    1,
                    &"b".repeat(64),
                )?,
            ),
            (
                "digest",
                opaque_handle(
                    native_handle_type(&port)?,
                    "wrong-digest",
                    1,
                    &"c".repeat(64),
                )?,
            ),
        ] {
            let mut forged = ImportedHandles::new(test_store_identity()?);
            remember_imported_handle(
                &mut forged,
                (port.type_id.to_string(), "b".repeat(64)),
                handle,
            );
            let failure = runtime_value(
                tensor_value(&"b".repeat(64))?,
                &port,
                &registry,
                "profile-a",
                &forged,
            )
            .expect_err("forged imported handle provenance must fail closed");
            assert_eq!(failure.code, "invalid_plugin_value", "{label}");
        }
        Ok(())
    }

    #[test]
    fn unmaterialized_plugin_payloads_cannot_forge_native_store_objects()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let port = input_port(&registry, "IMAGE", PortSerialization::Handle)?;
        let payload =
            NativeStoredPayload::Provider(Arc::new(sim_nodes::NativeProviderPayload::checked(
                NativeHandleType::new(NativeHandleKind::ProviderTask, "TEST_TASK")?,
                "test.signed",
                "1".repeat(64),
                vec![1],
            )?));
        let failure = plugin_value_from_stored(&port, ValueFamily::Tensor, &payload, &registry)
            .expect_err("an unrelated sealed payload must not satisfy an IMAGE input");
        assert_eq!(failure.code, "invalid_plugin_value");

        let failure = unmaterialized_plugin_output("image");
        assert_eq!(failure.code, "unmaterialized_plugin_payload");
        assert!(failure.message.contains("host-materialized payload"));

        let artifact_port = input_port(&registry, "SVG", PortSerialization::ArtifactReference)?;
        let artifact = artifact_value("provider.svg")?;
        let semantic_digest = value_digest(&artifact).ok_or("artifact digest is absent")?;
        let provider = sim_nodes::NativeProviderPayload::checked(
            NativeHandleType::new(NativeHandleKind::ProviderTask, "SVG")?,
            "test.provider",
            semantic_digest,
            artifact.abi_bytes()?,
        )?;
        let failure = plugin_value_from_stored(
            &artifact_port,
            ValueFamily::Artifact,
            &NativeStoredPayload::Provider(Arc::new(provider)),
            &registry,
        )
        .expect_err("artifact provider payloads are not provider-task identities");
        assert_eq!(failure.code, "invalid_plugin_value");
        Ok(())
    }

    #[test]
    fn registry_adapter_rejects_noncanonical_artifact_abi_paths() -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let port = PluginPort {
            id: "artifact".to_owned(),
            name: "artifact".to_owned(),
            direction: PortDirection::Output,
            type_id: registry.resolve("SVG")?.clone(),
            cardinality: PortCardinality::Singular,
            presence: PortPresence::Required,
            hidden: false,
            lazy: false,
            default: None,
            serialization: sim_plugin_sdk::PortSerialization::ArtifactReference,
            accepted_legacy_names: Vec::new(),
        };
        artifact_value_identity(
            "profile-a",
            match artifact_value("nested/fixture.svg")?.representation() {
                PluginValueRepresentation::Artifact(value) => value,
                _ => return Err("fixture is not an artifact".into()),
            },
        )?;
        let invalid = artifact_value("../escape.svg")?;
        let PluginValueRepresentation::Artifact(invalid) = invalid.representation() else {
            return Err("fixture is not an artifact".into());
        };
        assert!(artifact_value_identity("profile-a", invalid).is_err());
        assert_eq!(native_handle_type(&port)?.type_id, "SVG");
        Ok(())
    }
}

#[cfg(test)]
mod bfl_ultra_media_wiring_tests {
    use super::*;
    use serde_json::Value;
    use sim_nodes::{
        NativeHandleKind, NativeHandleType, NativeNodeComputeSession, NativeNodeContext,
        NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle, NativeStoredPayload,
        NativeValue,
    };
    use sim_plugin_sdk::{
        ApiVersion, CachePolicy, DeterminismPolicy, EffectPolicy, PluginNode, PluginPort,
        PluginValue, PortCardinality, PortDirection, PortPresence, PortSerialization, TypeRegistry,
    };
    use sim_runtime::{NativeHandleStoreGeneration, ProviderTransportEncoding};
    use sim_tensor::{
        CancellationToken, CpuWorkspaceAuthority, ImageTensor, NativeTensorPayload,
        NativeTensorRole, StreamId,
    };
    use sim_types::{AttemptId, NodeId, PromptId};

    use std::{collections::BTreeMap, error::Error, sync::Arc};
    use uuid::Uuid;

    fn node(port_id: &str) -> Result<PluginNode, Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        Ok(PluginNode {
            id: "FluxProUltraImageNode".to_owned(),
            version: ApiVersion::new(1, 0, 0),
            display_name: "BFL Ultra input preparation".to_owned(),
            category: "test/provider".to_owned(),
            ports: vec![PluginPort {
                id: port_id.to_owned(),
                name: port_id.to_owned(),
                direction: PortDirection::Input,
                type_id: registry.resolve("IMAGE")?.clone(),
                cardinality: PortCardinality::Singular,
                presence: PortPresence::Optional,
                hidden: false,
                lazy: false,
                default: None,
                serialization: PortSerialization::Handle,
                accepted_legacy_names: vec![],
            }],
            determinism: DeterminismPolicy::External,
            cache: CachePolicy::Never,
            effects: EffectPolicy::Provider,
        })
    }

    fn image_context(
        shape: &[u64],
        pixels: &[f32],
        capacity: u64,
        identity: u128,
    ) -> Result<
        (
            NativeNodeContext,
            NativeHandleStoreGeneration,
            NativeOpaqueHandle,
        ),
        Box<dyn Error>,
    > {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(capacity)?;
        let backend = Arc::new(backend);
        let workspace = authority.authorize_workspace(capacity)?;
        let attempt_id = AttemptId(Uuid::from_u128(identity));
        let node_id = NodeId::from("bfl-ultra-image-input");
        let generation =
            NativeHandleStoreGeneration::with_capacities(4, usize::try_from(capacity)?)?;
        let store = generation.handle_store_for_attempt(attempt_id);
        let compute = NativeNodeComputeSession::checked(
            NativeNodeServiceIdentity::checked(
                Uuid::from_u128(identity + 1),
                attempt_id,
                node_id.clone(),
            )?,
            backend.clone(),
            StreamId::DEFAULT,
            &workspace,
        )?;
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(identity + 2)),
            attempt_id,
            node_id,
            CancellationToken::default(),
            workspace,
            store.clone(),
            NativeNodeServices::checked(None, None, Some(compute))?,
        )?;
        let [batch, height, width, channels] = shape else {
            return Err("expected BHWC shape".into());
        };
        let execution = context.compute_session()?.execution_context(&context)?;
        let image = ImageTensor::from_f32(
            &backend, &execution, *batch, *height, *width, *channels, pixels,
        )?;
        let handle = store.publish(
            NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                NativeTensorRole::Image,
                image,
            )?)),
            &context.cancellation,
        )?;
        Ok((context, generation, handle))
    }

    fn prepare(
        node: &PluginNode,
        context: &NativeNodeContext,
        handle: NativeOpaqueHandle,
    ) -> Result<PreparedNativeProviderInvocation, sim_nodes::NativeNodeFailure> {
        let registry =
            sim_runtime::generated_native_node_registry_projection(None).map_err(|error| {
                sim_nodes::NativeNodeFailure {
                    code: "test_registry".to_owned(),
                    message: error.to_string(),
                    kind: sim_nodes::NativeNodeFailureKind::Failure,
                    retryable: false,
                }
            })?;
        let descriptor = registry
            .descriptor("FluxProUltraImageNode")
            .ok_or_else(|| sim_nodes::NativeNodeFailure {
                code: "test_descriptor".to_owned(),
                message: "missing BFL descriptor".to_owned(),
                kind: sim_nodes::NativeNodeFailureKind::Failure,
                retryable: false,
            })?;
        smol::block_on(prepare_native_provider_invocation(
            node,
            descriptor,
            BTreeMap::from([(
                node.ports
                    .first()
                    .ok_or_else(|| sim_nodes::NativeNodeFailure {
                        code: "test_port".to_owned(),
                        message: "missing port".to_owned(),
                        kind: sim_nodes::NativeNodeFailureKind::Failure,
                        retryable: false,
                    })?
                    .id
                    .clone(),
                NativeValue::Handle { value: handle },
            )]),
            "profile-a",
            context,
        ))
    }

    fn attachment(prepared: &PreparedNativeProviderInvocation) -> Result<Value, Box<dyn Error>> {
        let inputs = serde_json::to_value(&prepared.inputs)?;
        let batch = inputs
            .pointer("/media_batches/image_prompt/0")
            .and_then(Value::as_object)
            .ok_or("missing BFL PNG batch")?;
        assert_eq!(batch.len(), 1);
        Ok(batch.get("0").ok_or("missing first frame")?.clone())
    }

    #[test]
    fn bfl_ultra_media_wiring_preserves_first_frame_and_compact_tensor_identity()
    -> Result<(), Box<dyn Error>> {
        let node = node("image_prompt")?;
        let mut previous_png = None;
        let mut previous_source = None;
        let first = [
            0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.25, 0.75, 0.0, 1.0, 0.0, 0.75,
        ];
        for (index, second_value) in [0.0, 0.75].into_iter().enumerate() {
            let mut pixels = first.to_vec();
            pixels.extend([second_value; 12]);
            let (context, generation, handle) = image_context(
                &[2, 2, 2, 3],
                &pixels,
                64 * 1024,
                0x4371_0000 + u128::try_from(index)? * 4,
            )?;
            let resident = generation.resident_bytes();
            let expected = NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?;
            let payload =
                context
                    .handle_store()
                    .resolve(&handle, &expected, &context.cancellation)?;
            let NativeStoredPayload::Tensor(tensor) = payload.as_ref() else {
                return Err("expected IMAGE tensor".into());
            };
            let expected_png = context.prepare_provider_image_png(
                tensor.image().ok_or("missing canonical image")?,
                0,
                Some(2048 * 2048),
            )?;
            let prepared = prepare(&node, &context, handle.clone())?;
            let media = attachment(&prepared)?;
            let bytes: Vec<u8> = serde_json::from_value(media["bytes"].clone())?;
            assert_eq!(bytes, expected_png);
            let transport = prepared
                .request
                .ports()
                .first()
                .ok_or("missing request port")?
                .values()
                .first()
                .ok_or("missing compact value")?;
            assert_eq!(transport.encoding(), ProviderTransportEncoding::PluginValue);
            let image_value =
                PluginValue::from_abi_bytes(transport.abi_bytes(), &TypeRegistry::built_in()?)?;
            assert_eq!(image_value.family(), ValueFamily::Tensor);
            assert_eq!(
                media["source_value_sha256"],
                hex_digest(&Sha256::digest(transport.abi_bytes()).into())
            );
            assert_eq!(
                media["content_sha256"],
                hex_digest(&Sha256::digest(&bytes).into())
            );
            let decoded = sim_media::decode_downloaded_image_rgba(
                &bytes,
                sim_media::PngLimits::default(),
                &context.cancellation,
            )?;
            assert_eq!((decoded.width, decoded.height), (2, 2));
            assert_eq!(
                decoded.pixels_bhwc,
                vec![
                    0.0,
                    127.0 / 255.0,
                    1.0,
                    1.0,
                    1.0,
                    127.0 / 255.0,
                    0.0,
                    1.0,
                    63.0 / 255.0,
                    191.0 / 255.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    191.0 / 255.0,
                    1.0
                ]
            );
            if let Some(previous) = &previous_png {
                assert_eq!(&bytes, previous);
            }
            previous_png = Some(bytes);
            if let Some(previous) = &previous_source {
                assert_ne!(&media["source_value_sha256"], previous);
            }
            previous_source = Some(media["source_value_sha256"].clone());
            assert!(prepared.request.to_bytes()?.len() < 4096);
            assert!(
                context
                    .handle_store()
                    .resolve(&handle, &expected, &context.cancellation)
                    .is_ok()
            );
            assert_eq!(generation.len(), 1);
            assert_eq!(generation.resident_bytes(), resident);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        Ok(())
    }

    #[test]
    fn bfl_ultra_media_wiring_optional_absence_and_other_nodes_keep_generic_transport()
    -> Result<(), Box<dyn Error>> {
        let node = node("image_prompt")?;
        let (context, generation, handle) =
            image_context(&[1, 2, 2, 3], &[0.5; 12], 64 * 1024, 0x4371_0100)?;
        let registry = sim_runtime::generated_native_node_registry_projection(None)?;
        let descriptor = registry
            .descriptor("FluxProUltraImageNode")
            .ok_or("missing descriptor")?;
        let prepared = smol::block_on(prepare_native_provider_invocation(
            &node,
            descriptor,
            BTreeMap::new(),
            "profile-a",
            &context,
        ))?;
        let inputs = serde_json::to_value(prepared.inputs)?;
        assert_eq!(inputs.pointer("/values/image_prompt"), Some(&Value::Null));
        assert!(inputs.pointer("/media_batches/image_prompt").is_none());
        for (node_id, port_id) in [
            ("FluxProUltraImageNode", "reference_image"),
            ("Flux2ImageNode", "image_prompt"),
        ] {
            let mut different = self::node(port_id)?;
            different.id = node_id.to_owned();
            let prepared = prepare(&different, &context, handle.clone())?;
            let inputs = serde_json::to_value(prepared.inputs)?;
            assert!(inputs.pointer("/media_batches/image_prompt").is_none());
            assert_eq!(
                prepared
                    .request
                    .ports()
                    .first()
                    .ok_or("missing port")?
                    .values()
                    .first()
                    .ok_or("missing value")?
                    .encoding(),
                ProviderTransportEncoding::NativePayload
            );
        }
        assert_eq!(generation.len(), 1);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn bfl_ultra_media_wiring_rejects_foreign_stale_wrong_kind_and_cancelled_handles_without_leaks()
    -> Result<(), Box<dyn Error>> {
        let node = node("image_prompt")?;
        let (context, generation, handle) =
            image_context(&[1, 1, 1, 3], &[0.0, 0.5, 1.0], 64 * 1024, 0x4432_0000)?;
        let (foreign_context, foreign_generation, foreign) =
            image_context(&[1, 1, 1, 3], &[1.0, 0.5, 0.0], 64 * 1024, 0x4432_0010)?;
        let stale = NativeOpaqueHandle::new(
            handle.handle_type().clone(),
            handle.store_identity(),
            handle.identifier(),
            handle.generation() + 1,
            handle.digest_sha256().map(str::to_owned),
        )?;
        let wrong_kind = NativeOpaqueHandle::new(
            NativeHandleType::new(NativeHandleKind::Mask, "MASK")?,
            handle.store_identity(),
            handle.identifier(),
            handle.generation(),
            handle.digest_sha256().map(str::to_owned),
        )?;
        let resident = generation.resident_bytes();
        for rejected in [foreign, stale, wrong_kind] {
            assert!(prepare(&node, &context, rejected).is_err());
            assert_eq!(generation.len(), 1);
            assert_eq!(generation.resident_bytes(), resident);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        assert_eq!(foreign_generation.len(), 1);
        assert_eq!(foreign_context.scratch.in_use_bytes(), 0);
        assert!(context.cancellation.cancel());
        assert!(prepare(&node, &context, handle).is_err());
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), resident);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn bfl_ultra_media_wiring_backend_allocation_failure_and_wrong_port_leave_no_media_or_handles()
    -> Result<(), Box<dyn Error>> {
        let (context, generation, handle) =
            image_context(&[1, 4, 4, 3], &[0.5; 48], 48 * 4, 0x4433_0000)?;
        let resident = generation.resident_bytes();
        let result = prepare(&node("image_prompt")?, &context, handle);
        let failure = result
            .err()
            .ok_or("PNG preparation succeeded despite exhausted actual backend")?;
        assert_eq!(failure.code, "provider_image_preparation_failed");
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), resident);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        let (context, generation, handle) = image_context(
            &[2, 1, 1, 3],
            &[0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
            64 * 1024,
            0x4433_0010,
        )?;
        let prepared = prepare(&node("reference_image")?, &context, handle)?;
        let json = serde_json::to_value(&prepared.inputs)?;
        assert!(json.pointer("/media_batches/image").is_none());
        assert_eq!(
            prepared
                .request
                .ports()
                .first()
                .ok_or("missing request")?
                .values()
                .first()
                .ok_or("missing value")?
                .encoding(),
            ProviderTransportEncoding::NativePayload
        );
        assert_eq!(generation.len(), 1);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }
}
