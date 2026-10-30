use crate::ComponentHost;
use sim_plugin_sdk::{
    ApiVersion, CanonicalTypeId, CapabilityKind, EffectPolicy, LegacyInputTranslation,
    LegacyMapping, PluginManifest, PluginNode, PortCardinality, PortDirection, PortPresence,
    PortSerialization,
};
use sim_runtime::{Capability, PermissionError, PluginAuthorization, TrustError};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const MAX_LEGACY_REFERENCE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MappingTarget {
    pub plugin_identifier: String,
    pub node_identifier: String,
    pub node_version: ApiVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MappingSource {
    WorkflowPin,
    UserChoice,
    SignedRegistry,
    UniqueInstalled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappingCandidate {
    target: MappingTarget,
    provenance: String,
    compatibility_by_legacy_identifier: BTreeMap<String, LegacyCompatibilityProjection>,
    #[cfg(not(test))]
    _authorization: PluginAuthorization,
    #[cfg(test)]
    _authorization: Option<PluginAuthorization>,
}

impl MappingCandidate {
    pub fn new(
        target: MappingTarget,
        provenance: impl Into<String>,
        manifest: &PluginManifest,
        authorization: &PluginAuthorization,
    ) -> Result<Self, LegacyMappingError> {
        authorization.require_manifest(manifest)?;
        let provenance = provenance.into();
        if target.plugin_identifier != authorization.plugin_id()
            || provenance.trim().is_empty()
            || provenance != provenance.trim()
            || provenance.len() > 2_048
            || provenance.chars().any(char::is_control)
            || !manifest.nodes.iter().any(|node| {
                node.id == target.node_identifier && node.version == target.node_version
            })
        {
            return Err(LegacyMappingError::InvalidCandidate);
        }
        let node = manifest
            .nodes
            .iter()
            .find(|node| node.id == target.node_identifier && node.version == target.node_version)
            .ok_or(LegacyMappingError::InvalidCandidate)?;
        let compatibility_by_legacy_identifier = manifest
            .legacy_mappings
            .iter()
            .filter(|mapping| {
                mapping.node_id == target.node_identifier
                    && mapping.node_version == target.node_version
            })
            .map(|mapping| {
                Ok((
                    mapping.legacy_identifier.clone(),
                    LegacyCompatibilityProjection::from_manifest_node(manifest, node, mapping)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, LegacyMappingError>>()?;
        Ok(Self {
            target,
            provenance,
            compatibility_by_legacy_identifier,
            #[cfg(not(test))]
            _authorization: authorization.clone(),
            #[cfg(test)]
            _authorization: Some(authorization.clone()),
        })
    }

    #[cfg(test)]
    fn new_for_test(target: MappingTarget, provenance: impl Into<String>) -> Self {
        Self {
            target,
            provenance: provenance.into(),
            compatibility_by_legacy_identifier: BTreeMap::from([(
                "LegacyNode".to_owned(),
                LegacyCompatibilityProjection::default(),
            )]),
            _authorization: None,
        }
    }

    pub fn target(&self) -> &MappingTarget {
        &self.target
    }

    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    pub fn compatibility_for(
        &self,
        legacy_identifier: &str,
    ) -> Option<&LegacyCompatibilityProjection> {
        self.compatibility_by_legacy_identifier
            .get(legacy_identifier)
    }

    fn declares_legacy_identifier(&self, legacy_identifier: &str) -> bool {
        self.compatibility_by_legacy_identifier
            .contains_key(legacy_identifier)
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum LegacyMappingError {
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("legacy mapping candidate does not match its authorized plugin manifest")]
    InvalidCandidate,
    #[error("legacy node reference is invalid")]
    InvalidReference,
    #[error("legacy node reference exceeds the {MAX_LEGACY_REFERENCE_BYTES}-byte limit")]
    ReferenceTooLarge,
    #[error("legacy mapping `{0}` is not declared for the authorized plugin node")]
    UndeclaredLegacyMapping(String),
    #[error("provider node `{0}` has no declared provider capability")]
    ProviderNodeWithoutCapability(String),
    #[error(transparent)]
    Permission(#[from] PermissionError),
    #[error("installed component projection failed: {0}")]
    ComponentHost(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyNodeReference {
    legacy_identifier: String,
    serialized_fields: Vec<u8>,
    serialized_widgets: Vec<u8>,
    serialized_links: Vec<u8>,
    extension_data: Vec<u8>,
}

impl LegacyNodeReference {
    pub fn new(
        legacy_identifier: impl Into<String>,
        serialized_fields: Vec<u8>,
        serialized_widgets: Vec<u8>,
        serialized_links: Vec<u8>,
        extension_data: Vec<u8>,
    ) -> Result<Self, LegacyMappingError> {
        let reference = Self {
            legacy_identifier: legacy_identifier.into(),
            serialized_fields,
            serialized_widgets,
            serialized_links,
            extension_data,
        };
        reference.validate()?;
        Ok(reference)
    }

    pub fn legacy_identifier(&self) -> &str {
        &self.legacy_identifier
    }

    pub fn serialized_fields(&self) -> &[u8] {
        &self.serialized_fields
    }

    pub fn serialized_widgets(&self) -> &[u8] {
        &self.serialized_widgets
    }

    pub fn serialized_links(&self) -> &[u8] {
        &self.serialized_links
    }

    pub fn extension_data(&self) -> &[u8] {
        &self.extension_data
    }

    fn validate(&self) -> Result<(), LegacyMappingError> {
        validate_legacy_identifier(&self.legacy_identifier)?;
        let total_bytes = [
            self.legacy_identifier.len(),
            self.serialized_fields.len(),
            self.serialized_widgets.len(),
            self.serialized_links.len(),
            self.extension_data.len(),
        ]
        .into_iter()
        .try_fold(0_usize, usize::checked_add)
        .ok_or(LegacyMappingError::ReferenceTooLarge)?;
        if total_bytes > MAX_LEGACY_REFERENCE_BYTES {
            return Err(LegacyMappingError::ReferenceTooLarge);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyPortTranslation {
    direction: PortDirection,
    target_position: u32,
    accepted_legacy_names: Vec<String>,
    target_port_id: String,
    target_port_name: String,
    type_id: CanonicalTypeId,
    cardinality: PortCardinality,
    presence: PortPresence,
    serialization: PortSerialization,
    lazy: bool,
}

impl LegacyPortTranslation {
    pub fn direction(&self) -> PortDirection {
        self.direction
    }

    pub fn target_position(&self) -> u32 {
        self.target_position
    }

    pub fn accepted_legacy_names(&self) -> &[String] {
        &self.accepted_legacy_names
    }

    pub fn target_port_id(&self) -> &str {
        &self.target_port_id
    }

    pub fn target_port_name(&self) -> &str {
        &self.target_port_name
    }

    pub fn type_id(&self) -> &CanonicalTypeId {
        &self.type_id
    }

    pub fn cardinality(&self) -> PortCardinality {
        self.cardinality
    }

    pub fn presence(&self) -> PortPresence {
        self.presence
    }

    pub fn serialization(&self) -> PortSerialization {
        self.serialization
    }

    pub fn lazy(&self) -> bool {
        self.lazy
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LegacyInputSourceProjection {
    LegacyInput {
        legacy_input_id: String,
        legacy_widget_position: Option<u32>,
    },
    Constant {
        canonical_scalar_bytes: Vec<u8>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyInputPortTranslation {
    target: LegacyPortTranslation,
    source: LegacyInputSourceProjection,
}

impl LegacyInputPortTranslation {
    pub fn target(&self) -> &LegacyPortTranslation {
        &self.target
    }

    pub fn source(&self) -> &LegacyInputSourceProjection {
        &self.source
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyOutputPortTranslation {
    target: LegacyPortTranslation,
    legacy_output_index: u32,
}

impl LegacyOutputPortTranslation {
    pub fn target(&self) -> &LegacyPortTranslation {
        &self.target
    }

    pub fn legacy_output_index(&self) -> u32 {
        self.legacy_output_index
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LegacyProviderScope {
    provider: String,
    endpoint: String,
}

impl LegacyProviderScope {
    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LegacyProviderProjection {
    scopes: Vec<LegacyProviderScope>,
}

impl LegacyProviderProjection {
    pub fn scopes(&self) -> &[LegacyProviderScope] {
        &self.scopes
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LegacyCompatibilityProjection {
    ports: Vec<LegacyPortTranslation>,
    inputs: Vec<LegacyInputPortTranslation>,
    outputs: Vec<LegacyOutputPortTranslation>,
    provider: Option<LegacyProviderProjection>,
}

impl LegacyCompatibilityProjection {
    fn from_manifest_node(
        manifest: &PluginManifest,
        node: &PluginNode,
        mapping: &LegacyMapping,
    ) -> Result<Self, LegacyMappingError> {
        let mut direction_positions = BTreeMap::from([
            (PortDirection::Input, 0_u32),
            (PortDirection::Output, 0_u32),
        ]);
        let mut claimed_names = BTreeSet::new();
        let mut ports = Vec::with_capacity(node.ports.len());
        for port in &node.ports {
            let position = direction_positions
                .get_mut(&port.direction)
                .ok_or(LegacyMappingError::InvalidCandidate)?;
            let target_position = *position;
            *position = position
                .checked_add(1)
                .ok_or(LegacyMappingError::InvalidCandidate)?;
            let accepted_legacy_names = std::iter::once(port.id.clone())
                .chain(std::iter::once(port.name.clone()))
                .chain(port.accepted_legacy_names.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            for name in &accepted_legacy_names {
                if !claimed_names.insert((port.direction, name.clone())) {
                    return Err(LegacyMappingError::InvalidCandidate);
                }
            }
            ports.push(LegacyPortTranslation {
                direction: port.direction,
                target_position,
                accepted_legacy_names,
                target_port_id: port.id.clone(),
                target_port_name: port.name.clone(),
                type_id: port.type_id.clone(),
                cardinality: port.cardinality,
                presence: port.presence,
                serialization: port.serialization,
                lazy: port.lazy,
            });
        }

        let widget_positions = mapping
            .legacy_widget_names
            .iter()
            .enumerate()
            .map(|(position, name)| {
                u32::try_from(position)
                    .map(|position| (name.as_str(), position))
                    .map_err(|_| LegacyMappingError::InvalidCandidate)
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let input_ports = ports
            .iter()
            .filter(|port| port.direction == PortDirection::Input)
            .collect::<Vec<_>>();
        let input_translations = mapping
            .input_translations
            .iter()
            .map(|translation| (translation.target_port_id(), translation))
            .collect::<BTreeMap<_, _>>();
        let inputs = input_ports
            .iter()
            .map(|target| {
                let source = match input_translations.get(target.target_port_id.as_str()) {
                    Some(LegacyInputTranslation::Rename {
                        legacy_input_id, ..
                    }) => LegacyInputSourceProjection::LegacyInput {
                        legacy_input_id: legacy_input_id.clone(),
                        legacy_widget_position: widget_positions
                            .get(legacy_input_id.as_str())
                            .copied(),
                    },
                    Some(LegacyInputTranslation::Constant { value, .. }) => {
                        LegacyInputSourceProjection::Constant {
                            canonical_scalar_bytes: value
                                .abi_bytes()
                                .map_err(|_| LegacyMappingError::InvalidCandidate)?,
                        }
                    }
                    None => LegacyInputSourceProjection::LegacyInput {
                        legacy_input_id: target.target_port_id.clone(),
                        legacy_widget_position: widget_positions
                            .get(target.target_port_id.as_str())
                            .copied(),
                    },
                };
                Ok(LegacyInputPortTranslation {
                    target: (*target).clone(),
                    source,
                })
            })
            .collect::<Result<Vec<_>, LegacyMappingError>>()?;
        let output_ports = ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
            .collect::<Vec<_>>();
        let output_translations = mapping
            .output_translations
            .iter()
            .map(|translation| {
                (
                    translation.target_port_index,
                    translation.legacy_output_index,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let outputs = output_ports
            .iter()
            .enumerate()
            .map(|(target_position, target)| {
                let target_position = u32::try_from(target_position)
                    .map_err(|_| LegacyMappingError::InvalidCandidate)?;
                Ok(LegacyOutputPortTranslation {
                    target: (*target).clone(),
                    legacy_output_index: output_translations
                        .get(&target_position)
                        .copied()
                        .unwrap_or(target.target_position),
                })
            })
            .collect::<Result<Vec<_>, LegacyMappingError>>()?;

        let provider = if node.effects == EffectPolicy::Provider {
            let mut scopes = BTreeSet::new();
            for request in manifest
                .capabilities
                .iter()
                .filter(|request| request.kind == CapabilityKind::NetworkProvider)
            {
                if let Capability::ProviderNetwork { provider, endpoint } =
                    Capability::from_plugin_request(request)?
                {
                    scopes.insert(LegacyProviderScope { provider, endpoint });
                }
            }
            if scopes.is_empty() {
                return Err(LegacyMappingError::ProviderNodeWithoutCapability(
                    node.id.clone(),
                ));
            }
            Some(LegacyProviderProjection {
                scopes: scopes.into_iter().collect(),
            })
        } else {
            None
        };
        Ok(Self {
            ports,
            inputs,
            outputs,
            provider,
        })
    }

    pub fn ports(&self) -> &[LegacyPortTranslation] {
        &self.ports
    }

    pub fn provider(&self) -> Option<&LegacyProviderProjection> {
        self.provider.as_ref()
    }

    pub fn inputs(&self) -> &[LegacyInputPortTranslation] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[LegacyOutputPortTranslation] {
        &self.outputs
    }

    pub fn port_by_name(
        &self,
        direction: PortDirection,
        legacy_name: &str,
    ) -> Option<&LegacyPortTranslation> {
        self.ports.iter().find(|translation| {
            translation.direction == direction
                && translation
                    .accepted_legacy_names
                    .iter()
                    .any(|name| name == legacy_name)
        })
    }

    pub fn port_by_target_position(
        &self,
        direction: PortDirection,
        target_position: u32,
    ) -> Option<&LegacyPortTranslation> {
        self.ports.iter().find(|translation| {
            translation.direction == direction && translation.target_position == target_position
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappingProvenance {
    pub source: MappingSource,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LegacyResolution {
    Projected {
        original: LegacyNodeReference,
        target: MappingTarget,
        compatibility: LegacyCompatibilityProjection,
        provenance: MappingProvenance,
        rewrite_accepted: bool,
    },
    Placeholder {
        original: LegacyNodeReference,
        reason: String,
        choices: Vec<MappingTarget>,
    },
}

impl LegacyResolution {
    pub fn original(&self) -> &LegacyNodeReference {
        match self {
            Self::Projected { original, .. } | Self::Placeholder { original, .. } => original,
        }
    }

    pub fn accept_rewrite(&mut self) -> Option<AcceptedRewrite> {
        match self {
            Self::Projected {
                original,
                target,
                compatibility,
                provenance,
                rewrite_accepted,
            } => {
                *rewrite_accepted = true;
                Some(AcceptedRewrite {
                    legacy_identifier: original.legacy_identifier.clone(),
                    target: target.clone(),
                    compatibility: compatibility.clone(),
                    provenance: provenance.clone(),
                })
            }
            Self::Placeholder { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedRewrite {
    pub legacy_identifier: String,
    pub target: MappingTarget,
    pub compatibility: LegacyCompatibilityProjection,
    pub provenance: MappingProvenance,
}

#[derive(Clone, Debug, Default)]
pub struct InstalledMappingProjection {
    mappings: BTreeMap<String, Vec<MappingCandidate>>,
}

impl InstalledMappingProjection {
    pub fn from_component_host(component_host: &ComponentHost) -> Result<Self, LegacyMappingError> {
        let mut projection = Self::default();
        for plugin in component_host
            .installed_plugins()
            .map_err(|error| LegacyMappingError::ComponentHost(error.to_string()))?
        {
            for mapping in &plugin.manifest().legacy_mappings {
                let candidate = MappingCandidate::new(
                    MappingTarget {
                        plugin_identifier: plugin.manifest().identifier.clone(),
                        node_identifier: mapping.node_id.clone(),
                        node_version: mapping.node_version,
                    },
                    format!(
                        "installed extension {} {}",
                        plugin.extension_id(),
                        plugin.extension_version()
                    ),
                    plugin.manifest(),
                    plugin.authorization(),
                )?;
                projection.insert(mapping.legacy_identifier.clone(), candidate)?;
            }
        }
        Ok(projection)
    }

    fn insert(
        &mut self,
        legacy: String,
        candidate: MappingCandidate,
    ) -> Result<(), LegacyMappingError> {
        validate_legacy_identifier(&legacy)?;
        if !candidate.declares_legacy_identifier(&legacy) {
            return Err(LegacyMappingError::UndeclaredLegacyMapping(legacy));
        }
        self.mappings.entry(legacy).or_default().push(candidate);
        Ok(())
    }

    #[cfg(test)]
    fn from_test_candidates(
        candidates: impl IntoIterator<Item = (String, MappingCandidate)>,
    ) -> Result<Self, LegacyMappingError> {
        let mut projection = Self::default();
        for (legacy, candidate) in candidates {
            projection.insert(legacy, candidate)?;
        }
        Ok(projection)
    }
}

#[derive(Clone, Debug, Default)]
pub struct LegacyMappingResolver {
    workflow_pins: BTreeMap<String, MappingCandidate>,
    user_choices: BTreeMap<String, MappingCandidate>,
    signed_registry: BTreeMap<String, Vec<MappingCandidate>>,
    installed: InstalledMappingProjection,
}

impl LegacyMappingResolver {
    pub fn set_workflow_pin(
        &mut self,
        legacy: impl Into<String>,
        candidate: MappingCandidate,
    ) -> Result<(), LegacyMappingError> {
        insert_selected_candidate(&mut self.workflow_pins, legacy.into(), candidate)
    }

    pub fn set_user_choice(
        &mut self,
        legacy: impl Into<String>,
        candidate: MappingCandidate,
    ) -> Result<(), LegacyMappingError> {
        insert_selected_candidate(&mut self.user_choices, legacy.into(), candidate)
    }

    pub fn add_signed_registry(
        &mut self,
        legacy: impl Into<String>,
        candidate: MappingCandidate,
    ) -> Result<(), LegacyMappingError> {
        let legacy = legacy.into();
        validate_legacy_identifier(&legacy)?;
        if !candidate.declares_legacy_identifier(&legacy) {
            return Err(LegacyMappingError::UndeclaredLegacyMapping(legacy));
        }
        self.signed_registry
            .entry(legacy)
            .or_default()
            .push(candidate);
        Ok(())
    }

    pub fn replace_installed_projection(&mut self, installed: InstalledMappingProjection) {
        self.installed = installed;
    }

    pub fn resolve(
        &self,
        reference: &LegacyNodeReference,
    ) -> Result<LegacyResolution, LegacyMappingError> {
        reference.validate()?;
        let legacy = reference.legacy_identifier.as_str();
        if let Some(candidate) = self.workflow_pins.get(legacy) {
            return projection(reference, candidate, MappingSource::WorkflowPin);
        }
        if let Some(candidate) = self.user_choices.get(legacy) {
            return projection(reference, candidate, MappingSource::UserChoice);
        }

        let registry_candidates = unique_candidates(
            self.signed_registry.get(legacy).into_iter().flatten(),
            legacy,
        );
        if registry_candidates.len() == 1 {
            return projection(
                reference,
                registry_candidates[0],
                MappingSource::SignedRegistry,
            );
        }

        let installed_candidates = unique_candidates(
            self.installed.mappings.get(legacy).into_iter().flatten(),
            legacy,
        );
        if installed_candidates.len() == 1 {
            return projection(
                reference,
                installed_candidates[0],
                MappingSource::UniqueInstalled,
            );
        }

        let choices: Vec<MappingTarget> = registry_candidates
            .into_iter()
            .chain(installed_candidates)
            .map(|candidate| candidate.target().clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let reason = if choices.is_empty() {
            format!("no Rust/WASM mapping is available for `{legacy}`")
        } else {
            format!("mapping for `{legacy}` is ambiguous and requires an explicit choice")
        };
        Ok(LegacyResolution::Placeholder {
            original: reference.clone(),
            reason,
            choices,
        })
    }
}

fn insert_selected_candidate(
    candidates: &mut BTreeMap<String, MappingCandidate>,
    legacy: String,
    candidate: MappingCandidate,
) -> Result<(), LegacyMappingError> {
    validate_legacy_identifier(&legacy)?;
    if !candidate.declares_legacy_identifier(&legacy) {
        return Err(LegacyMappingError::UndeclaredLegacyMapping(legacy));
    }
    candidates.insert(legacy, candidate);
    Ok(())
}

fn validate_legacy_identifier(legacy: &str) -> Result<(), LegacyMappingError> {
    if legacy.is_empty() || legacy.len() > 512 || legacy.chars().any(char::is_control) {
        return Err(LegacyMappingError::InvalidReference);
    }
    Ok(())
}

fn unique_candidates<'a>(
    candidates: impl Iterator<Item = &'a MappingCandidate>,
    legacy_identifier: &str,
) -> Vec<&'a MappingCandidate> {
    let mut unique = Vec::new();
    for candidate in candidates {
        let duplicate = unique.iter().any(|existing: &&MappingCandidate| {
            existing.target() == candidate.target()
                && existing.provenance() == candidate.provenance()
                && existing.compatibility_for(legacy_identifier)
                    == candidate.compatibility_for(legacy_identifier)
        });
        if !duplicate {
            unique.push(candidate);
        }
    }
    unique
}

fn projection(
    reference: &LegacyNodeReference,
    candidate: &MappingCandidate,
    source: MappingSource,
) -> Result<LegacyResolution, LegacyMappingError> {
    let compatibility = candidate
        .compatibility_for(reference.legacy_identifier())
        .ok_or_else(|| {
            LegacyMappingError::UndeclaredLegacyMapping(reference.legacy_identifier().to_owned())
        })?;
    Ok(LegacyResolution::Projected {
        original: reference.clone(),
        target: candidate.target().clone(),
        compatibility: compatibility.clone(),
        provenance: MappingProvenance {
            source,
            detail: candidate.provenance().to_owned(),
        },
        rewrite_accepted: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    fn reference() -> Result<LegacyNodeReference, LegacyMappingError> {
        LegacyNodeReference::new(
            "LegacyNode",
            b"fields".to_vec(),
            b"widgets".to_vec(),
            b"links".to_vec(),
            b"extension".to_vec(),
        )
    }

    fn candidate(node: &str) -> MappingCandidate {
        MappingCandidate::new_for_test(
            MappingTarget {
                plugin_identifier: "example.plugin".to_owned(),
                node_identifier: node.to_owned(),
                node_version: ApiVersion::new(1, 0, 0),
            },
            "test registry",
        )
    }

    #[test]
    fn precedence_and_explicit_rewrite_are_deterministic() -> Result<(), Box<dyn Error>> {
        let mut resolver = LegacyMappingResolver::default();
        resolver.replace_installed_projection(InstalledMappingProjection::from_test_candidates([
            ("LegacyNode".to_owned(), candidate("installed")),
        ])?);
        resolver.add_signed_registry("LegacyNode", candidate("registry"))?;
        resolver.set_user_choice("LegacyNode", candidate("choice"))?;
        resolver.set_workflow_pin("LegacyNode", candidate("pinned"))?;

        let mut resolution = resolver.resolve(&reference()?)?;
        let LegacyResolution::Projected {
            target,
            rewrite_accepted,
            original,
            ..
        } = &resolution
        else {
            return Err("expected a projected mapping".into());
        };
        assert_eq!(target.node_identifier, "pinned");
        assert!(!rewrite_accepted);
        assert_eq!(original.extension_data(), b"extension");
        let rewrite = resolution
            .accept_rewrite()
            .ok_or("projection could not be accepted")?;
        assert_eq!(rewrite.target.node_identifier, "pinned");
        Ok(())
    }

    #[test]
    fn ambiguous_mapping_preserves_exact_placeholder_data() -> Result<(), Box<dyn Error>> {
        let mut resolver = LegacyMappingResolver::default();
        resolver.replace_installed_projection(InstalledMappingProjection::from_test_candidates([
            ("LegacyNode".to_owned(), candidate("one")),
            ("LegacyNode".to_owned(), candidate("two")),
        ])?);
        let reference = reference()?;
        let resolution = resolver.resolve(&reference)?;
        let LegacyResolution::Placeholder {
            original, choices, ..
        } = resolution
        else {
            return Err("expected placeholder".into());
        };
        assert_eq!(original, reference);
        assert_eq!(choices.len(), 2);
        Ok(())
    }

    #[test]
    fn registry_and_installation_cannot_invent_legacy_keys() {
        let mut resolver = LegacyMappingResolver::default();
        assert!(matches!(
            resolver.add_signed_registry("InventedLegacy", candidate("registry")),
            Err(LegacyMappingError::UndeclaredLegacyMapping(identifier))
                if identifier == "InventedLegacy"
        ));
        assert!(matches!(
            InstalledMappingProjection::from_test_candidates([(
                "InventedLegacy".to_owned(),
                candidate("installed"),
            )]),
            Err(LegacyMappingError::UndeclaredLegacyMapping(identifier))
                if identifier == "InventedLegacy"
        ));
        assert!(matches!(
            resolver.set_workflow_pin("InventedLegacy", candidate("pinned")),
            Err(LegacyMappingError::UndeclaredLegacyMapping(identifier))
                if identifier == "InventedLegacy"
        ));
        assert!(matches!(
            resolver.set_user_choice("InventedLegacy", candidate("choice")),
            Err(LegacyMappingError::UndeclaredLegacyMapping(identifier))
                if identifier == "InventedLegacy"
        ));
    }

    #[test]
    fn conflicting_registry_provenance_for_one_target_is_ambiguous() -> Result<(), Box<dyn Error>> {
        let mut resolver = LegacyMappingResolver::default();
        resolver.add_signed_registry("LegacyNode", candidate("same-target"))?;
        let mut conflicting = candidate("same-target");
        conflicting.provenance = "different signed registry record".to_owned();
        resolver.add_signed_registry("LegacyNode", conflicting)?;

        let LegacyResolution::Placeholder {
            original,
            choices,
            reason,
        } = resolver.resolve(&reference()?)?
        else {
            return Err("conflicting signed candidates were silently collapsed".into());
        };
        assert_eq!(original, reference()?);
        assert_eq!(choices.len(), 1);
        assert!(reason.contains("ambiguous"));
        Ok(())
    }

    #[test]
    fn references_are_bounded_before_resolution() {
        assert!(matches!(
            LegacyNodeReference::new(
                "LegacyNode",
                vec![0; MAX_LEGACY_REFERENCE_BYTES],
                vec![1],
                Vec::new(),
                Vec::new(),
            ),
            Err(LegacyMappingError::ReferenceTooLarge)
        ));
        assert!(matches!(
            LegacyNodeReference::new(
                "bad\nidentifier",
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new()
            ),
            Err(LegacyMappingError::InvalidReference)
        ));
        let maximum_identifier = "L".repeat(512);
        assert!(
            LegacyNodeReference::new(
                maximum_identifier.clone(),
                vec![0; MAX_LEGACY_REFERENCE_BYTES - maximum_identifier.len()],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .is_ok()
        );
        assert!(matches!(
            LegacyNodeReference::new(
                maximum_identifier.clone(),
                vec![0; MAX_LEGACY_REFERENCE_BYTES - maximum_identifier.len() + 1],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ),
            Err(LegacyMappingError::ReferenceTooLarge)
        ));
    }
}
