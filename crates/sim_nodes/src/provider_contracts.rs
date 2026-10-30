use crate::{CatalogNodeStatus, NativeNodeBinding, NodeRegistry, NodeRegistryError};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
use thiserror::Error;

pub const PROVIDER_COMPONENT_CONTRACT_CATALOG: &[u8] = include_bytes!(
    "../../../crates/sim_test_support/parity/catalogs/provider-component-contracts.json"
);
pub const PROVIDER_COMPONENT_CONTRACT_CATALOG_SHA256: &str =
    "cce8bc726f93d275731b54e706f89ae71831f55c882247bb9a255486ccc3c69e";
pub const PROVIDER_NODE_CONTRACT_COUNT: usize = 224;
pub const PROVIDER_NAMESPACE_COUNT: usize = 33;

const PROVIDER_CATALOG_SCHEMA_VERSION: u16 = 1;
const PROVIDER_NAMESPACE_PREFIX: &str = "zed.sim.provider.";
const MAX_PROVIDER_CATALOG_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderNodeContract {
    feature_id: String,
    node_identifier: String,
    vendor: String,
    implementation_namespace: String,
}

impl ProviderNodeContract {
    pub fn feature_id(&self) -> &str {
        &self.feature_id
    }

    pub fn node_identifier(&self) -> &str {
        &self.node_identifier
    }

    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    pub fn implementation_namespace(&self) -> &str {
        &self.implementation_namespace
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderNamespaceProjection {
    by_feature_id: BTreeMap<String, ProviderNodeContract>,
    feature_id_by_node: BTreeMap<String, String>,
    node_ids_by_namespace: BTreeMap<String, Vec<String>>,
}

impl ProviderNamespaceProjection {
    fn checked_from_catalog_json(catalog: &[u8]) -> Result<Self, ProviderContractError> {
        if catalog.len() > MAX_PROVIDER_CATALOG_BYTES {
            return Err(ProviderContractError::CatalogTooLarge);
        }
        let document: ProviderCatalogWire = serde_json::from_slice(catalog)
            .map_err(|error| ProviderContractError::MalformedCatalog(error.to_string()))?;
        if document.schema_version != PROVIDER_CATALOG_SCHEMA_VERSION {
            return Err(ProviderContractError::UnsupportedSchema(
                document.schema_version,
            ));
        }
        validate_catalog_header(&document)?;

        let mut vendors = BTreeMap::new();
        let mut aliases = BTreeSet::new();
        let mut previous_vendor = None;
        for vendor in &document.vendors {
            if previous_vendor.is_some_and(|previous| previous > vendor.vendor.as_str()) {
                return Err(ProviderContractError::UnsortedVendor(vendor.vendor.clone()));
            }
            previous_vendor = Some(vendor.vendor.as_str());
            validate_vendor_identifier(&vendor.vendor)?;
            let expected_namespace = format!("{PROVIDER_NAMESPACE_PREFIX}{}", vendor.vendor);
            if vendor.namespace != expected_namespace {
                return Err(ProviderContractError::NamespaceMismatch {
                    identity: vendor.vendor.clone(),
                    expected: expected_namespace,
                    actual: vendor.namespace.clone(),
                });
            }
            for alias in &vendor.aliases {
                validate_vendor_identifier(alias)?;
                if !aliases.insert(alias.clone()) || alias == &vendor.vendor {
                    return Err(ProviderContractError::DuplicateAlias(alias.clone()));
                }
            }
            if vendors.insert(vendor.vendor.clone(), vendor).is_some() {
                return Err(ProviderContractError::DuplicateVendor(
                    vendor.vendor.clone(),
                ));
            }
        }
        if let Some(alias) = aliases.iter().find(|alias| vendors.contains_key(*alias)) {
            return Err(ProviderContractError::DuplicateAlias(alias.clone()));
        }

        let mut by_feature_id = BTreeMap::new();
        let mut feature_id_by_node = BTreeMap::new();
        let mut node_ids_by_namespace = BTreeMap::<String, Vec<String>>::new();
        let mut previous_feature_id = None;
        for node in &document.nodes {
            if previous_feature_id.is_some_and(|previous| previous > node.feature_id.as_str()) {
                return Err(ProviderContractError::UnsortedFeature(
                    node.feature_id.clone(),
                ));
            }
            previous_feature_id = Some(node.feature_id.as_str());
            if node.disposition != "provider_required"
                || !valid_feature_id(&node.feature_id)
                || !valid_node_identifier(&node.node_identifier)
            {
                return Err(ProviderContractError::InvalidNodeClaim(
                    node.feature_id.clone(),
                ));
            }
            validate_source(&node.source, &node.node_identifier)?;
            let vendor = vendors
                .get(&node.vendor)
                .ok_or_else(|| ProviderContractError::MissingVendor(node.vendor.clone()))?;
            if node.namespace != vendor.namespace {
                return Err(ProviderContractError::NamespaceMismatch {
                    identity: node.feature_id.clone(),
                    expected: vendor.namespace.clone(),
                    actual: node.namespace.clone(),
                });
            }
            let contract = ProviderNodeContract {
                feature_id: node.feature_id.clone(),
                node_identifier: node.node_identifier.clone(),
                vendor: node.vendor.clone(),
                implementation_namespace: node.namespace.clone(),
            };
            if by_feature_id
                .insert(node.feature_id.clone(), contract)
                .is_some()
            {
                return Err(ProviderContractError::DuplicateFeature(
                    node.feature_id.clone(),
                ));
            }
            if feature_id_by_node
                .insert(node.node_identifier.clone(), node.feature_id.clone())
                .is_some()
            {
                return Err(ProviderContractError::DuplicateNode(
                    node.node_identifier.clone(),
                ));
            }
            node_ids_by_namespace
                .entry(node.namespace.clone())
                .or_default()
                .push(node.node_identifier.clone());
        }

        validate_vendor_claims(&document, &vendors, &by_feature_id)?;
        validate_route_claims(&document, &vendors)?;
        if by_feature_id.len() != PROVIDER_NODE_CONTRACT_COUNT
            || vendors.len() != PROVIDER_NAMESPACE_COUNT
            || node_ids_by_namespace.len() != PROVIDER_NAMESPACE_COUNT
        {
            return Err(ProviderContractError::SummaryMismatch);
        }
        Ok(Self {
            by_feature_id,
            feature_id_by_node,
            node_ids_by_namespace,
        })
    }

    pub fn len(&self) -> usize {
        self.by_feature_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_feature_id.is_empty()
    }

    pub fn namespace_len(&self) -> usize {
        self.node_ids_by_namespace.len()
    }

    pub fn contract_for_feature_id(&self, feature_id: &str) -> Option<&ProviderNodeContract> {
        self.by_feature_id.get(feature_id)
    }

    pub fn contract_for_node(&self, node_identifier: &str) -> Option<&ProviderNodeContract> {
        self.feature_id_by_node
            .get(node_identifier)
            .and_then(|feature_id| self.by_feature_id.get(feature_id))
    }

    pub fn namespace_members(&self, namespace: &str) -> Option<&[String]> {
        self.node_ids_by_namespace.get(namespace).map(Vec::as_slice)
    }

    pub fn namespaces(&self) -> impl ExactSizeIterator<Item = &str> {
        self.node_ids_by_namespace.keys().map(String::as_str)
    }

    fn project_provider_required_bindings(
        &self,
        bindings: &mut [NativeNodeBinding],
    ) -> Result<(), ProviderContractError> {
        let mut projected = Vec::new();
        projected
            .try_reserve_exact(bindings.len())
            .map_err(|error| ProviderContractError::AllocationFailed(error.to_string()))?;
        for binding in bindings.iter().cloned() {
            projected.push(self.project_binding(binding)?);
        }
        bindings.clone_from_slice(&projected);
        Ok(())
    }

    fn project_binding(
        &self,
        binding: NativeNodeBinding,
    ) -> Result<NativeNodeBinding, ProviderContractError> {
        let feature_id = binding.feature_id().to_owned();
        let node_identifier = binding.descriptor().class_type.clone();
        let feature_contract = self.contract_for_feature_id(&feature_id);
        let node_contract = self.contract_for_node(&node_identifier);
        match (feature_contract, node_contract) {
            (None, None) => return Ok(binding),
            (Some(feature), Some(node)) if feature == node => {}
            _ => {
                return Err(ProviderContractError::BindingIdentityMismatch {
                    feature_id,
                    node_identifier,
                });
            }
        }
        let contract =
            feature_contract.ok_or_else(|| ProviderContractError::BindingIdentityMismatch {
                feature_id: feature_id.clone(),
                node_identifier: node_identifier.clone(),
            })?;
        let NativeNodeBinding::ProviderRequired {
            feature_id,
            descriptor,
            presentation,
            provider,
            reason,
        } = binding
        else {
            return Err(ProviderContractError::BindingDispositionMismatch(
                node_identifier,
            ));
        };
        if provider != contract.implementation_namespace {
            return Err(ProviderContractError::BindingNamespaceMismatch {
                node_identifier: contract.node_identifier.clone(),
                expected: contract.implementation_namespace.clone(),
                actual: provider,
            });
        }
        Ok(NativeNodeBinding::ProviderRequired {
            feature_id,
            descriptor,
            presentation,
            provider: contract.implementation_namespace.clone(),
            reason,
        })
    }

    fn validate_registry(&self, registry: &NodeRegistry) -> Result<(), ProviderContractError> {
        let expected = registry
            .registered()
            .values()
            .filter(|descriptor| descriptor.catalog_status == CatalogNodeStatus::ProviderRequired)
            .collect::<Vec<_>>();
        if expected.len() != self.len() {
            return Err(ProviderContractError::RegistryMismatch);
        }
        for descriptor in expected {
            let Some(contract) = self.contract_for_feature_id(&descriptor.feature_id) else {
                return Err(ProviderContractError::RegistryMismatch);
            };
            if contract.node_identifier != descriptor.node_identifier {
                return Err(ProviderContractError::RegistryMismatch);
            }
        }
        Ok(())
    }
}

pub fn authoritative_provider_namespace_projection()
-> Result<&'static ProviderNamespaceProjection, ProviderContractError> {
    static PROJECTION: OnceLock<ProviderNamespaceProjection> = OnceLock::new();
    if let Some(projection) = PROJECTION.get() {
        return Ok(projection);
    }
    let actual_digest = format!("{:x}", Sha256::digest(PROVIDER_COMPONENT_CONTRACT_CATALOG));
    if actual_digest != PROVIDER_COMPONENT_CONTRACT_CATALOG_SHA256 {
        return Err(ProviderContractError::CatalogDigestMismatch {
            expected: PROVIDER_COMPONENT_CONTRACT_CATALOG_SHA256.to_owned(),
            actual: actual_digest,
        });
    }
    let projection = ProviderNamespaceProjection::checked_from_catalog_json(
        PROVIDER_COMPONENT_CONTRACT_CATALOG,
    )?;
    projection.validate_registry(&NodeRegistry::built_in()?)?;
    if PROJECTION.set(projection).is_err() {
        return PROJECTION
            .get()
            .ok_or(ProviderContractError::InitializationRace);
    }
    PROJECTION
        .get()
        .ok_or(ProviderContractError::InitializationRace)
}

pub fn authoritative_provider_namespace(
    feature_id: &str,
    node_identifier: &str,
) -> Result<&'static str, ProviderContractError> {
    let projection = authoritative_provider_namespace_projection()?;
    let contract = projection
        .contract_for_feature_id(feature_id)
        .ok_or_else(|| ProviderContractError::MissingBinding(feature_id.to_owned()))?;
    if contract.node_identifier != node_identifier {
        return Err(ProviderContractError::BindingIdentityMismatch {
            feature_id: feature_id.to_owned(),
            node_identifier: node_identifier.to_owned(),
        });
    }
    Ok(contract.implementation_namespace())
}

pub fn project_authoritative_provider_bindings(
    bindings: &mut [NativeNodeBinding],
) -> Result<(), ProviderContractError> {
    authoritative_provider_namespace_projection()?.project_provider_required_bindings(bindings)
}

pub fn validate_provider_component_catalog(catalog: &[u8]) -> Result<(), ProviderContractError> {
    ProviderNamespaceProjection::checked_from_catalog_json(catalog).map(drop)
}

fn validate_catalog_header(document: &ProviderCatalogWire) -> Result<(), ProviderContractError> {
    if document.classification != "source-fingerprinted provider component contract catalog"
        || document.summary.provider_nodes != PROVIDER_NODE_CONTRACT_COUNT
        || document.summary.vendors != PROVIDER_NAMESPACE_COUNT
        || document.summary.route_rows != 226
        || document.summary.resolved_unknown_methods != 61
        || document.summary.synthetic_prefix_tombstones != 40
        || document.summary.unknown_methods != 0
        || document.nodes.len() != document.summary.provider_nodes
        || document.vendors.len() != document.summary.vendors
        || document.routes.len() != document.summary.route_rows
        || document.source_snapshot.files != 78
        || document.source_snapshot.root != "projects/comfy/ComfyUI/comfy_api_nodes"
        || !valid_sha256(&document.source_snapshot.tree_sha256)
        || !valid_sha256(&document.input.backend_external_services_sha256)
        || !valid_sha256(&document.input.backend_node_contracts_sha256)
    {
        return Err(ProviderContractError::SummaryMismatch);
    }
    Ok(())
}

fn validate_vendor_claims(
    document: &ProviderCatalogWire,
    vendors: &BTreeMap<String, &ProviderVendorWire>,
    by_feature_id: &BTreeMap<String, ProviderNodeContract>,
) -> Result<(), ProviderContractError> {
    for (vendor_name, vendor) in vendors {
        if !vendor
            .node_feature_ids
            .windows(2)
            .all(|pair| pair[0] < pair[1])
            || !vendor
                .route_feature_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        {
            return Err(ProviderContractError::UnsortedVendorClaim(
                vendor_name.clone(),
            ));
        }
        let expected_nodes = by_feature_id
            .values()
            .filter(|contract| contract.vendor == *vendor_name)
            .map(|contract| contract.feature_id.clone())
            .collect::<Vec<_>>();
        let expected_routes = document
            .routes
            .iter()
            .filter(|route| route.vendor == *vendor_name)
            .map(|route| route.feature_id.clone())
            .collect::<Vec<_>>();
        if vendor.node_feature_ids != expected_nodes || vendor.route_feature_ids != expected_routes
        {
            return Err(ProviderContractError::VendorClaimMismatch(
                vendor_name.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_route_claims(
    document: &ProviderCatalogWire,
    vendors: &BTreeMap<String, &ProviderVendorWire>,
) -> Result<(), ProviderContractError> {
    let mut feature_ids = BTreeSet::new();
    let mut resolved_unknown_methods = 0usize;
    let mut synthetic_prefix_tombstones = 0usize;
    for route in &document.routes {
        if !feature_ids.insert(route.feature_id.clone())
            || !matches!(
                route.disposition.as_str(),
                "executable" | "synthetic_prefix_tombstone"
            )
            || !matches!(route.method.as_str(), "GET" | "PATCH" | "POST")
            || !matches!(
                route.original_method.as_str(),
                "GET" | "PATCH" | "POST" | "UNKNOWN"
            )
            || route.path.is_empty()
            || route.provider.is_empty()
        {
            return Err(ProviderContractError::InvalidRouteClaim(
                route.feature_id.clone(),
            ));
        }
        validate_source(&route.source, &route.feature_id)?;
        let vendor = vendors
            .get(&route.vendor)
            .ok_or_else(|| ProviderContractError::MissingVendor(route.vendor.clone()))?;
        if route.namespace != vendor.namespace {
            return Err(ProviderContractError::NamespaceMismatch {
                identity: route.feature_id.clone(),
                expected: vendor.namespace.clone(),
                actual: route.namespace.clone(),
            });
        }
        if route.original_method == "UNKNOWN" {
            resolved_unknown_methods = resolved_unknown_methods
                .checked_add(1)
                .ok_or(ProviderContractError::SummaryMismatch)?;
        }
        if route.disposition == "synthetic_prefix_tombstone" {
            synthetic_prefix_tombstones = synthetic_prefix_tombstones
                .checked_add(1)
                .ok_or(ProviderContractError::SummaryMismatch)?;
        }
    }
    if resolved_unknown_methods != document.summary.resolved_unknown_methods
        || synthetic_prefix_tombstones != document.summary.synthetic_prefix_tombstones
    {
        return Err(ProviderContractError::SummaryMismatch);
    }
    Ok(())
}

fn validate_source(
    source: &ProviderSourceWire,
    identity: &str,
) -> Result<(), ProviderContractError> {
    if source.line == 0
        || source.path.is_empty()
        || source.symbol.is_empty()
        || !valid_sha256(&source.sha256)
    {
        return Err(ProviderContractError::InvalidSource(identity.to_owned()));
    }
    Ok(())
}

fn valid_feature_id(value: &str) -> bool {
    value
        .strip_prefix("COMFY-NODE-")
        .is_some_and(|suffix| suffix.len() == 4 && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

fn valid_node_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && !value.contains('\0')
        && !value.chars().any(char::is_control)
}

fn validate_vendor_identifier(value: &str) -> Result<(), ProviderContractError> {
    if value.is_empty()
        || value.len() > 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
    {
        return Err(ProviderContractError::InvalidVendor(value.to_owned()));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ProviderContractError {
    #[error("provider component catalog exceeds its byte limit")]
    CatalogTooLarge,
    #[error("provider component catalog is malformed: {0}")]
    MalformedCatalog(String),
    #[error("provider component catalog schema {0} is unsupported")]
    UnsupportedSchema(u16),
    #[error("provider component catalog digest differs: expected {expected}, got {actual}")]
    CatalogDigestMismatch { expected: String, actual: String },
    #[error("provider component catalog summary or source identity is invalid")]
    SummaryMismatch,
    #[error("provider vendor `{0}` is invalid")]
    InvalidVendor(String),
    #[error("provider vendor `{0}` is duplicated")]
    DuplicateVendor(String),
    #[error("provider vendor `{0}` is not sorted")]
    UnsortedVendor(String),
    #[error("provider alias `{0}` is duplicated")]
    DuplicateAlias(String),
    #[error("provider vendor `{0}` is missing")]
    MissingVendor(String),
    #[error("provider feature `{0}` is duplicated")]
    DuplicateFeature(String),
    #[error("provider feature `{0}` is not sorted")]
    UnsortedFeature(String),
    #[error("provider node `{0}` is duplicated")]
    DuplicateNode(String),
    #[error("provider node claim `{0}` is invalid")]
    InvalidNodeClaim(String),
    #[error("provider route claim `{0}` is invalid")]
    InvalidRouteClaim(String),
    #[error("provider source claim `{0}` is invalid")]
    InvalidSource(String),
    #[error("provider namespace for `{identity}` differs: expected `{expected}`, got `{actual}`")]
    NamespaceMismatch {
        identity: String,
        expected: String,
        actual: String,
    },
    #[error("provider claims for vendor `{0}` are not sorted")]
    UnsortedVendorClaim(String),
    #[error("provider claims for vendor `{0}` are incomplete or stale")]
    VendorClaimMismatch(String),
    #[error("provider catalog and registered-node catalog differ")]
    RegistryMismatch,
    #[error("provider binding `{0}` is absent from the authoritative projection")]
    MissingBinding(String),
    #[error("provider binding feature `{feature_id}` and node `{node_identifier}` disagree")]
    BindingIdentityMismatch {
        feature_id: String,
        node_identifier: String,
    },
    #[error("provider binding `{0}` has the wrong disposition")]
    BindingDispositionMismatch(String),
    #[error(
        "provider binding `{node_identifier}` namespace differs: expected `{expected}`, got `{actual}`"
    )]
    BindingNamespaceMismatch {
        node_identifier: String,
        expected: String,
        actual: String,
    },
    #[error("provider projection allocation failed: {0}")]
    AllocationFailed(String),
    #[error("provider projection initialization raced without a result")]
    InitializationRace,
    #[error(transparent)]
    NodeRegistry(#[from] NodeRegistryError),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderCatalogWire {
    classification: String,
    input: ProviderInputWire,
    nodes: Vec<ProviderNodeWire>,
    routes: Vec<ProviderRouteWire>,
    schema_version: u16,
    source_snapshot: ProviderSourceSnapshotWire,
    summary: ProviderSummaryWire,
    vendors: Vec<ProviderVendorWire>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderInputWire {
    backend_external_services_sha256: String,
    backend_node_contracts_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderNodeWire {
    disposition: String,
    feature_id: String,
    namespace: String,
    node_identifier: String,
    source: ProviderSourceWire,
    vendor: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderRouteWire {
    disposition: String,
    feature_id: String,
    method: String,
    namespace: String,
    original_method: String,
    path: String,
    provider: String,
    source: ProviderSourceWire,
    vendor: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderSourceWire {
    line: usize,
    path: String,
    sha256: String,
    symbol: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderSourceSnapshotWire {
    files: usize,
    root: String,
    tree_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderSummaryWire {
    provider_nodes: usize,
    resolved_unknown_methods: usize,
    route_rows: usize,
    synthetic_prefix_tombstones: usize,
    unknown_methods: usize,
    vendors: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderVendorWire {
    aliases: Vec<String>,
    namespace: String,
    node_feature_ids: Vec<String>,
    route_feature_ids: Vec<String>,
    vendor: String,
}
