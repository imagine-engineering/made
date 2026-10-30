use crate::{
    CatalogNodeDescriptor, CatalogNodeSchemaMetadata, CatalogNodeSource, CatalogNodeStatus,
    NODE_DESCRIPTOR_SCHEMA_VERSION, NativeNodeBinding, NativeNodeBindingDisposition,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    sync::{Arc, OnceLock},
};

pub const REGISTERED_NODE_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/backend-nodes.csv");
pub const INACTIVE_NODE_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/backend-inactive-nodes.csv");
pub const NODE_CONTRACT_CATALOG: &[u8] =
    include_bytes!("../../../crates/sim_test_support/parity/catalogs/backend-node-contracts.json");

const MAX_CATALOG_BYTES: usize = 16 * 1024 * 1024;
const MAX_CATALOG_ROWS: usize = 100_000;
const MAX_COLUMNS: usize = 128;
const MAX_FIELD_BYTES: usize = 2 * 1024 * 1024;
const REGISTERED_HEADER: &[&str] = &[
    "node_identifier",
    "class_name",
    "display_name",
    "category",
    "product",
    "classification",
    "availability",
    "evidence_level",
    "confidence",
    "schema_api",
    "schema_source",
    "inputs",
    "outputs",
    "input_is_list",
    "output_is_list",
    "lazy_inputs",
    "output_node",
    "execution_function",
    "validation",
    "caching",
    "change_detection",
    "execution_blocking",
    "error_behavior",
    "source_file",
    "source_symbol",
    "source_line",
    "test_evidence",
    "registration_evidence",
    "feature_id",
];

const INACTIVE_HEADER: &[&str] = &[
    "node_identifier",
    "class_name",
    "display_name",
    "category",
    "classification",
    "availability",
    "evidence_level",
    "confidence",
    "reason",
    "inputs",
    "outputs",
    "source_file",
    "source_symbol",
    "source_line",
    "test_evidence",
    "zed_status",
    "parity_gap",
    "feature_id",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NodeRegistryError {
    CatalogTooLarge,
    TooManyRows,
    TooManyColumns {
        row: usize,
    },
    FieldTooLarge {
        row: usize,
        column: usize,
    },
    MalformedCsv {
        position: usize,
        reason: String,
    },
    HeaderMismatch,
    ColumnCount {
        row: usize,
        expected: usize,
        actual: usize,
    },
    InvalidBoolean {
        row: usize,
        field: &'static str,
    },
    InvalidNumber {
        row: usize,
        field: &'static str,
    },
    InvalidDescriptor {
        row: usize,
        field: &'static str,
    },
    MissingSourceProjection(String),
    DuplicateNode(String),
    DuplicateFeature(String),
    InvalidNativeBinding {
        identifier: String,
        reason: String,
    },
}

impl fmt::Display for NodeRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CatalogTooLarge => formatter.write_str("node catalog exceeds its byte limit"),
            Self::TooManyRows => formatter.write_str("node catalog exceeds its row limit"),
            Self::TooManyColumns { row } => {
                write!(formatter, "node catalog row {row} exceeds its column limit")
            }
            Self::FieldTooLarge { row, column } => write!(
                formatter,
                "node catalog field at row {row}, column {column} exceeds its byte limit"
            ),
            Self::MalformedCsv { position, reason } => {
                write!(
                    formatter,
                    "malformed node catalog CSV at {position}: {reason}"
                )
            }
            Self::HeaderMismatch => formatter.write_str("node catalog header does not match"),
            Self::ColumnCount {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "node catalog row {row} has {actual} columns, expected {expected}"
            ),
            Self::InvalidBoolean { row, field } => {
                write!(
                    formatter,
                    "node catalog row {row} has invalid boolean `{field}`"
                )
            }
            Self::InvalidNumber { row, field } => {
                write!(
                    formatter,
                    "node catalog row {row} has invalid number `{field}`"
                )
            }
            Self::InvalidDescriptor { row, field } => {
                write!(formatter, "node catalog row {row} has invalid `{field}`")
            }
            Self::MissingSourceProjection(identifier) => write!(
                formatter,
                "node catalog descriptor `{identifier}` has no checked source module projection"
            ),
            Self::DuplicateNode(identifier) => {
                write!(formatter, "duplicate node identifier `{identifier}`")
            }
            Self::DuplicateFeature(identifier) => {
                write!(formatter, "duplicate node feature ID `{identifier}`")
            }
            Self::InvalidNativeBinding { identifier, reason } => {
                write!(
                    formatter,
                    "native node binding for `{identifier}` is invalid: {reason}"
                )
            }
        }
    }
}

impl Error for NodeRegistryError {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeContractCatalogWire {
    classification: String,
    contracts: Vec<NodeContractWire>,
    input: NodeContractInputWire,
    schema_version: u16,
    source_snapshot: NodeContractSourceSnapshotWire,
    summary: NodeContractSummaryWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeContractInputWire {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeContractSourceSnapshotWire {
    files: usize,
    manifest_sha256: String,
    root: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeContractSummaryWire {
    executable: usize,
    normalized_v1: usize,
    normalized_v3: usize,
    preserved_schema_contracts: usize,
    provider_required: usize,
    rows: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeContractWire {
    availability: String,
    binding_disposition: String,
    capability_hints: NodeContractCapabilitiesWire,
    category: String,
    classification: String,
    feature_id: String,
    input_is_list: String,
    lazy_inputs: String,
    node_identifier: String,
    output_is_list: String,
    output_node: bool,
    schema: NodeContractSchemaWire,
    schema_api: String,
    source: NodeContractSourceWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeContractCapabilitiesWire {
    asset_or_effect: bool,
    provider: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeContractSchemaWire {
    catalog_correlation: String,
    catalog_sha256: String,
    catalog_source: String,
    #[serde(rename = "contract")]
    _contract: serde::de::IgnoredAny,
    definition_sha256: String,
    method: String,
    portable: CatalogNodeSchemaMetadata,
    status: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeContractSourceWire {
    catalog_line: Option<u32>,
    path: String,
    sha256: String,
    symbol: NodeContractSymbolWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeContractSymbolWire {
    end_line: u32,
    kind: String,
    line: u32,
    sha256: String,
    status: String,
    symbol: String,
}

fn built_in_contract_schemas()
-> Result<Arc<BTreeMap<String, CatalogNodeSchemaMetadata>>, NodeRegistryError> {
    static SCHEMAS: OnceLock<
        Result<Arc<BTreeMap<String, CatalogNodeSchemaMetadata>>, NodeRegistryError>,
    > = OnceLock::new();
    SCHEMAS.get_or_init(parse_built_in_contract_schemas).clone()
}

pub fn built_in_source_schema(
    identifier: &str,
) -> Result<CatalogNodeSchemaMetadata, NodeRegistryError> {
    built_in_contract_schemas()?
        .get(identifier)
        .cloned()
        .ok_or_else(|| NodeRegistryError::MissingSourceProjection(identifier.to_owned()))
}

fn parse_built_in_contract_schemas()
-> Result<Arc<BTreeMap<String, CatalogNodeSchemaMetadata>>, NodeRegistryError> {
    let mut schemas = parse_contract_schemas(NODE_CONTRACT_CATALOG)?;
    let inherited_outputs = schemas
        .get("ModelSamplingSD3")
        .ok_or_else(|| NodeRegistryError::MissingSourceProjection("ModelSamplingSD3".to_owned()))?
        .outputs
        .clone();
    let alias = Arc::make_mut(&mut schemas)
        .get_mut("ModelSamplingAuraFlow")
        .ok_or_else(|| {
            NodeRegistryError::MissingSourceProjection("ModelSamplingAuraFlow".to_owned())
        })?;
    if alias.outputs.is_empty() {
        alias.outputs = inherited_outputs;
    }
    Ok(schemas)
}

fn parse_contract_schemas(
    bytes: &[u8],
) -> Result<Arc<BTreeMap<String, CatalogNodeSchemaMetadata>>, NodeRegistryError> {
    if bytes.len() > MAX_CATALOG_BYTES {
        return Err(NodeRegistryError::CatalogTooLarge);
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let catalog = NodeContractCatalogWire::deserialize(&mut deserializer).map_err(|error| {
        NodeRegistryError::InvalidNativeBinding {
            identifier: "backend-node-contracts.json".to_owned(),
            reason: error.to_string(),
        }
    })?;
    deserializer
        .end()
        .map_err(|error| NodeRegistryError::InvalidNativeBinding {
            identifier: "backend-node-contracts.json".to_owned(),
            reason: error.to_string(),
        })?;
    if catalog.schema_version != 2
        || catalog.classification.is_empty()
        || catalog.input.path != "catalogs/backend-nodes.csv"
        || catalog.input.sha256 != sha256_hex(REGISTERED_NODE_CATALOG.as_bytes())
        || catalog.source_snapshot.root != "projects/comfy/ComfyUI"
        || catalog.source_snapshot.files == 0
        || !valid_sha256(&catalog.source_snapshot.manifest_sha256)
        || catalog.summary.rows != catalog.contracts.len()
        || catalog.summary.rows != 789
        || catalog.summary.executable != 565
        || catalog.summary.provider_required != 224
        || catalog.summary.normalized_v1 != 135
        || catalog.summary.normalized_v3 != 654
        || catalog.summary.preserved_schema_contracts != 0
    {
        return Err(NodeRegistryError::InvalidDescriptor {
            row: 0,
            field: "node_contract_catalog",
        });
    }
    let mut schemas = BTreeMap::new();
    let mut feature_ids = BTreeSet::new();
    for (index, contract) in catalog.contracts.into_iter().enumerate() {
        let row = index + 1;
        validate_contract_wire(row, &contract)?;
        if !feature_ids.insert(contract.feature_id.clone())
            || schemas
                .insert(contract.node_identifier.clone(), contract.schema.portable)
                .is_some()
        {
            return Err(NodeRegistryError::InvalidDescriptor {
                row,
                field: "node_contract_identity",
            });
        }
    }
    Ok(Arc::new(schemas))
}

fn validate_contract_wire(
    row: usize,
    contract: &NodeContractWire,
) -> Result<(), NodeRegistryError> {
    let expected_provider = contract.classification == "API node";
    if contract.binding_disposition
        != if expected_provider {
            "provider_required"
        } else {
            "executable"
        }
        || contract.capability_hints.provider != expected_provider
        || contract.category.is_empty()
        || contract.classification.is_empty()
        || contract.feature_id.is_empty()
        || contract.input_is_list.is_empty()
        || contract.lazy_inputs.is_empty()
        || contract.output_is_list.is_empty()
        || contract.schema.catalog_source.is_empty()
        || contract.schema.catalog_sha256 != sha256_hex(contract.schema.catalog_source.as_bytes())
        || contract.schema.definition_sha256 != contract.schema.portable.definition_sha256
        || contract.schema.catalog_sha256 != contract.schema.portable.catalog_sha256
        || !matches!(
            contract.schema.catalog_correlation.as_str(),
            "direct" | "verified_inherited_method" | "verified_inherited_base"
        )
        || contract.schema.method
            != if contract.schema_api == "V3" {
                "define_schema"
            } else {
                "INPUT_TYPES"
            }
        || contract.schema.status
            != if contract.schema_api == "V3" {
                "normalized_v3"
            } else {
                "normalized_v1"
            }
        || contract.source.catalog_line.is_some_and(|line| line == 0)
        || contract.source.path.is_empty()
        || !valid_sha256(&contract.source.sha256)
        || contract.source.symbol.line == 0
        || contract.source.symbol.end_line < contract.source.symbol.line
        || contract.source.symbol.kind != "ClassDef"
        || !valid_sha256(&contract.source.symbol.sha256)
        || contract.source.symbol.status != "parsed_definition"
        || contract.source.symbol.symbol.is_empty()
        || contract.schema.portable.node.feature_id.as_deref() != Some(&contract.feature_id)
        || (contract.availability == "deprecated/dead"
            && !contract.schema.portable.presentation.is_deprecated)
        || contract.output_node && contract.capability_hints.asset_or_effect == expected_provider
    {
        return Err(NodeRegistryError::InvalidDescriptor {
            row,
            field: "node_contract",
        });
    }
    contract
        .schema
        .portable
        .validate()
        .map_err(|_| NodeRegistryError::InvalidDescriptor {
            row,
            field: "portable_schema",
        })
}

fn sha256_hex(value: &[u8]) -> String {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_length = (value.len() as u64).wrapping_mul(8);
    let mut message = value.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_length.to_be_bytes());

    let mut hash = INITIAL;
    for chunk in message.chunks_exact(64) {
        let mut schedule = [0_u32; 64];
        for (index, bytes) in chunk.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for index in 16..64 {
            let small_zero = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let small_one = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(small_zero)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(small_one);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = hash;
        for index in 0..64 {
            let big_one = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temporary_one = h
                .wrapping_add(big_one)
                .wrapping_add(choose)
                .wrapping_add(ROUND[index])
                .wrapping_add(schedule[index]);
            let big_zero = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary_two = big_zero.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary_one);
            d = c;
            c = b;
            b = a;
            a = temporary_one.wrapping_add(temporary_two);
        }
        hash[0] = hash[0].wrapping_add(a);
        hash[1] = hash[1].wrapping_add(b);
        hash[2] = hash[2].wrapping_add(c);
        hash[3] = hash[3].wrapping_add(d);
        hash[4] = hash[4].wrapping_add(e);
        hash[5] = hash[5].wrapping_add(f);
        hash[6] = hash[6].wrapping_add(g);
        hash[7] = hash[7].wrapping_add(h);
    }
    hash.iter().map(|word| format!("{word:08x}")).collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeRegistry {
    registered: BTreeMap<String, CatalogNodeDescriptor>,
    inactive: BTreeMap<String, CatalogNodeDescriptor>,
    schemas: Arc<BTreeMap<String, CatalogNodeSchemaMetadata>>,
}

impl NodeRegistry {
    pub fn built_in() -> Result<Self, NodeRegistryError> {
        static REGISTRY: OnceLock<Result<NodeRegistry, NodeRegistryError>> = OnceLock::new();
        REGISTRY
            .get_or_init(|| {
                let mut registry = NodeRegistryGenerator::from_catalogs(
                    REGISTERED_NODE_CATALOG,
                    INACTIVE_NODE_CATALOG,
                )?
                .finish();
                registry.schemas = built_in_contract_schemas()?;
                if registry.schemas.len() != registry.registered.len()
                    || registry.registered.iter().any(|(identifier, descriptor)| {
                        registry.schemas.get(identifier).is_none_or(|schema| {
                            schema.node.feature_id.as_deref() != Some(&descriptor.feature_id)
                                || schema.catalog_sha256
                                    != sha256_hex(descriptor.schema_source.as_bytes())
                        })
                    })
                {
                    return Err(NodeRegistryError::InvalidDescriptor {
                        row: 0,
                        field: "node_contract_catalog",
                    });
                }
                for identifier in crate::GENERATED_DESCRIPTOR_IDS {
                    if !registry.registered.contains_key(*identifier) {
                        return Err(NodeRegistryError::InvalidDescriptor {
                            row: 0,
                            field: "generated_descriptor_id",
                        });
                    }
                }
                Ok(registry)
            })
            .clone()
    }

    pub fn registered(&self) -> &BTreeMap<String, CatalogNodeDescriptor> {
        &self.registered
    }

    pub fn inactive(&self) -> &BTreeMap<String, CatalogNodeDescriptor> {
        &self.inactive
    }

    pub fn descriptor(&self, identifier: &str) -> Option<&CatalogNodeDescriptor> {
        self.registered
            .get(identifier)
            .or_else(|| self.inactive.get(identifier))
    }

    pub fn source_schema(&self, identifier: &str) -> Option<&CatalogNodeSchemaMetadata> {
        self.schemas.get(identifier)
    }

    pub fn source_python_module(&self, identifier: &str) -> Option<String> {
        self.descriptor(identifier)
            .and_then(|descriptor| source_python_module(&descriptor.source_file))
    }

    pub fn validate_native_binding(
        &self,
        binding: &NativeNodeBinding,
    ) -> Result<(), NodeRegistryError> {
        let identifier = binding.descriptor().class_type.clone();
        binding
            .validate()
            .map_err(|error| NodeRegistryError::InvalidNativeBinding {
                identifier: identifier.clone(),
                reason: error.to_string(),
            })?;
        let catalog = self.descriptor(&identifier).ok_or_else(|| {
            NodeRegistryError::InvalidNativeBinding {
                identifier: identifier.clone(),
                reason: "catalog descriptor is absent".to_owned(),
            }
        })?;
        let expected_category = match catalog.category.as_str() {
            "(empty root category declared by source)" => "",
            category => category,
        };
        let expected_disposition = match (catalog.source, catalog.catalog_status) {
            (CatalogNodeSource::Inactive, _) => NativeNodeBindingDisposition::Unavailable,
            (CatalogNodeSource::Registered, CatalogNodeStatus::ProviderRequired) => {
                NativeNodeBindingDisposition::ProviderRequired
            }
            (
                CatalogNodeSource::Registered,
                CatalogNodeStatus::DescriptorOnly | CatalogNodeStatus::Inactive,
            ) => NativeNodeBindingDisposition::Executable,
        };
        let source_schema_mismatch = binding
            .descriptor()
            .source_schema
            .as_ref()
            .filter(|schema| {
                matches!(
                    schema.node.provenance,
                    crate::NativeSchemaProvenance::SourceV1
                        | crate::NativeSchemaProvenance::SourceV3
                )
            })
            .is_some_and(|schema| {
                let Some(catalog_schema) = self.source_schema(&identifier) else {
                    return true;
                };
                let catalog_input_names = catalog_schema
                    .inputs
                    .iter()
                    .map(|input| input.schema.name.clone())
                    .collect::<Vec<_>>();
                let Some(expected) = catalog_schema
                    .bind_execution_ports(
                        &catalog_input_names,
                        &schema.dynamic_inputs,
                        &binding
                            .descriptor()
                            .outputs
                            .iter()
                            .map(|output| output.name.clone())
                            .collect::<Vec<_>>(),
                    )
                    .ok()
                else {
                    return true;
                };
                let catalog_input_names = catalog_input_names.into_iter().collect::<BTreeSet<_>>();
                let supplemental_inputs_are_hidden =
                    schema
                        .inputs
                        .iter()
                        .filter(|input| !catalog_input_names.contains(&input.name))
                        .all(|input| {
                            binding.descriptor().inputs.iter().any(|descriptor| {
                                descriptor.hidden && descriptor.name == input.name
                            })
                        });
                let mut comparable = schema.clone();
                comparable
                    .inputs
                    .retain(|input| catalog_input_names.contains(&input.name));
                !supplemental_inputs_are_hidden || expected != comparable
            });
        let mismatch = if binding.feature_id() != catalog.feature_id {
            Some("feature_id")
        } else if binding
            .descriptor()
            .source_schema
            .as_ref()
            .and_then(|schema| schema.node.feature_id.as_deref())
            .is_some_and(|feature_id| feature_id != binding.feature_id())
        {
            Some("source_schema.feature_id")
        } else if source_schema_mismatch {
            Some("source_schema")
        } else if binding.presentation().display_name != catalog.display_name {
            Some("display_name")
        } else if binding.presentation().category != expected_category {
            Some("category")
        } else if binding.descriptor().output_node != catalog.output_node {
            Some("output_node")
        } else if binding.disposition() != expected_disposition {
            Some("disposition")
        } else {
            None
        };
        if let Some(field) = mismatch {
            return Err(NodeRegistryError::InvalidNativeBinding {
                identifier,
                reason: format!("catalog field `{field}` does not match"),
            });
        }
        Ok(())
    }
}

pub struct NodeRegistryGenerator {
    registered: BTreeMap<String, CatalogNodeDescriptor>,
    inactive: BTreeMap<String, CatalogNodeDescriptor>,
    feature_ids: BTreeSet<String>,
}

impl NodeRegistryGenerator {
    pub fn from_catalogs(
        registered_catalog: &str,
        inactive_catalog: &str,
    ) -> Result<Self, NodeRegistryError> {
        let registered_rows = parse_csv(registered_catalog)?;
        validate_header(&registered_rows, REGISTERED_HEADER)?;
        let inactive_rows = parse_csv(inactive_catalog)?;
        validate_header(&inactive_rows, INACTIVE_HEADER)?;
        let mut generator = Self {
            registered: BTreeMap::new(),
            inactive: BTreeMap::new(),
            feature_ids: BTreeSet::new(),
        };
        for (row_index, row) in registered_rows.iter().enumerate().skip(1) {
            generator.insert_catalog_descriptor(parse_registered_row(row_index + 1, row)?)?;
        }
        for (row_index, row) in inactive_rows.iter().enumerate().skip(1) {
            generator.insert_catalog_descriptor(parse_inactive_row(row_index + 1, row)?)?;
        }
        Ok(generator)
    }

    pub fn finish(self) -> NodeRegistry {
        NodeRegistry {
            registered: self.registered,
            inactive: self.inactive,
            schemas: Arc::new(BTreeMap::new()),
        }
    }

    fn insert_catalog_descriptor(
        &mut self,
        descriptor: CatalogNodeDescriptor,
    ) -> Result<(), NodeRegistryError> {
        if self.registered.contains_key(&descriptor.node_identifier)
            || self.inactive.contains_key(&descriptor.node_identifier)
        {
            return Err(NodeRegistryError::DuplicateNode(descriptor.node_identifier));
        }
        if !self.feature_ids.insert(descriptor.feature_id.clone()) {
            return Err(NodeRegistryError::DuplicateFeature(descriptor.feature_id));
        }
        match descriptor.source {
            CatalogNodeSource::Registered => {
                self.registered
                    .insert(descriptor.node_identifier.clone(), descriptor);
            }
            CatalogNodeSource::Inactive => {
                self.inactive
                    .insert(descriptor.node_identifier.clone(), descriptor);
            }
        }
        Ok(())
    }
}

fn parse_registered_row(
    row_number: usize,
    row: &[String],
) -> Result<CatalogNodeDescriptor, NodeRegistryError> {
    validate_column_count(row_number, row, REGISTERED_HEADER.len())?;
    validate_common(row_number, row, 28, "COMFY-NODE-")?;
    validate_source_file(row_number, &row[23])?;
    Ok(CatalogNodeDescriptor {
        schema_version: NODE_DESCRIPTOR_SCHEMA_VERSION,
        source: CatalogNodeSource::Registered,
        node_identifier: row[0].clone(),
        class_name: row[1].clone(),
        display_name: row[2].clone(),
        category: row[3].clone(),
        product: row[4].clone(),
        classification: row[5].clone(),
        availability: row[6].clone(),
        evidence_level: row[7].clone(),
        confidence: row[8].clone(),
        schema_api: Some(row[9].clone()),
        schema_source: row[10].clone(),
        inputs: row[11].clone(),
        outputs: row[12].clone(),
        input_is_list: row[13].clone(),
        output_is_list: row[14].clone(),
        lazy_inputs: row[15].clone(),
        output_node: parse_bool(row_number, "output_node", &row[16])?,
        execution_function: row[17].clone(),
        validation: row[18].clone(),
        caching: row[19].clone(),
        change_detection: row[20].clone(),
        execution_blocking: row[21].clone(),
        error_behavior: row[22].clone(),
        source_file: row[23].clone(),
        source_symbol: row[24].clone(),
        source_line: parse_optional_u32(row_number, "source_line", &row[25])?,
        test_evidence: row[26].clone(),
        registration_evidence: row[27].clone(),
        inactive_reason: None,
        zed_status: None,
        parity_gap: None,
        feature_id: row[28].clone(),
        catalog_status: status_for_registration(&row[5], &row[6], false),
    })
}

fn parse_inactive_row(
    row_number: usize,
    row: &[String],
) -> Result<CatalogNodeDescriptor, NodeRegistryError> {
    validate_column_count(row_number, row, INACTIVE_HEADER.len())?;
    validate_common(row_number, row, 17, "COMFY-NODE-INACTIVE-")?;
    validate_source_file(row_number, &row[11])?;
    Ok(CatalogNodeDescriptor {
        schema_version: NODE_DESCRIPTOR_SCHEMA_VERSION,
        source: CatalogNodeSource::Inactive,
        node_identifier: row[0].clone(),
        class_name: row[1].clone(),
        display_name: row[2].clone(),
        category: row[3].clone(),
        product: "ComfyUI".to_owned(),
        classification: row[4].clone(),
        availability: row[5].clone(),
        evidence_level: row[6].clone(),
        confidence: row[7].clone(),
        schema_api: None,
        schema_source: String::new(),
        inputs: row[9].clone(),
        outputs: row[10].clone(),
        input_is_list: String::new(),
        output_is_list: String::new(),
        lazy_inputs: String::new(),
        output_node: false,
        execution_function: String::new(),
        validation: String::new(),
        caching: String::new(),
        change_detection: String::new(),
        execution_blocking: String::new(),
        error_behavior: String::new(),
        source_file: row[11].clone(),
        source_symbol: row[12].clone(),
        source_line: parse_optional_u32(row_number, "source_line", &row[13])?,
        test_evidence: row[14].clone(),
        registration_evidence: String::new(),
        inactive_reason: Some(row[8].clone()),
        zed_status: Some(row[15].clone()),
        parity_gap: Some(row[16].clone()),
        feature_id: row[17].clone(),
        catalog_status: status_for_registration(&row[4], &row[5], true),
    })
}

fn validate_common(
    row_number: usize,
    row: &[String],
    feature_index: usize,
    feature_prefix: &str,
) -> Result<(), NodeRegistryError> {
    for (index, field) in [
        (0, "node_identifier"),
        (1, "class_name"),
        (2, "display_name"),
        (3, "category"),
    ] {
        if row[index].is_empty() {
            return Err(NodeRegistryError::InvalidDescriptor {
                row: row_number,
                field,
            });
        }
    }
    if !valid_catalog_identifier(&row[0]) {
        return Err(NodeRegistryError::InvalidDescriptor {
            row: row_number,
            field: "node_identifier",
        });
    }
    if !row[feature_index].starts_with(feature_prefix) {
        return Err(NodeRegistryError::InvalidDescriptor {
            row: row_number,
            field: "feature_id",
        });
    }
    Ok(())
}

fn status_for_registration(
    classification: &str,
    availability: &str,
    inactive: bool,
) -> CatalogNodeStatus {
    if inactive {
        CatalogNodeStatus::Inactive
    } else if classification == "API node" {
        CatalogNodeStatus::ProviderRequired
    } else if availability == "deprecated/dead" {
        CatalogNodeStatus::Inactive
    } else {
        CatalogNodeStatus::DescriptorOnly
    }
}

fn parse_bool(row: usize, field: &'static str, value: &str) -> Result<bool, NodeRegistryError> {
    match value {
        "True" => Ok(true),
        "False" => Ok(false),
        _ => Err(NodeRegistryError::InvalidBoolean { row, field }),
    }
}

fn parse_optional_u32(
    row: usize,
    field: &'static str,
    value: &str,
) -> Result<Option<u32>, NodeRegistryError> {
    if value.is_empty() {
        Ok(None)
    } else {
        value
            .parse()
            .map(Some)
            .map_err(|_| NodeRegistryError::InvalidNumber { row, field })
    }
}

fn valid_catalog_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 4_096
        && value.chars().all(|character| !character.is_control())
}

fn validate_source_file(row: usize, value: &str) -> Result<(), NodeRegistryError> {
    if source_python_module(value).is_some() {
        Ok(())
    } else {
        Err(NodeRegistryError::InvalidDescriptor {
            row,
            field: "source_file",
        })
    }
}

fn source_python_module(source_file: &str) -> Option<String> {
    if source_file == "nodes.py" {
        return Some("nodes".to_owned());
    }
    let (parent, filename) = source_file.split_once('/')?;
    if !matches!(parent, "comfy_extras" | "comfy_api_nodes")
        || filename.contains('/')
        || !filename.ends_with(".py")
    {
        return None;
    }
    let stem = filename.strip_suffix(".py")?;
    if stem.is_empty()
        || stem.len() > 4_096
        || !stem
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return None;
    }
    Some(format!("{parent}.{stem}"))
}

fn validate_header(rows: &[Vec<String>], expected: &[&str]) -> Result<(), NodeRegistryError> {
    let Some(header) = rows.first() else {
        return Err(NodeRegistryError::HeaderMismatch);
    };
    if header.len() != expected.len()
        || header
            .iter()
            .zip(expected)
            .any(|(actual, expected)| actual != expected)
    {
        return Err(NodeRegistryError::HeaderMismatch);
    }
    Ok(())
}

fn validate_column_count(
    row_number: usize,
    row: &[String],
    expected: usize,
) -> Result<(), NodeRegistryError> {
    if row.len() == expected {
        Ok(())
    } else {
        Err(NodeRegistryError::ColumnCount {
            row: row_number,
            expected,
            actual: row.len(),
        })
    }
}

fn parse_csv(input: &str) -> Result<Vec<Vec<String>>, NodeRegistryError> {
    if input.len() > MAX_CATALOG_BYTES {
        return Err(NodeRegistryError::CatalogTooLarge);
    }
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut characters = input.char_indices().peekable();
    let mut in_quotes = false;
    let mut quote_closed = false;
    while let Some((position, character)) = characters.next() {
        if in_quotes {
            if character == '"' {
                if characters.peek().is_some_and(|(_, next)| *next == '"') {
                    characters.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                    quote_closed = true;
                }
            } else {
                field.push(character);
            }
        } else {
            match character {
                '"' if field.is_empty() && !quote_closed => in_quotes = true,
                '"' => {
                    return Err(NodeRegistryError::MalformedCsv {
                        position,
                        reason: "quote appeared inside an unquoted field".to_owned(),
                    });
                }
                ',' => {
                    push_field(&mut row, &mut field)?;
                    quote_closed = false;
                }
                '\n' | '\r' => {
                    if character == '\r' && characters.peek().is_some_and(|(_, next)| *next == '\n')
                    {
                        characters.next();
                    }
                    push_field(&mut row, &mut field)?;
                    push_row(&mut rows, &mut row)?;
                    quote_closed = false;
                }
                _ if quote_closed => {
                    return Err(NodeRegistryError::MalformedCsv {
                        position,
                        reason: "characters followed a closing quote".to_owned(),
                    });
                }
                _ => field.push(character),
            }
        }
        if field.len() > MAX_FIELD_BYTES {
            return Err(NodeRegistryError::FieldTooLarge {
                row: rows.len() + 1,
                column: row.len() + 1,
            });
        }
    }
    if in_quotes {
        return Err(NodeRegistryError::MalformedCsv {
            position: input.len(),
            reason: "quoted field was not closed".to_owned(),
        });
    }
    if !field.is_empty() || !row.is_empty() || quote_closed {
        push_field(&mut row, &mut field)?;
        push_row(&mut rows, &mut row)?;
    }
    Ok(rows)
}

fn push_field(row: &mut Vec<String>, field: &mut String) -> Result<(), NodeRegistryError> {
    if row.len() >= MAX_COLUMNS {
        return Err(NodeRegistryError::TooManyColumns { row: 0 });
    }
    row.push(std::mem::take(field));
    Ok(())
}

fn push_row(rows: &mut Vec<Vec<String>>, row: &mut Vec<String>) -> Result<(), NodeRegistryError> {
    if rows.len() >= MAX_CATALOG_ROWS {
        return Err(NodeRegistryError::TooManyRows);
    }
    rows.push(std::mem::take(row));
    Ok(())
}

#[cfg(test)]
fn canonical_csv(rows: &[Vec<String>]) -> String {
    let mut output = String::new();
    for row in rows {
        for (index, field) in row.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            if field
                .chars()
                .any(|character| matches!(character, ',' | '"' | '\n' | '\r'))
            {
                output.push('"');
                for character in field.chars() {
                    if character == '"' {
                        output.push('"');
                    }
                    output.push(character);
                }
                output.push('"');
            } else {
                output.push_str(field);
            }
        }
        output.push('\n');
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CatalogNodeStatus, DIFFUSION_SLICE_NODE_IDS, EarlySliceRegistry, IMAGE_SLICE_NODE_IDS,
        NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCachePolicy, NativeDescriptorSchemaMetadata,
        NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor, NativeNode,
        NativeNodeContext, NativeNodeDescriptor, NativeNodeFailure, NativeNodeOutcome,
        NativeNodePresentation, NativeOutputDescriptor, NativePortCardinality, NativeSchemaError,
        NativeSchemaValue, NativeTypeUnion, NativeValue, NativeValueType, ObjectInfoRegistry,
    };
    use futures::future::BoxFuture;
    use serde::Serialize;
    use std::{fs, path::Path};

    fn registered_catalog_row(descriptor: &CatalogNodeDescriptor) -> Vec<String> {
        vec![
            descriptor.node_identifier.clone(),
            descriptor.class_name.clone(),
            descriptor.display_name.clone(),
            descriptor.category.clone(),
            descriptor.product.clone(),
            descriptor.classification.clone(),
            descriptor.availability.clone(),
            descriptor.evidence_level.clone(),
            descriptor.confidence.clone(),
            descriptor.schema_api.clone().unwrap_or_default(),
            descriptor.schema_source.clone(),
            descriptor.inputs.clone(),
            descriptor.outputs.clone(),
            descriptor.input_is_list.clone(),
            descriptor.output_is_list.clone(),
            descriptor.lazy_inputs.clone(),
            if descriptor.output_node {
                "True".to_owned()
            } else {
                "False".to_owned()
            },
            descriptor.execution_function.clone(),
            descriptor.validation.clone(),
            descriptor.caching.clone(),
            descriptor.change_detection.clone(),
            descriptor.execution_blocking.clone(),
            descriptor.error_behavior.clone(),
            descriptor.source_file.clone(),
            descriptor.source_symbol.clone(),
            descriptor
                .source_line
                .map_or_else(String::new, |line| line.to_string()),
            descriptor.test_evidence.clone(),
            descriptor.registration_evidence.clone(),
            descriptor.feature_id.clone(),
        ]
    }

    fn inactive_catalog_row(descriptor: &CatalogNodeDescriptor) -> Vec<String> {
        vec![
            descriptor.node_identifier.clone(),
            descriptor.class_name.clone(),
            descriptor.display_name.clone(),
            descriptor.category.clone(),
            descriptor.classification.clone(),
            descriptor.availability.clone(),
            descriptor.evidence_level.clone(),
            descriptor.confidence.clone(),
            descriptor.inactive_reason.clone().unwrap_or_default(),
            descriptor.inputs.clone(),
            descriptor.outputs.clone(),
            descriptor.source_file.clone(),
            descriptor.source_symbol.clone(),
            descriptor
                .source_line
                .map_or_else(String::new, |line| line.to_string()),
            descriptor.test_evidence.clone(),
            descriptor.zed_status.clone().unwrap_or_default(),
            descriptor.parity_gap.clone().unwrap_or_default(),
            descriptor.feature_id.clone(),
        ]
    }

    fn reconstruct_catalogs(
        registry: &NodeRegistry,
        registered_rows: &[Vec<String>],
        inactive_rows: &[Vec<String>],
    ) -> Result<(String, String), Box<dyn Error>> {
        let mut reconstructed_registered = vec![
            REGISTERED_HEADER
                .iter()
                .map(|field| (*field).to_owned())
                .collect::<Vec<_>>(),
        ];
        for row in registered_rows.iter().skip(1) {
            let identifier = row.first().ok_or("registered row has no identifier")?;
            let descriptor = registry
                .registered()
                .get(identifier)
                .ok_or("registered descriptor is absent")?;
            let reconstructed = registered_catalog_row(descriptor);
            assert_eq!(&reconstructed, row);
            let encoded = serde_json::to_vec(descriptor)?;
            assert_eq!(
                serde_json::from_slice::<CatalogNodeDescriptor>(&encoded)?,
                *descriptor
            );
            reconstructed_registered.push(reconstructed);
        }

        let mut reconstructed_inactive = vec![
            INACTIVE_HEADER
                .iter()
                .map(|field| (*field).to_owned())
                .collect::<Vec<_>>(),
        ];
        for row in inactive_rows.iter().skip(1) {
            let identifier = row.first().ok_or("inactive row has no identifier")?;
            let descriptor = registry
                .inactive()
                .get(identifier)
                .ok_or("inactive descriptor is absent")?;
            let reconstructed = inactive_catalog_row(descriptor);
            assert_eq!(&reconstructed, row);
            let encoded = serde_json::to_vec(descriptor)?;
            assert_eq!(
                serde_json::from_slice::<CatalogNodeDescriptor>(&encoded)?,
                *descriptor
            );
            reconstructed_inactive.push(reconstructed);
        }
        Ok((
            canonical_csv(&reconstructed_registered),
            canonical_csv(&reconstructed_inactive),
        ))
    }

    fn object_info_matches_descriptor(
        registry: &NodeRegistry,
        object_info: &ObjectInfoRegistry,
        identifier: &str,
        descriptor: &CatalogNodeDescriptor,
    ) -> bool {
        object_info
            .nodes()
            .get(identifier)
            .is_some_and(|projected| {
                projected.schema_version == crate::OBJECT_INFO_SCHEMA_VERSION
                    && projected.node_identifier == descriptor.node_identifier
                    && projected.display_name == descriptor.display_name
                    && projected.category == descriptor.category
                    && projected.source_python_module
                        == registry
                            .source_python_module(identifier)
                            .unwrap_or_default()
                    && projected.schema_source == descriptor.schema_source
                    && projected.source_schema.as_ref() == registry.source_schema(identifier)
                    && projected.input.raw == descriptor.inputs
                    && projected.input.input_is_list == descriptor.input_is_list
                    && projected.input.lazy_inputs == descriptor.lazy_inputs
                    && projected.output.raw == descriptor.outputs
                    && projected.output.output_is_list == descriptor.output_is_list
                    && projected.output.output_node == descriptor.output_node
                    && projected.availability == descriptor.availability
                    && projected.catalog_status == descriptor.catalog_status
                    && projected.inactive_reason == descriptor.inactive_reason
                    && projected.feature_id == descriptor.feature_id
            })
    }

    #[derive(Serialize)]
    struct ValidationEnvironment<'a> {
        operating_system: &'a str,
        architecture: &'a str,
        backend_identity: &'a str,
        registry_role: &'a str,
    }

    #[derive(Serialize)]
    struct ValidationCases {
        catalog_round_trip: bool,
        catalog_status_is_read_only: bool,
        generated_descriptor_membership_is_exact: bool,
        early_slice_membership_is_exact: bool,
        object_info_is_read_only_projection: bool,
        mutable_plugin_and_execution_apis_are_absent: bool,
    }

    #[derive(Serialize)]
    struct ValidationArtifact<'a> {
        validation_id: &'a str,
        registered_rows: usize,
        inactive_rows: usize,
        registered_catalog_sha256: String,
        inactive_catalog_sha256: String,
        generated_descriptor_ids_sha256: String,
        generated_descriptor_ids_encoding: &'a str,
        generated_descriptor_ids: &'a [&'a str],
        image_slice: &'a [&'a str],
        diffusion_slice: &'a [&'a str],
        environment: ValidationEnvironment<'a>,
        cases: ValidationCases,
        passed: usize,
        failed: usize,
        skipped: usize,
    }

    #[derive(Serialize)]
    struct NodeSchemaValidationCase {
        identifier: String,
        feature_id: String,
        source_kind: String,
        binding_disposition: String,
        input_count: usize,
        dynamic_input_count: usize,
        output_count: usize,
        definition_sha256: String,
        source_sha256: Option<String>,
        passed: bool,
    }

    #[derive(Serialize)]
    struct NodeSchemaValidationSummary {
        catalog_csv_round_trip_is_exact: bool,
        registered_contracts_are_exact: bool,
        inactive_descriptors_are_exact: bool,
        object_info_projection_is_exact: bool,
        csv_parser_bounds_fail_closed: bool,
        contract_catalog_bounds_fail_closed: bool,
        portable_schema_bounds_fail_closed: bool,
    }

    #[derive(Serialize)]
    struct NodeSchemaValidationArtifact {
        validation_id: &'static str,
        contract_catalog_sha256: String,
        registered_catalog_sha256: String,
        inactive_catalog_sha256: String,
        registered_rows: usize,
        inactive_rows: usize,
        normalized_v1: usize,
        normalized_v3: usize,
        executable_rows: usize,
        provider_required_rows: usize,
        environment: ValidationEnvironment<'static>,
        summary: NodeSchemaValidationSummary,
        cases: Vec<NodeSchemaValidationCase>,
        passed: usize,
        failed: usize,
        skipped: usize,
    }

    fn validation_target_directory() -> Result<std::path::PathBuf, Box<dyn Error>> {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        Ok(match std::env::var_os("CARGO_TARGET_DIR") {
            Some(directory) => {
                let directory = std::path::PathBuf::from(directory);
                if directory.is_absolute() {
                    directory
                } else {
                    workspace_root.join(directory)
                }
            }
            None => workspace_root.join("target"),
        })
    }

    fn write_validation_artifact(artifact: &ValidationArtifact<'_>) -> Result<(), Box<dyn Error>> {
        let target_directory = validation_target_directory()?;
        let artifact_directory = target_directory.join("sim-parity");
        fs::create_dir_all(&artifact_directory)?;
        let mut bytes = serde_json::to_vec_pretty(artifact)?;
        bytes.push(b'\n');
        fs::write(artifact_directory.join("val-node-registry-001.json"), bytes)?;
        Ok(())
    }

    fn write_node_schema_validation_artifact(
        artifact: &NodeSchemaValidationArtifact,
    ) -> Result<(), Box<dyn Error>> {
        let artifact_directory = validation_target_directory()?.join("sim-parity");
        fs::create_dir_all(&artifact_directory)?;
        let mut bytes = serde_json::to_vec_pretty(artifact)?;
        bytes.push(b'\n');
        fs::write(artifact_directory.join("val-node-001.json"), bytes)?;
        Ok(())
    }

    #[test]
    fn built_in_registry_calls_own_independent_catalog_maps() -> Result<(), Box<dyn Error>> {
        let original = NodeRegistry::built_in()?;
        let mut changed = NodeRegistry::built_in()?;
        assert!(!original.registered().is_empty());
        assert!(!original.inactive().is_empty());

        changed.registered.clear();
        changed.inactive.clear();

        assert!(changed.registered().is_empty());
        assert!(changed.inactive().is_empty());
        assert_eq!(NodeRegistry::built_in()?, original);
        Ok(())
    }

    #[test]
    fn checked_in_catalogs_round_trip_without_schema_loss() -> Result<(), Box<dyn Error>> {
        let registered = parse_csv(REGISTERED_NODE_CATALOG)?;
        let inactive = parse_csv(INACTIVE_NODE_CATALOG)?;
        assert_eq!(registered.len(), 790);
        assert_eq!(inactive.len(), 13);
        let registry = NodeRegistry::built_in()?;
        let (reconstructed_registered, reconstructed_inactive) =
            reconstruct_catalogs(&registry, &registered, &inactive)?;
        assert_eq!(reconstructed_registered, REGISTERED_NODE_CATALOG);
        assert_eq!(reconstructed_inactive, INACTIVE_NODE_CATALOG);
        Ok(())
    }

    #[test]
    fn registry_preserves_every_row_without_owning_runtime_executability()
    -> Result<(), Box<dyn Error>> {
        let registry = NodeRegistry::built_in()?;
        assert_eq!(registry.registered().len(), 789);
        assert_eq!(registry.inactive().len(), 12);
        assert_eq!(
            registry.registered()["LoadImage"].feature_id,
            "COMFY-NODE-0339"
        );
        assert_eq!(
            registry.registered()["LoadImage"].catalog_status,
            CatalogNodeStatus::DescriptorOnly
        );
        assert_eq!(registry.registered()["SaveImage"].output_node, true);
        assert!(
            registry
                .inactive()
                .values()
                .all(|descriptor| descriptor.catalog_status == CatalogNodeStatus::Inactive)
        );
        let provider_required = registry
            .registered()
            .values()
            .filter(|descriptor| descriptor.catalog_status == CatalogNodeStatus::ProviderRequired)
            .count();
        assert_eq!(provider_required, 224);
        Ok(())
    }

    #[test]
    fn registry_projects_exact_v1_v3_and_autogrow_source_schema() -> Result<(), Box<dyn Error>> {
        let registry = NodeRegistry::built_in()?;
        let sampler = registry
            .source_schema("KSampler")
            .ok_or("KSampler source schema is missing")?;
        let seed = sampler
            .inputs
            .iter()
            .find(|input| input.schema.name == "seed")
            .ok_or("KSampler seed schema is missing")?;
        assert_eq!(
            seed.schema.maximum,
            Some(NativeSchemaValue::UnsignedInteger { value: u64::MAX })
        );
        assert_eq!(
            sampler
                .inputs
                .iter()
                .find(|input| input.schema.name == "cfg")
                .and_then(|input| input.schema.step.as_ref()),
            Some(&NativeSchemaValue::FiniteDecimal {
                value: "0.1".to_owned()
            })
        );

        let batch = registry
            .source_schema("BatchImagesNode")
            .ok_or("BatchImagesNode source schema is missing")?;
        assert_eq!(batch.dynamic_inputs.len(), 1);
        assert_eq!(batch.dynamic_inputs[0].identity, "image{index}");
        assert_eq!(batch.dynamic_inputs[0].prefix.as_deref(), Some("image"));
        assert_eq!(batch.dynamic_inputs[0].minimum_count, 1);
        assert_eq!(batch.dynamic_inputs[0].maximum_count, 50);
        assert_eq!(
            batch.dynamic_inputs[0].input.source_type_names,
            ["IMAGE".to_owned()]
        );

        let open_ai = registry
            .source_schema("OpenAIGPTImage1")
            .ok_or("OpenAIGPTImage1 source schema is missing")?;
        let quality = open_ai
            .inputs
            .iter()
            .find(|input| input.schema.name == "quality")
            .ok_or("OpenAIGPTImage1 quality schema is missing")?;
        assert_eq!(
            quality.schema.choices,
            ["low", "medium", "high"]
                .into_iter()
                .map(|value| NativeSchemaValue::String {
                    value: value.to_owned()
                })
                .collect::<Vec<_>>()
        );
        assert!(open_ai.presentation.is_deprecated);
        Ok(())
    }

    #[test]
    fn malformed_or_duplicate_catalogs_are_rejected() -> Result<(), Box<dyn Error>> {
        assert!(matches!(
            parse_csv("a,b\n\"unterminated"),
            Err(NodeRegistryError::MalformedCsv { .. })
        ));
        let header = REGISTERED_NODE_CATALOG
            .lines()
            .next()
            .ok_or("registered catalog header is missing")?;
        let first = REGISTERED_NODE_CATALOG
            .lines()
            .nth(1)
            .ok_or("registered catalog row is missing")?;
        let duplicated = format!("{header}\n{first}\n{first}\n");
        assert!(matches!(
            NodeRegistryGenerator::from_catalogs(&duplicated, INACTIVE_NODE_CATALOG),
            Err(NodeRegistryError::DuplicateNode(_)) | Err(NodeRegistryError::DuplicateFeature(_))
        ));
        let mut registered = parse_csv(REGISTERED_NODE_CATALOG)?;
        for invalid_source in [
            "foreign/node.py",
            "comfy_extras/nested/node.py",
            "comfy_extras/node",
            "comfy_extras/.py",
            "comfy_extras/not-valid.py",
        ] {
            registered[1][23] = invalid_source.to_owned();
            let invalid_catalog = canonical_csv(&registered);
            assert!(matches!(
                NodeRegistryGenerator::from_catalogs(&invalid_catalog, INACTIVE_NODE_CATALOG),
                Err(NodeRegistryError::InvalidDescriptor {
                    field: "source_file",
                    ..
                })
            ));
        }
        let mut inactive = parse_csv(INACTIVE_NODE_CATALOG)?;
        inactive[1][11] = "comfy_api_nodes/nested/node.py".to_owned();
        let invalid_inactive_catalog = canonical_csv(&inactive);
        assert!(matches!(
            NodeRegistryGenerator::from_catalogs(
                REGISTERED_NODE_CATALOG,
                &invalid_inactive_catalog
            ),
            Err(NodeRegistryError::InvalidDescriptor {
                row: 2,
                field: "source_file",
            })
        ));
        Ok(())
    }

    struct CatalogTestNode(&'static str);

    impl NativeNode for CatalogTestNode {
        fn class_type(&self) -> &str {
            self.0
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            _context: NativeNodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
            Box::pin(async {
                Ok(NativeNodeOutcome::Blocked {
                    reason: "not executed by this metadata test".to_owned(),
                })
            })
        }
    }

    #[test]
    fn native_bindings_are_checked_against_atomic_catalog_presentation()
    -> Result<(), Box<dyn Error>> {
        let descriptor = NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: "wanBlockSwap".to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: Some(NativeDescriptorSchemaMetadata::synthetic(
                ["model".to_owned()],
                std::iter::empty(),
                ["model".to_owned()],
            )),
            inputs: vec![NativeInputDescriptor {
                name: "model".to_owned(),
                accepted_types: NativeTypeUnion::new([NativeValueType::Handle(
                    NativeHandleType::new(NativeHandleKind::Model, "MODEL")?,
                )])?,
                required: true,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal: false,
            }],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "model".to_owned(),
                produced_type: NativeValueType::Handle(NativeHandleType::new(
                    NativeHandleKind::Model,
                    "MODEL",
                )?),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        };
        let presentation = NativeNodePresentation {
            display_name: "wanBlockSwap".to_owned(),
            category: String::new(),
            description: "Intercepts an unstable custom node.".to_owned(),
            output_names: vec!["model".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: true,
            is_experimental: false,
        };
        let binding = NativeNodeBinding::Executable {
            feature_id: "COMFY-NODE-0757".to_owned(),
            descriptor: descriptor.clone(),
            presentation: presentation.clone(),
            node: std::sync::Arc::new(CatalogTestNode("wanBlockSwap")),
        };
        let registry = NodeRegistry::built_in()?;
        registry.validate_native_binding(&binding)?;

        let mismatched = NativeNodeBinding::Unavailable {
            feature_id: "COMFY-NODE-0757".to_owned(),
            descriptor,
            presentation,
            reason: "unavailable for this test".to_owned(),
        };
        assert!(matches!(
            registry.validate_native_binding(&mismatched),
            Err(NodeRegistryError::InvalidNativeBinding { reason, .. })
                if reason.contains("disposition")
        ));

        let image_type = NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?;
        let descriptor_only_descriptor = NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: "ImageInvert".to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: Some(NativeDescriptorSchemaMetadata::synthetic(
                ["image".to_owned()],
                std::iter::empty(),
                ["image".to_owned()],
            )),
            inputs: vec![NativeInputDescriptor {
                name: "image".to_owned(),
                accepted_types: NativeTypeUnion::new([NativeValueType::Handle(
                    image_type.clone(),
                )])?,
                required: true,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal: false,
            }],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "image".to_owned(),
                produced_type: NativeValueType::Handle(image_type),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        };
        let descriptor_only_presentation = NativeNodePresentation {
            display_name: "Invert Image Colors".to_owned(),
            category: "image/color".to_owned(),
            description: String::new(),
            output_names: vec!["image".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: false,
            is_experimental: false,
        };
        registry.validate_native_binding(&NativeNodeBinding::Executable {
            feature_id: "COMFY-NODE-0254".to_owned(),
            descriptor: descriptor_only_descriptor.clone(),
            presentation: descriptor_only_presentation.clone(),
            node: std::sync::Arc::new(CatalogTestNode("ImageInvert")),
        })?;
        assert!(matches!(
            registry.validate_native_binding(&NativeNodeBinding::Unavailable {
                feature_id: "COMFY-NODE-0254".to_owned(),
                descriptor: descriptor_only_descriptor,
                presentation: descriptor_only_presentation,
                reason: "wrong disposition".to_owned(),
            }),
            Err(NodeRegistryError::InvalidNativeBinding { reason, .. })
                if reason.contains("disposition")
        ));

        let provider_descriptor = NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: "BeebleSwitchXImageEdit".to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: Some(NativeDescriptorSchemaMetadata::synthetic(
                std::iter::empty(),
                std::iter::empty(),
                ["image".to_owned(), "alpha".to_owned()],
            )),
            inputs: Vec::new(),
            dynamic_inputs: Vec::new(),
            outputs: vec![
                NativeOutputDescriptor {
                    name: "image".to_owned(),
                    produced_type: NativeValueType::Handle(NativeHandleType::new(
                        NativeHandleKind::Image,
                        "IMAGE",
                    )?),
                    is_list: false,
                },
                NativeOutputDescriptor {
                    name: "alpha".to_owned(),
                    produced_type: NativeValueType::Handle(NativeHandleType::new(
                        NativeHandleKind::Mask,
                        "MASK",
                    )?),
                    is_list: false,
                },
            ],
            output_node: false,
            effect: NativeEffectClass::Provider,
            cache: NativeCachePolicy::Never,
        };
        let provider_presentation = NativeNodePresentation {
            display_name: "Beeble SwitchX Image Edit".to_owned(),
            category: "partner/image/Beeble".to_owned(),
            description: String::new(),
            output_names: vec!["image".to_owned(), "alpha".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: false,
            is_experimental: false,
        };
        registry.validate_native_binding(&NativeNodeBinding::ProviderRequired {
            feature_id: "COMFY-NODE-0020".to_owned(),
            descriptor: provider_descriptor.clone(),
            presentation: provider_presentation.clone(),
            provider: "sim-api".to_owned(),
            reason: "cloud provider authorization is required".to_owned(),
        })?;
        assert!(matches!(
            registry.validate_native_binding(&NativeNodeBinding::Unavailable {
                feature_id: "COMFY-NODE-0020".to_owned(),
                descriptor: provider_descriptor,
                presentation: provider_presentation,
                reason: "wrong disposition".to_owned(),
            }),
            Err(NodeRegistryError::InvalidNativeBinding { reason, .. })
                if reason.contains("disposition")
        ));

        let inactive_descriptor = NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: "AutogrowNamesTestNode".to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: Some(NativeDescriptorSchemaMetadata::synthetic(
                std::iter::empty(),
                std::iter::empty(),
                ["string".to_owned()],
            )),
            inputs: Vec::new(),
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "string".to_owned(),
                produced_type: NativeValueType::Primitive(crate::NativePrimitiveType::String),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::Never,
        };
        let inactive_presentation = NativeNodePresentation {
            display_name: "AutogrowNamesTest".to_owned(),
            category: "utilities/logic".to_owned(),
            description: String::new(),
            output_names: vec!["string".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: true,
            is_experimental: false,
        };
        registry.validate_native_binding(&NativeNodeBinding::Unavailable {
            feature_id: "COMFY-NODE-INACTIVE-0001".to_owned(),
            descriptor: inactive_descriptor.clone(),
            presentation: inactive_presentation.clone(),
            reason: "source node is inactive".to_owned(),
        })?;
        assert!(matches!(
            registry.validate_native_binding(&NativeNodeBinding::ProviderRequired {
                feature_id: "COMFY-NODE-INACTIVE-0001".to_owned(),
                descriptor: inactive_descriptor,
                presentation: inactive_presentation,
                provider: "invalid-provider".to_owned(),
                reason: "wrong disposition".to_owned(),
            }),
            Err(NodeRegistryError::InvalidNativeBinding { reason, .. })
                if reason.contains("disposition")
        ));
        Ok(())
    }

    #[test]
    fn val_node_001() -> Result<(), Box<dyn Error>> {
        let mut deserializer = serde_json::Deserializer::from_slice(NODE_CONTRACT_CATALOG);
        let catalog = NodeContractCatalogWire::deserialize(&mut deserializer)?;
        deserializer.end()?;
        let registry = NodeRegistry::built_in()?;
        let object_info = ObjectInfoRegistry::from_node_registry(&registry)?;
        let registered_rows = parse_csv(REGISTERED_NODE_CATALOG)?;
        let inactive_rows = parse_csv(INACTIVE_NODE_CATALOG)?;
        let (reconstructed_registered, reconstructed_inactive) =
            reconstruct_catalogs(&registry, &registered_rows, &inactive_rows)?;
        let catalog_csv_round_trip_is_exact = reconstructed_registered == REGISTERED_NODE_CATALOG
            && reconstructed_inactive == INACTIVE_NODE_CATALOG;
        let registered_rows_by_identifier = registered_rows
            .iter()
            .skip(1)
            .filter_map(|row| row.first().map(|identifier| (identifier.as_str(), row)))
            .collect::<BTreeMap<_, _>>();
        let mut cases = Vec::with_capacity(catalog.contracts.len() + registry.inactive().len());

        for (index, contract) in catalog.contracts.iter().enumerate() {
            let mut expected_schema = contract.schema.portable.clone();
            if contract.node_identifier == "ModelSamplingAuraFlow" {
                assert!(expected_schema.outputs.is_empty());
                expected_schema.outputs = catalog
                    .contracts
                    .iter()
                    .find(|contract| contract.node_identifier == "ModelSamplingSD3")
                    .ok_or("AuraFlow source superclass contract is absent")?
                    .schema
                    .portable
                    .outputs
                    .clone();
                assert_eq!(expected_schema.outputs.len(), 1);
                assert_eq!(expected_schema.outputs[0].source_type_name, "MODEL");
            }
            let descriptor = registry.registered().get(&contract.node_identifier);
            let schema = registry.source_schema(&contract.node_identifier);
            let expected_status = if contract.binding_disposition == "provider_required" {
                CatalogNodeStatus::ProviderRequired
            } else if contract.availability == "deprecated/dead" {
                CatalogNodeStatus::Inactive
            } else {
                CatalogNodeStatus::DescriptorOnly
            };
            let passed = descriptor.is_some_and(|descriptor| {
                descriptor.node_identifier == contract.node_identifier
                    && descriptor.feature_id == contract.feature_id
                    && descriptor.category == contract.category
                    && descriptor.classification == contract.classification
                    && descriptor.availability == contract.availability
                    && descriptor.schema_api.as_deref() == Some(contract.schema_api.as_str())
                    && descriptor.input_is_list == contract.input_is_list
                    && descriptor.lazy_inputs == contract.lazy_inputs
                    && descriptor.output_is_list == contract.output_is_list
                    && descriptor.output_node == contract.output_node
                    && descriptor.source_file == contract.source.path
                    && descriptor.source_symbol == contract.source.symbol.symbol
                    && descriptor.source_line == Some(contract.source.symbol.line)
                    && descriptor.catalog_status == expected_status
                    && registered_rows_by_identifier
                        .get(contract.node_identifier.as_str())
                        .is_some_and(|row| {
                            registered_catalog_row(descriptor).as_slice() == row.as_slice()
                        })
                    && object_info_matches_descriptor(
                        &registry,
                        &object_info,
                        &contract.node_identifier,
                        descriptor,
                    )
            }) && schema == Some(&expected_schema)
                && contract.schema.portable.validate().is_ok()
                && validate_contract_wire(index + 1, contract).is_ok();
            cases.push(NodeSchemaValidationCase {
                identifier: contract.node_identifier.clone(),
                feature_id: contract.feature_id.clone(),
                source_kind: contract.schema_api.clone(),
                binding_disposition: contract.binding_disposition.clone(),
                input_count: contract.schema.portable.inputs.len(),
                dynamic_input_count: contract.schema.portable.dynamic_inputs.len(),
                output_count: expected_schema.outputs.len(),
                definition_sha256: contract.schema.definition_sha256.clone(),
                source_sha256: Some(contract.source.sha256.clone()),
                passed,
            });
        }

        for row in inactive_rows.iter().skip(1) {
            let identifier = row.first().ok_or("inactive row has no identifier")?;
            let descriptor = registry
                .inactive()
                .get(identifier)
                .ok_or("inactive descriptor is absent")?;
            let passed = descriptor.catalog_status == CatalogNodeStatus::Inactive
                && descriptor.source == CatalogNodeSource::Inactive
                && registry.source_schema(identifier).is_none()
                && inactive_catalog_row(descriptor).as_slice() == row.as_slice()
                && object_info_matches_descriptor(&registry, &object_info, identifier, descriptor);
            cases.push(NodeSchemaValidationCase {
                identifier: descriptor.node_identifier.clone(),
                feature_id: descriptor.feature_id.clone(),
                source_kind: "inactive".to_owned(),
                binding_disposition: "inactive".to_owned(),
                input_count: 0,
                dynamic_input_count: 0,
                output_count: 0,
                definition_sha256: sha256_hex(descriptor.schema_source.as_bytes()),
                source_sha256: None,
                passed,
            });
        }

        let catalog_identifiers = catalog
            .contracts
            .iter()
            .map(|contract| contract.node_identifier.as_str())
            .collect::<BTreeSet<_>>();
        let registered_identifiers = registry
            .registered()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let registered_contracts_are_exact = registry.registered().len() == 789
            && catalog.contracts.len() == 789
            && catalog_identifiers == registered_identifiers
            && cases
                .iter()
                .take(catalog.contracts.len())
                .all(|case| case.passed);
        let inactive_descriptors_are_exact = registry.inactive().len() == 12
            && cases
                .iter()
                .skip(catalog.contracts.len())
                .all(|case| case.passed);
        let object_info_projection_is_exact = object_info.nodes().len()
            == registry.registered().len() + registry.inactive().len()
            && registry.registered().iter().chain(registry.inactive()).all(
                |(identifier, descriptor)| {
                    object_info_matches_descriptor(&registry, &object_info, identifier, descriptor)
                },
            );
        let oversized_field = "x".repeat(MAX_FIELD_BYTES + 1);
        let csv_parser_bounds_fail_closed = matches!(
            parse_csv("a,b\n\"unterminated"),
            Err(NodeRegistryError::MalformedCsv { .. })
        ) && matches!(
            parse_csv(&format!("field\n{oversized_field}\n")),
            Err(NodeRegistryError::FieldTooLarge { .. })
        );
        let mut trailing_contract_catalog = NODE_CONTRACT_CATALOG.to_vec();
        trailing_contract_catalog.extend_from_slice(b"{}");
        let oversized_contract_catalog = vec![b' '; MAX_CATALOG_BYTES + 1];
        let mut duplicate_contract_catalog: serde_json::Value =
            serde_json::from_slice(NODE_CONTRACT_CATALOG)?;
        let contracts = duplicate_contract_catalog
            .get_mut("contracts")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("node contract catalog has no contracts array")?;
        let duplicate_identifier = contracts
            .first()
            .and_then(|contract| contract.get("node_identifier"))
            .and_then(serde_json::Value::as_str)
            .ok_or("first node contract has no identifier")?
            .to_owned();
        let second_contract = contracts
            .get_mut(1)
            .and_then(serde_json::Value::as_object_mut)
            .ok_or("node contract catalog has no second contract")?;
        second_contract.insert(
            "node_identifier".to_owned(),
            serde_json::Value::String(duplicate_identifier),
        );
        let duplicate_contract_catalog = serde_json::to_vec(&duplicate_contract_catalog)?;
        let contract_catalog_bounds_fail_closed =
            matches!(
                parse_contract_schemas(&oversized_contract_catalog),
                Err(NodeRegistryError::CatalogTooLarge)
            ) && parse_contract_schemas(b"{\"unknown\":true}").is_err()
                && parse_contract_schemas(&trailing_contract_catalog).is_err()
                && matches!(
                    parse_contract_schemas(&duplicate_contract_catalog),
                    Err(NodeRegistryError::InvalidDescriptor {
                        field: "node_contract_identity",
                        ..
                    })
                );
        let first_input = catalog
            .contracts
            .iter()
            .find_map(|contract| contract.schema.portable.inputs.first())
            .ok_or("node contract catalog has no input schema")?;
        let base_schema = catalog
            .contracts
            .first()
            .ok_or("node contract catalog is empty")?
            .schema
            .portable
            .clone();
        let mut oversized_schema = base_schema.clone();
        oversized_schema.inputs = vec![first_input.clone(); 4_097];
        let item_bound_fails = matches!(
            oversized_schema.validate(),
            Err(NativeSchemaError::ItemCountExceeded)
        );
        let mut nested_value = NativeSchemaValue::Null;
        for _ in 0..18 {
            nested_value = NativeSchemaValue::List {
                values: vec![nested_value],
            };
        }
        let mut deep_schema = base_schema.clone();
        deep_schema.hidden = vec![nested_value];
        let depth_bound_fails = matches!(
            deep_schema.validate(),
            Err(NativeSchemaError::DepthExceeded)
        );
        let mut text_schema = base_schema.clone();
        text_schema.presentation.description = Some("x".repeat(256 * 1024 + 1));
        let text_bound_fails =
            matches!(text_schema.validate(), Err(NativeSchemaError::TextTooLarge));
        let mut total_schema = base_schema.clone();
        total_schema.inputs = (0..9)
            .map(|index| {
                let mut input = first_input.clone();
                input.schema.name = format!("aggregate_value_{index}");
                input.schema.choices = vec![NativeSchemaValue::String {
                    value: "x".repeat(250_000),
                }];
                input
            })
            .collect();
        let total_bound_fails = matches!(
            total_schema.validate(),
            Err(NativeSchemaError::TotalBytesExceeded)
        );
        let mut preserved_expression_schema = base_schema;
        preserved_expression_schema.hidden = (0..4_095)
            .map(|_| NativeSchemaValue::PreservedExpression {
                source: "x".repeat(450),
                sha256: "a".repeat(64),
            })
            .collect();
        let preserved_expression_bound_fails = matches!(
            preserved_expression_schema.validate(),
            Err(NativeSchemaError::TotalBytesExceeded)
        );
        let portable_schema_bounds_fail_closed = item_bound_fails
            && depth_bound_fails
            && text_bound_fails
            && total_bound_fails
            && preserved_expression_bound_fails;
        assert!(catalog_csv_round_trip_is_exact);
        let failed_registered = cases
            .iter()
            .take(catalog.contracts.len())
            .filter(|case| !case.passed)
            .map(|case| case.identifier.as_str())
            .collect::<Vec<_>>();
        assert!(
            registered_contracts_are_exact,
            "registered schema correlations failed for {failed_registered:?}"
        );
        assert!(inactive_descriptors_are_exact);
        assert!(object_info_projection_is_exact);
        assert!(csv_parser_bounds_fail_closed);
        assert!(contract_catalog_bounds_fail_closed);
        assert!(portable_schema_bounds_fail_closed);

        let summary_passed = [
            catalog_csv_round_trip_is_exact,
            registered_contracts_are_exact,
            inactive_descriptors_are_exact,
            object_info_projection_is_exact,
            csv_parser_bounds_fail_closed,
            contract_catalog_bounds_fail_closed,
            portable_schema_bounds_fail_closed,
        ]
        .into_iter()
        .filter(|passed| *passed)
        .count();
        let row_passed = cases.iter().filter(|case| case.passed).count();
        let row_failed = cases.len() - row_passed;
        write_node_schema_validation_artifact(&NodeSchemaValidationArtifact {
            validation_id: "VAL-NODE-001",
            contract_catalog_sha256: sha256_hex(NODE_CONTRACT_CATALOG),
            registered_catalog_sha256: sha256_hex(REGISTERED_NODE_CATALOG.as_bytes()),
            inactive_catalog_sha256: sha256_hex(INACTIVE_NODE_CATALOG.as_bytes()),
            registered_rows: registry.registered().len(),
            inactive_rows: registry.inactive().len(),
            normalized_v1: catalog.summary.normalized_v1,
            normalized_v3: catalog.summary.normalized_v3,
            executable_rows: catalog.summary.executable,
            provider_required_rows: catalog.summary.provider_required,
            environment: ValidationEnvironment {
                operating_system: std::env::consts::OS,
                architecture: std::env::consts::ARCH,
                backend_identity: "portable-source-schema-v2",
                registry_role: "schema-v2 source projection and read-only object-info",
            },
            summary: NodeSchemaValidationSummary {
                catalog_csv_round_trip_is_exact,
                registered_contracts_are_exact,
                inactive_descriptors_are_exact,
                object_info_projection_is_exact,
                csv_parser_bounds_fail_closed,
                contract_catalog_bounds_fail_closed,
                portable_schema_bounds_fail_closed,
            },
            passed: row_passed + summary_passed,
            failed: row_failed + 7 - summary_passed,
            skipped: 0,
            cases,
        })?;
        Ok(())
    }

    #[test]
    fn val_node_registry_001() -> Result<(), Box<dyn Error>> {
        let registered_rows = parse_csv(REGISTERED_NODE_CATALOG)?;
        let inactive_rows = parse_csv(INACTIVE_NODE_CATALOG)?;
        let registry = NodeRegistry::built_in()?;
        let (reconstructed_registered, reconstructed_inactive) =
            reconstruct_catalogs(&registry, &registered_rows, &inactive_rows)?;
        let catalog_round_trip = registered_rows.len() == 790
            && inactive_rows.len() == 13
            && reconstructed_registered == REGISTERED_NODE_CATALOG
            && reconstructed_inactive == INACTIVE_NODE_CATALOG;
        assert!(catalog_round_trip);

        let catalog_status_is_read_only = registry.registered().values().all(|descriptor| {
            matches!(
                descriptor.catalog_status,
                CatalogNodeStatus::DescriptorOnly
                    | CatalogNodeStatus::ProviderRequired
                    | CatalogNodeStatus::Inactive
            )
        }) && registry
            .inactive()
            .values()
            .all(|descriptor| descriptor.catalog_status == CatalogNodeStatus::Inactive);
        assert!(catalog_status_is_read_only);

        let generated_slice_modules = crate::GENERATED_MODULES
            .iter()
            .copied()
            .filter(|module| module.starts_with("slices/"))
            .collect::<Vec<_>>();
        let generated_family_modules = crate::GENERATED_MODULES
            .iter()
            .copied()
            .filter(|module| module.starts_with("families/"))
            .collect::<Vec<_>>();
        let slice_descriptor_ids = crate::generated_native_image::NODE_DESCRIPTOR_IDS
            .iter()
            .chain(crate::generated_native_diffusion::NODE_DESCRIPTOR_IDS)
            .copied()
            .collect::<BTreeSet<_>>();
        let family_descriptor_ids = crate::GENERATED_FAMILY_DESCRIPTOR_IDS
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let generated_descriptor_ids = crate::GENERATED_DESCRIPTOR_IDS
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let generated_descriptor_membership_is_exact = generated_slice_modules
            == ["slices/native_diffusion", "slices/native_image"]
            && generated_family_modules == crate::GENERATED_FAMILY_MODULES
            && crate::GENERATED_MODULES
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            && crate::GENERATED_DESCRIPTOR_IDS
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            && crate::GENERATED_FAMILY_DESCRIPTOR_IDS
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            && slice_descriptor_ids.is_disjoint(&family_descriptor_ids)
            && generated_descriptor_ids
                == slice_descriptor_ids
                    .union(&family_descriptor_ids)
                    .copied()
                    .collect()
            && crate::GENERATED_DESCRIPTOR_IDS
                .iter()
                .all(|identifier| registry.registered().contains_key(*identifier));
        assert!(generated_descriptor_membership_is_exact);

        let slices = EarlySliceRegistry::from_node_registry(&registry)?;
        let early_slice_membership_is_exact = slices.image()
            == IMAGE_SLICE_NODE_IDS
                .iter()
                .map(|identifier| (*identifier).to_owned())
                .collect::<Vec<_>>()
            && slices.diffusion()
                == DIFFUSION_SLICE_NODE_IDS
                    .iter()
                    .map(|identifier| (*identifier).to_owned())
                    .collect::<Vec<_>>();
        assert!(early_slice_membership_is_exact);

        let object_info = ObjectInfoRegistry::from_node_registry(&registry)?;
        let object_info_is_read_only_projection =
            registry.registered().iter().chain(registry.inactive()).all(
                |(identifier, descriptor)| {
                    object_info
                        .nodes()
                        .get(identifier)
                        .is_some_and(|projected| {
                            projected.node_identifier == descriptor.node_identifier
                                && projected.display_name == descriptor.display_name
                                && projected.category == descriptor.category
                                && projected.source_python_module
                                    == registry
                                        .source_python_module(identifier)
                                        .unwrap_or_default()
                                && projected.schema_source == descriptor.schema_source
                                && projected.input.raw == descriptor.inputs
                                && projected.input.input_is_list == descriptor.input_is_list
                                && projected.input.lazy_inputs == descriptor.lazy_inputs
                                && projected.output.raw == descriptor.outputs
                                && projected.output.output_is_list == descriptor.output_is_list
                                && projected.output.output_node == descriptor.output_node
                                && projected.availability == descriptor.availability
                                && projected.catalog_status == descriptor.catalog_status
                                && projected.inactive_reason == descriptor.inactive_reason
                                && projected.feature_id == descriptor.feature_id
                        })
                },
            );
        assert!(object_info_is_read_only_projection);

        let registry_source = include_str!("registry_generator.rs");
        let descriptor_source = include_str!("descriptor.rs");
        let mutable_plugin_and_execution_apis_are_absent = !registry_source
            .contains(&["register", "_plugin_nodes"].concat())
            && !registry_source.contains(&["plugin", "_nodes:"].concat())
            && !registry_source.contains(&["is", "_executable"].concat())
            && !descriptor_source.contains(&["NodeExecution", "Status"].concat())
            && !descriptor_source.contains(&["Signed", "Plugin"].concat());
        assert!(mutable_plugin_and_execution_apis_are_absent);

        let mut generated_descriptor_ids_bytes =
            crate::GENERATED_DESCRIPTOR_IDS.join("\n").into_bytes();
        generated_descriptor_ids_bytes.push(b'\n');

        write_validation_artifact(&ValidationArtifact {
            validation_id: "VAL-NODE-REGISTRY-001",
            registered_rows: registry.registered().len(),
            inactive_rows: registry.inactive().len(),
            registered_catalog_sha256: sha256_hex(REGISTERED_NODE_CATALOG.as_bytes()),
            inactive_catalog_sha256: sha256_hex(INACTIVE_NODE_CATALOG.as_bytes()),
            generated_descriptor_ids_sha256: sha256_hex(&generated_descriptor_ids_bytes),
            generated_descriptor_ids_encoding: "utf8-lines-lf",
            generated_descriptor_ids: crate::GENERATED_DESCRIPTOR_IDS,
            image_slice: IMAGE_SLICE_NODE_IDS,
            diffusion_slice: DIFFUSION_SLICE_NODE_IDS,
            environment: ValidationEnvironment {
                operating_system: std::env::consts::OS,
                architecture: std::env::consts::ARCH,
                backend_identity: "read-only-native-node-registry",
                registry_role: "read-side compatibility descriptors only",
            },
            cases: ValidationCases {
                catalog_round_trip,
                catalog_status_is_read_only,
                generated_descriptor_membership_is_exact,
                early_slice_membership_is_exact,
                object_info_is_read_only_projection,
                mutable_plugin_and_execution_apis_are_absent,
            },
            passed: 6,
            failed: 0,
            skipped: 0,
        })?;
        Ok(())
    }
}
